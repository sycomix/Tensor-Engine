use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
#[cfg(feature = "parallel")]
use std::sync::OnceLock;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const ELEMENTWISE_PAR_MIN_LEN_DEFAULT: usize = 16_384;
#[cfg(feature = "parallel")]
const BROADCAST_BACKWARD_PAR_MIN_LEN_DEFAULT: usize = 16_384;
#[cfg(feature = "parallel")]
const MATMUL_PAR_MIN_WORK_DEFAULT: usize = 300_000;

#[cfg(feature = "parallel")]
static ELEMENTWISE_PAR_MIN_LEN: OnceLock<usize> = OnceLock::new();
#[cfg(feature = "parallel")]
static BROADCAST_BACKWARD_PAR_MIN_LEN: OnceLock<usize> = OnceLock::new();
#[cfg(feature = "parallel")]
static MATMUL_PAR_MIN_WORK: OnceLock<usize> = OnceLock::new();

#[derive(Debug, Clone, PartialEq)]
pub enum AutogradError {
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        op: &'static str,
    },
    InvalidRank {
        expected: usize,
        found: usize,
        op: &'static str,
    },
    MatMulDimMismatch {
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        lhs_inner: usize,
        rhs_inner: usize,
    },
    DomainError {
        op: &'static str,
        index: usize,
        value: f32,
    },
    NonScalarBackward {
        shape: Vec<usize>,
    },
    BackwardGradLenMismatch {
        expected: usize,
        found: usize,
    },
    GraphCycleDetected,
    DivisionByZero {
        index: usize,
    },
}

impl Display for AutogradError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AutogradError::ShapeMismatch { lhs, rhs, op } => {
                write!(f, "shape mismatch for {}: lhs={:?} rhs={:?}", op, lhs, rhs)
            }
            AutogradError::InvalidRank {
                expected,
                found,
                op,
            } => write!(
                f,
                "invalid rank for {}: expected {}, found {}",
                op, expected, found
            ),
            AutogradError::MatMulDimMismatch {
                lhs_shape,
                rhs_shape,
                lhs_inner,
                rhs_inner,
            } => write!(
                f,
                "matmul incompatible inner dimensions: lhs={:?} rhs={:?} ({} != {})",
                lhs_shape, rhs_shape, lhs_inner, rhs_inner
            ),
            AutogradError::DomainError { op, index, value } => write!(
                f,
                "domain error in {} at index {} for value {}",
                op, index, value
            ),
            AutogradError::NonScalarBackward { shape } => {
                write!(f, "backward() requires scalar output, got {:?}", shape)
            }
            AutogradError::BackwardGradLenMismatch { expected, found } => write!(
                f,
                "backward grad_output length mismatch: expected {}, found {}",
                expected, found
            ),
            AutogradError::GraphCycleDetected => {
                write!(f, "graph cycle detected; autograd graph must be a DAG")
            }
            AutogradError::DivisionByZero { index } => {
                write!(f, "division by zero at index {}", index)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Op {
    Leaf,
    Add,
    Mul,
    Div,
    MatMul,
    Transpose2D,
    ReLU,
    Tanh,
    Sigmoid,
    Exp,
    Log,
    Sqrt,
    Sum,
    SumLastDim,
    SoftmaxLastDim,
    CausalMaskUpper,
    Mean,
}

#[derive(Clone)]
pub struct Tensor {
    node: Rc<RefCell<Node>>,
}

#[derive(Clone)]
struct Node {
    data: Vec<f32>,
    grad: Vec<f32>,
    shape: Vec<usize>,
    requires_grad: bool,
    parents: Vec<Tensor>,
    op: Op,
}

impl Tensor {
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let size = shape.iter().product::<usize>();
        assert_eq!(data.len(), size, "data size must equal shape product");

        Self {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; data.len()],
                data,
                shape,
                requires_grad,
                parents: Vec::new(),
                op: Op::Leaf,
            })),
        }
    }

    pub fn data(&self) -> Vec<f32> {
        self.node.borrow().data.clone()
    }

    pub fn data_ref(&self) -> Ref<'_, [f32]> {
        Ref::map(self.node.borrow(), |n| n.data.as_slice())
    }

    pub fn with_data<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let n = self.node.borrow();
        f(&n.data)
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        let node = self.node.borrow();
        if node.requires_grad {
            Some(node.grad.clone())
        } else {
            None
        }
    }

    pub fn grad_ref(&self) -> Option<Ref<'_, [f32]>> {
        let node = self.node.borrow();
        if !node.requires_grad {
            return None;
        }
        Some(Ref::map(node, |n| n.grad.as_slice()))
    }

    pub fn with_grad<R>(&self, f: impl FnOnce(Option<&[f32]>) -> R) -> R {
        let n = self.node.borrow();
        if n.requires_grad {
            f(Some(&n.grad))
        } else {
            f(None)
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.node.borrow().shape.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.node.borrow().requires_grad
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        let b = other.node.borrow();
        let a_shape = a.shape.clone();
        let b_shape = b.shape.clone();
        let out_shape = broadcast_shape(&a_shape, &b_shape, "add")?;
        let out_len = numel(&out_shape);
        let out_strides = strides(&out_shape);
        let a_input_strides = strides(&a_shape);
        let b_input_strides = strides(&b_shape);
        let a_leading = out_shape.len().saturating_sub(a_shape.len());
        let b_leading = out_shape.len().saturating_sub(b_shape.len());

        #[cfg(feature = "parallel")]
        let a_data = a.data.clone();
        #[cfg(feature = "parallel")]
        let b_data = b.data.clone();

        let mut out = vec![0.0; out_len];
        #[cfg(feature = "parallel")]
        if should_parallel_elementwise(out_len) {
            out.par_iter_mut().enumerate().for_each(|(out_idx, dst)| {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                *dst = a_data[a_idx] + b_data[b_idx];
            });
        } else {
            for (out_idx, dst) in out.iter_mut().enumerate() {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                *dst = a_data[a_idx] + b_data[b_idx];
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (out_idx, dst) in out.iter_mut().enumerate() {
            let a_idx = broadcasted_input_linear_index_precomputed(
                out_idx,
                &out_shape,
                &out_strides,
                &a_shape,
                &a_input_strides,
                a_leading,
            );
            let b_idx = broadcasted_input_linear_index_precomputed(
                out_idx,
                &out_shape,
                &out_strides,
                &b_shape,
                &b_input_strides,
                b_leading,
            );
            *dst = a.data[a_idx] + b.data[b_idx];
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: out_shape,
                requires_grad: a.requires_grad || b.requires_grad,
                parents: vec![self.clone(), other.clone()],
                op: Op::Add,
            })),
        })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        let b = other.node.borrow();
        let a_shape = a.shape.clone();
        let b_shape = b.shape.clone();
        let out_shape = broadcast_shape(&a_shape, &b_shape, "mul")?;
        let out_len = numel(&out_shape);
        let out_strides = strides(&out_shape);
        let a_input_strides = strides(&a_shape);
        let b_input_strides = strides(&b_shape);
        let a_leading = out_shape.len().saturating_sub(a_shape.len());
        let b_leading = out_shape.len().saturating_sub(b_shape.len());

        #[cfg(feature = "parallel")]
        let a_data = a.data.clone();
        #[cfg(feature = "parallel")]
        let b_data = b.data.clone();

        let mut out = vec![0.0; out_len];
        #[cfg(feature = "parallel")]
        if should_parallel_elementwise(out_len) {
            out.par_iter_mut().enumerate().for_each(|(out_idx, dst)| {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                *dst = a_data[a_idx] * b_data[b_idx];
            });
        } else {
            for (out_idx, dst) in out.iter_mut().enumerate() {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                *dst = a_data[a_idx] * b_data[b_idx];
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (out_idx, dst) in out.iter_mut().enumerate() {
            let a_idx = broadcasted_input_linear_index_precomputed(
                out_idx,
                &out_shape,
                &out_strides,
                &a_shape,
                &a_input_strides,
                a_leading,
            );
            let b_idx = broadcasted_input_linear_index_precomputed(
                out_idx,
                &out_shape,
                &out_strides,
                &b_shape,
                &b_input_strides,
                b_leading,
            );
            *dst = a.data[a_idx] * b.data[b_idx];
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: out_shape,
                requires_grad: a.requires_grad || b.requires_grad,
                parents: vec![self.clone(), other.clone()],
                op: Op::Mul,
            })),
        })
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        let b = other.node.borrow();
        let a_shape = a.shape.clone();
        let b_shape = b.shape.clone();
        let out_shape = broadcast_shape(&a_shape, &b_shape, "div")?;
        let out_len = numel(&out_shape);
        let out_strides = strides(&out_shape);
        let a_input_strides = strides(&a_shape);
        let b_input_strides = strides(&b_shape);
        let a_leading = out_shape.len().saturating_sub(a_shape.len());
        let b_leading = out_shape.len().saturating_sub(b_shape.len());

        #[cfg(feature = "parallel")]
        let a_data = a.data.clone();
        #[cfg(feature = "parallel")]
        let b_data = b.data.clone();

        let mut out = vec![0.0; out_len];
        #[cfg(feature = "parallel")]
        if should_parallel_elementwise(out_len) {
            out.par_iter_mut().enumerate().try_for_each(|(out_idx, dst)| {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                let denom = b_data[b_idx];
                if denom == 0.0 {
                    return Err(AutogradError::DivisionByZero { index: out_idx });
                }
                *dst = a_data[a_idx] / denom;
                Ok::<(), AutogradError>(())
            })?;
        } else {
            for (out_idx, dst) in out.iter_mut().enumerate() {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                let denom = b_data[b_idx];
                if denom == 0.0 {
                    return Err(AutogradError::DivisionByZero { index: out_idx });
                }
                *dst = a_data[a_idx] / denom;
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (out_idx, dst) in out.iter_mut().enumerate() {
            let a_idx = broadcasted_input_linear_index_precomputed(
                out_idx,
                &out_shape,
                &out_strides,
                &a_shape,
                &a_input_strides,
                a_leading,
            );
            let b_idx = broadcasted_input_linear_index_precomputed(
                out_idx,
                &out_shape,
                &out_strides,
                &b_shape,
                &b_input_strides,
                b_leading,
            );
            let denom = b.data[b_idx];
            if denom == 0.0 {
                return Err(AutogradError::DivisionByZero { index: out_idx });
            }
            *dst = a.data[a_idx] / denom;
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: out_shape,
                requires_grad: a.requires_grad || b.requires_grad,
                parents: vec![self.clone(), other.clone()],
                op: Op::Div,
            })),
        })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        let b = other.node.borrow();
        if a.shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: 2,
                found: a.shape.len(),
                op: "matmul(lhs)",
            });
        }
        if b.shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: 2,
                found: b.shape.len(),
                op: "matmul(rhs)",
            });
        }

        let m = a.shape[0];
        let n = a.shape[1];
        let n2 = b.shape[0];
        let p = b.shape[1];
        if n != n2 {
            return Err(AutogradError::MatMulDimMismatch {
                lhs_shape: a.shape.clone(),
                rhs_shape: b.shape.clone(),
                lhs_inner: n,
                rhs_inner: n2,
            });
        }

        #[cfg(feature = "parallel")]
        let a_data = a.data.clone();
        #[cfg(feature = "parallel")]
        let b_data = b.data.clone();

        let mut out = vec![0.0; m * p];
        #[cfg(feature = "parallel")]
        if should_parallel_matmul(m, n, p) {
            out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
                for (k, dst) in row.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += a_data[i * n + j] * b_data[j * p + k];
                    }
                    *dst = acc;
                }
            });
        } else {
            for i in 0..m {
                for k in 0..p {
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += a_data[i * n + j] * b_data[j * p + k];
                    }
                    out[i * p + k] = acc;
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..m {
            for k in 0..p {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += a.data[i * n + j] * b.data[j * p + k];
                }
                out[i * p + k] = acc;
            }
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: vec![m, p],
                requires_grad: a.requires_grad || b.requires_grad,
                parents: vec![self.clone(), other.clone()],
                op: Op::MatMul,
            })),
        })
    }

    pub fn transpose2d(&self) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        if a.shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: 2,
                found: a.shape.len(),
                op: "transpose2d",
            });
        }

        let rows = a.shape[0];
        let cols = a.shape[1];
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = a.data[r * cols + c];
            }
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: vec![cols, rows],
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Transpose2D,
            })),
        })
    }

    pub fn relu(&self) -> Tensor {
        let a = self.node.borrow();
        let mut out = vec![0.0; a.data.len()];
        for (i, val) in a.data.iter().enumerate() {
            out[i] = if *val > 0.0 { *val } else { 0.0 };
        }

        Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::ReLU,
            })),
        }
    }

    pub fn tanh(&self) -> Tensor {
        let a = self.node.borrow();
        let mut out = vec![0.0; a.data.len()];
        for (i, val) in a.data.iter().enumerate() {
            out[i] = val.tanh();
        }

        Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Tanh,
            })),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let a = self.node.borrow();
        let mut out = vec![0.0; a.data.len()];
        for (i, val) in a.data.iter().enumerate() {
            out[i] = 1.0 / (1.0 + (-*val).exp());
        }

        Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Sigmoid,
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
        let a = self.node.borrow();
        let mut out = vec![0.0; a.data.len()];
        for (i, val) in a.data.iter().enumerate() {
            out[i] = val.exp();
        }

        Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Exp,
            })),
        }
    }

    pub fn log(&self) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        let mut out = vec![0.0; a.data.len()];
        for (i, val) in a.data.iter().enumerate() {
            if *val <= 0.0 {
                return Err(AutogradError::DomainError {
                    op: "log",
                    index: i,
                    value: *val,
                });
            }
            out[i] = val.ln();
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Log,
            })),
        })
    }

    pub fn sqrt(&self) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        let mut out = vec![0.0; a.data.len()];
        for (i, val) in a.data.iter().enumerate() {
            if *val < 0.0 {
                return Err(AutogradError::DomainError {
                    op: "sqrt",
                    index: i,
                    value: *val,
                });
            }
            out[i] = val.sqrt();
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Sqrt,
            })),
        })
    }

    pub fn sum(&self) -> Tensor {
        let a = self.node.borrow();
        let total: f32 = a.data.iter().sum();

        Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; 1],
                data: vec![total],
                shape: vec![1],
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Sum,
            })),
        }
    }

    pub fn sum_last_dim(&self) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        if a.shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: 2,
                found: a.shape.len(),
                op: "sum_last_dim",
            });
        }

        let rows = a.shape[0];
        let cols = a.shape[1];
        let mut out = vec![0.0; rows];
        for r in 0..rows {
            let mut acc = 0.0;
            for c in 0..cols {
                acc += a.data[r * cols + c];
            }
            out[r] = acc;
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: vec![rows, 1],
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::SumLastDim,
            })),
        })
    }

    pub fn softmax_last_dim(&self) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        if a.shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: 2,
                found: a.shape.len(),
                op: "softmax_last_dim",
            });
        }

        let rows = a.shape[0];
        let cols = a.shape[1];
        let mut out = vec![0.0; rows * cols];

        for r in 0..rows {
            let row_start = r * cols;
            let row = &a.data[row_start..row_start + cols];
            let mut max_v = f32::NEG_INFINITY;
            for &v in row {
                if v > max_v {
                    max_v = v;
                }
            }

            let mut denom = 0.0;
            for c in 0..cols {
                let e = (row[c] - max_v).exp();
                out[row_start + c] = e;
                denom += e;
            }

            for c in 0..cols {
                out[row_start + c] /= denom;
            }
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::SoftmaxLastDim,
            })),
        })
    }

    pub fn causal_mask_upper(&self, mask_value: f32) -> Result<Tensor, AutogradError> {
        let a = self.node.borrow();
        if a.shape.len() != 2 {
            return Err(AutogradError::InvalidRank {
                expected: 2,
                found: a.shape.len(),
                op: "causal_mask_upper",
            });
        }

        let rows = a.shape[0];
        let cols = a.shape[1];
        let mut out = a.data.clone();
        for r in 0..rows {
            for c in 0..cols {
                if c > r {
                    out[r * cols + c] = mask_value;
                }
            }
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; out.len()],
                data: out,
                shape: a.shape.clone(),
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::CausalMaskUpper,
            })),
        })
    }

    pub fn mean(&self) -> Tensor {
        let a = self.node.borrow();
        let total: f32 = a.data.iter().sum();
        let m = a.data.len() as f32;

        Tensor {
            node: Rc::new(RefCell::new(Node {
                grad: vec![0.0; 1],
                data: vec![total / m],
                shape: vec![1],
                requires_grad: a.requires_grad,
                parents: vec![self.clone()],
                op: Op::Mean,
            })),
        }
    }

    pub fn zero_grad(&self) {
        if let Ok(topo) = topo_sort_checked(self) {
            for t in topo {
                let mut n = t.node.borrow_mut();
                for g in &mut n.grad {
                    *g = 0.0;
                }
            }
        }
    }

    pub fn backward(&self) -> Result<(), AutogradError> {
        if self.shape() != vec![1] {
            return Err(AutogradError::NonScalarBackward {
                shape: self.shape(),
            });
        }

        self.backward_with_grad(&[1.0])
    }

    pub fn backward_with_grad(&self, grad_output: &[f32]) -> Result<(), AutogradError> {
        let expected = self.node.borrow().data.len();
        if grad_output.len() != expected {
            return Err(AutogradError::BackwardGradLenMismatch {
                expected,
                found: grad_output.len(),
            });
        }

        let topo = topo_sort_checked(self)?;
        for t in &topo {
            let mut n = t.node.borrow_mut();
            for g in &mut n.grad {
                *g = 0.0;
            }
        }

        {
            let mut root = self.node.borrow_mut();
            root.grad.clone_from_slice(grad_output);
        }

        for t in topo.into_iter().rev() {
            backward_step(&t);
        }

        Ok(())
    }
}

fn backward_step(t: &Tensor) {
    let (op, parents, grad_out) = {
        let n = t.node.borrow();
        (n.op.clone(), n.parents.clone(), n.grad.clone())
    };

    match op {
        Op::Leaf => {}
        Op::Add => {
            let a_shape = parents[0].node.borrow().shape.clone();
            let b_shape = parents[1].node.borrow().shape.clone();
            let out_shape = t.node.borrow().shape.clone();
            let out_strides = strides(&out_shape);
            let a_input_strides = strides(&a_shape);
            let b_input_strides = strides(&b_shape);
            let a_leading = out_shape.len().saturating_sub(a_shape.len());
            let b_leading = out_shape.len().saturating_sub(b_shape.len());

            let mut grad_a = vec![0.0; numel(&a_shape)];
            let mut grad_b = vec![0.0; numel(&b_shape)];

            #[cfg(feature = "parallel")]
            {
                if should_parallel_broadcast_backward(grad_out.len()) {
                    let grad_a_len = grad_a.len();
                    let grad_b_len = grad_b.len();
                    let (par_a, par_b) = (0..grad_out.len())
                        .into_par_iter()
                        .fold(
                            || (vec![0.0; grad_a_len], vec![0.0; grad_b_len]),
                            |(mut local_a, mut local_b), out_idx| {
                                let g = grad_out[out_idx];
                                let a_idx = broadcasted_input_linear_index_precomputed(
                                    out_idx,
                                    &out_shape,
                                    &out_strides,
                                    &a_shape,
                                    &a_input_strides,
                                    a_leading,
                                );
                                let b_idx = broadcasted_input_linear_index_precomputed(
                                    out_idx,
                                    &out_shape,
                                    &out_strides,
                                    &b_shape,
                                    &b_input_strides,
                                    b_leading,
                                );
                                local_a[a_idx] += g;
                                local_b[b_idx] += g;
                                (local_a, local_b)
                            },
                        )
                        .reduce(
                            || (vec![0.0; grad_a_len], vec![0.0; grad_b_len]),
                            |(mut a1, mut b1), (a2, b2)| {
                                for (dst, src) in a1.iter_mut().zip(a2.into_iter()) {
                                    *dst += src;
                                }
                                for (dst, src) in b1.iter_mut().zip(b2.into_iter()) {
                                    *dst += src;
                                }
                                (a1, b1)
                            },
                        );
                    grad_a = par_a;
                    grad_b = par_b;
                } else {
                    for (out_idx, g) in grad_out.iter().copied().enumerate() {
                        let a_idx = broadcasted_input_linear_index_precomputed(
                            out_idx,
                            &out_shape,
                            &out_strides,
                            &a_shape,
                            &a_input_strides,
                            a_leading,
                        );
                        let b_idx = broadcasted_input_linear_index_precomputed(
                            out_idx,
                            &out_shape,
                            &out_strides,
                            &b_shape,
                            &b_input_strides,
                            b_leading,
                        );
                        grad_a[a_idx] += g;
                        grad_b[b_idx] += g;
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            for (out_idx, g) in grad_out.iter().copied().enumerate() {
                let a_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &a_shape,
                    &a_input_strides,
                    a_leading,
                );
                let b_idx = broadcasted_input_linear_index_precomputed(
                    out_idx,
                    &out_shape,
                    &out_strides,
                    &b_shape,
                    &b_input_strides,
                    b_leading,
                );
                grad_a[a_idx] += g;
                grad_b[b_idx] += g;
            }

            add_grad_if_needed(&parents[0], &grad_a);
            add_grad_if_needed(&parents[1], &grad_b);
        }
        Op::Mul => {
            let out_shape = t.node.borrow().shape.clone();
            let (a_shape, b_shape) = {
                let a = parents[0].node.borrow();
                let b = parents[1].node.borrow();
                (a.shape.clone(), b.shape.clone())
            };
            let out_strides = strides(&out_shape);
            let a_input_strides = strides(&a_shape);
            let b_input_strides = strides(&b_shape);
            let a_leading = out_shape.len().saturating_sub(a_shape.len());
            let b_leading = out_shape.len().saturating_sub(b_shape.len());

            let mut grad_a = vec![0.0; numel(&a_shape)];
            let mut grad_b = vec![0.0; numel(&b_shape)];

            #[cfg(feature = "parallel")]
            {
                let a_data = parents[0].node.borrow().data.clone();
                let b_data = parents[1].node.borrow().data.clone();
                if should_parallel_broadcast_backward(grad_out.len()) {
                    let grad_a_len = grad_a.len();
                    let grad_b_len = grad_b.len();

                    let (par_a, par_b) = (0..grad_out.len())
                        .into_par_iter()
                        .fold(
                            || (vec![0.0; grad_a_len], vec![0.0; grad_b_len]),
                            |(mut local_a, mut local_b), out_idx| {
                                let g = grad_out[out_idx];
                                let a_idx = broadcasted_input_linear_index_precomputed(
                                    out_idx,
                                    &out_shape,
                                    &out_strides,
                                    &a_shape,
                                    &a_input_strides,
                                    a_leading,
                                );
                                let b_idx = broadcasted_input_linear_index_precomputed(
                                    out_idx,
                                    &out_shape,
                                    &out_strides,
                                    &b_shape,
                                    &b_input_strides,
                                    b_leading,
                                );
                                local_a[a_idx] += g * b_data[b_idx];
                                local_b[b_idx] += g * a_data[a_idx];
                                (local_a, local_b)
                            },
                        )
                        .reduce(
                            || (vec![0.0; grad_a_len], vec![0.0; grad_b_len]),
                            |(mut a1, mut b1), (a2, b2)| {
                                for (dst, src) in a1.iter_mut().zip(a2.into_iter()) {
                                    *dst += src;
                                }
                                for (dst, src) in b1.iter_mut().zip(b2.into_iter()) {
                                    *dst += src;
                                }
                                (a1, b1)
                            },
                        );
                    grad_a = par_a;
                    grad_b = par_b;
                } else {
                    for (out_idx, g) in grad_out.iter().copied().enumerate() {
                        let a_idx = broadcasted_input_linear_index_precomputed(
                            out_idx,
                            &out_shape,
                            &out_strides,
                            &a_shape,
                            &a_input_strides,
                            a_leading,
                        );
                        let b_idx = broadcasted_input_linear_index_precomputed(
                            out_idx,
                            &out_shape,
                            &out_strides,
                            &b_shape,
                            &b_input_strides,
                            b_leading,
                        );
                        grad_a[a_idx] += g * b_data[b_idx];
                        grad_b[b_idx] += g * a_data[a_idx];
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                let a = parents[0].node.borrow();
                let b = parents[1].node.borrow();

                for (out_idx, g) in grad_out.iter().copied().enumerate() {
                    let a_idx = broadcasted_input_linear_index_precomputed(
                        out_idx,
                        &out_shape,
                        &out_strides,
                        &a_shape,
                        &a_input_strides,
                        a_leading,
                    );
                    let b_idx = broadcasted_input_linear_index_precomputed(
                        out_idx,
                        &out_shape,
                        &out_strides,
                        &b_shape,
                        &b_input_strides,
                        b_leading,
                    );
                    grad_a[a_idx] += g * b.data[b_idx];
                    grad_b[b_idx] += g * a.data[a_idx];
                }
            }

            add_grad_if_needed(&parents[0], &grad_a);
            add_grad_if_needed(&parents[1], &grad_b);
        }
        Op::Div => {
            let out_shape = t.node.borrow().shape.clone();
            let (a_shape, b_shape) = {
                let a = parents[0].node.borrow();
                let b = parents[1].node.borrow();
                (a.shape.clone(), b.shape.clone())
            };
            let out_strides = strides(&out_shape);
            let a_input_strides = strides(&a_shape);
            let b_input_strides = strides(&b_shape);
            let a_leading = out_shape.len().saturating_sub(a_shape.len());
            let b_leading = out_shape.len().saturating_sub(b_shape.len());

            let mut grad_a = vec![0.0; numel(&a_shape)];
            let mut grad_b = vec![0.0; numel(&b_shape)];

            #[cfg(feature = "parallel")]
            {
                let a_data = parents[0].node.borrow().data.clone();
                let b_data = parents[1].node.borrow().data.clone();
                if should_parallel_broadcast_backward(grad_out.len()) {
                    let grad_a_len = grad_a.len();
                    let grad_b_len = grad_b.len();

                    let (par_a, par_b) = (0..grad_out.len())
                        .into_par_iter()
                        .fold(
                            || (vec![0.0; grad_a_len], vec![0.0; grad_b_len]),
                            |(mut local_a, mut local_b), out_idx| {
                                let g = grad_out[out_idx];
                                let a_idx = broadcasted_input_linear_index_precomputed(
                                    out_idx,
                                    &out_shape,
                                    &out_strides,
                                    &a_shape,
                                    &a_input_strides,
                                    a_leading,
                                );
                                let b_idx = broadcasted_input_linear_index_precomputed(
                                    out_idx,
                                    &out_shape,
                                    &out_strides,
                                    &b_shape,
                                    &b_input_strides,
                                    b_leading,
                                );
                                let bval = b_data[b_idx];
                                let b2 = bval * bval;
                                local_a[a_idx] += g / bval;
                                local_b[b_idx] += -g * a_data[a_idx] / b2;
                                (local_a, local_b)
                            },
                        )
                        .reduce(
                            || (vec![0.0; grad_a_len], vec![0.0; grad_b_len]),
                            |(mut a1, mut b1), (a2, b2)| {
                                for (dst, src) in a1.iter_mut().zip(a2.into_iter()) {
                                    *dst += src;
                                }
                                for (dst, src) in b1.iter_mut().zip(b2.into_iter()) {
                                    *dst += src;
                                }
                                (a1, b1)
                            },
                        );
                    grad_a = par_a;
                    grad_b = par_b;
                } else {
                    for (out_idx, g) in grad_out.iter().copied().enumerate() {
                        let a_idx = broadcasted_input_linear_index_precomputed(
                            out_idx,
                            &out_shape,
                            &out_strides,
                            &a_shape,
                            &a_input_strides,
                            a_leading,
                        );
                        let b_idx = broadcasted_input_linear_index_precomputed(
                            out_idx,
                            &out_shape,
                            &out_strides,
                            &b_shape,
                            &b_input_strides,
                            b_leading,
                        );
                        let bval = b_data[b_idx];
                        let b2 = bval * bval;
                        grad_a[a_idx] += g / bval;
                        grad_b[b_idx] += -g * a_data[a_idx] / b2;
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                let a = parents[0].node.borrow();
                let b = parents[1].node.borrow();

                for (out_idx, g) in grad_out.iter().copied().enumerate() {
                    let a_idx = broadcasted_input_linear_index_precomputed(
                        out_idx,
                        &out_shape,
                        &out_strides,
                        &a_shape,
                        &a_input_strides,
                        a_leading,
                    );
                    let b_idx = broadcasted_input_linear_index_precomputed(
                        out_idx,
                        &out_shape,
                        &out_strides,
                        &b_shape,
                        &b_input_strides,
                        b_leading,
                    );
                    let bval = b.data[b_idx];
                    let b2 = bval * bval;
                    grad_a[a_idx] += g / bval;
                    grad_b[b_idx] += -g * a.data[a_idx] / b2;
                }
            }

            add_grad_if_needed(&parents[0], &grad_a);
            add_grad_if_needed(&parents[1], &grad_b);
        }
        Op::MatMul => {
            let (a_shape, b_shape) = {
                let a = parents[0].node.borrow();
                let b = parents[1].node.borrow();
                (a.shape.clone(), b.shape.clone())
            };

            let m = a_shape[0];
            let n = a_shape[1];
            let p = b_shape[1];

            let mut grad_a = vec![0.0; m * n];
            let mut grad_b = vec![0.0; n * p];

            {
                let a = parents[0].node.borrow();
                let b = parents[1].node.borrow();

                #[cfg(feature = "parallel")]
                let a_data = a.data.clone();
                #[cfg(feature = "parallel")]
                let b_data = b.data.clone();

                #[cfg(feature = "parallel")]
                if should_parallel_matmul(m, n, p) {
                    grad_a.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                        for (j, dst) in row.iter_mut().enumerate() {
                            let mut acc = 0.0;
                            for k in 0..p {
                                acc += grad_out[i * p + k] * b_data[j * p + k];
                            }
                            *dst = acc;
                        }
                    });
                } else {
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0.0;
                            for k in 0..p {
                                acc += grad_out[i * p + k] * b_data[j * p + k];
                            }
                            grad_a[i * n + j] = acc;
                        }
                    }
                }

                #[cfg(not(feature = "parallel"))]
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for k in 0..p {
                            acc += grad_out[i * p + k] * b.data[j * p + k];
                        }
                        grad_a[i * n + j] = acc;
                    }
                }

                #[cfg(feature = "parallel")]
                if should_parallel_matmul(m, n, p) {
                    grad_b.par_chunks_mut(p).enumerate().for_each(|(j, row)| {
                        for (k, dst) in row.iter_mut().enumerate() {
                            let mut acc = 0.0;
                            for i in 0..m {
                                acc += a_data[i * n + j] * grad_out[i * p + k];
                            }
                            *dst = acc;
                        }
                    });
                } else {
                    for j in 0..n {
                        for k in 0..p {
                            let mut acc = 0.0;
                            for i in 0..m {
                                acc += a_data[i * n + j] * grad_out[i * p + k];
                            }
                            grad_b[j * p + k] = acc;
                        }
                    }
                }

                #[cfg(not(feature = "parallel"))]
                for j in 0..n {
                    for k in 0..p {
                        let mut acc = 0.0;
                        for i in 0..m {
                            acc += a.data[i * n + j] * grad_out[i * p + k];
                        }
                        grad_b[j * p + k] = acc;
                    }
                }
            }

            add_grad_if_needed(&parents[0], &grad_a);
            add_grad_if_needed(&parents[1], &grad_b);
        }
        Op::Transpose2D => {
            let parent_shape = parents[0].node.borrow().shape.clone();
            let rows = parent_shape[0];
            let cols = parent_shape[1];
            let mut grad_in = vec![0.0; rows * cols];

            for r in 0..rows {
                for c in 0..cols {
                    grad_in[r * cols + c] = grad_out[c * rows + r];
                }
            }

            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::ReLU => {
            let grad_in = {
                let input = parents[0].node.borrow();
                let mut grad_in = vec![0.0; input.data.len()];
                for i in 0..input.data.len() {
                    grad_in[i] = if input.data[i] > 0.0 { grad_out[i] } else { 0.0 };
                }
                grad_in
            };
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Tanh => {
            let grad_in = {
                let y_data = t.node.borrow();
                unary_grad_from_output(&grad_out, &y_data.data, |y| 1.0 - y * y)
            };
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Sigmoid => {
            let grad_in = {
                let y_data = t.node.borrow();
                unary_grad_from_output(&grad_out, &y_data.data, |y| y * (1.0 - y))
            };
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Exp => {
            let grad_in = {
                let y_data = t.node.borrow();
                unary_grad_from_output(&grad_out, &y_data.data, |y| y)
            };
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Log => {
            let grad_in = {
                let parent = parents[0].node.borrow();
                let mut grad_in = vec![0.0; grad_out.len()];
                for i in 0..grad_out.len() {
                    grad_in[i] = grad_out[i] / parent.data[i];
                }
                grad_in
            };
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Sqrt => {
            const SQRT_EPS: f32 = 1e-12;
            let grad_in = {
                let y = t.node.borrow();
                let mut grad_in = vec![0.0; grad_out.len()];
                for i in 0..grad_out.len() {
                    let denom = (2.0 * y.data[i]).max(SQRT_EPS);
                    grad_in[i] = grad_out[i] / denom;
                }
                grad_in
            };
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Sum => {
            let parent_len = parents[0].node.borrow().data.len();
            let grad_in = vec![grad_out[0]; parent_len];
            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::SumLastDim => {
            let parent_shape = parents[0].node.borrow().shape.clone();
            let rows = parent_shape[0];
            let cols = parent_shape[1];
            let mut grad_in = vec![0.0; rows * cols];

            for r in 0..rows {
                let g = grad_out[r];
                for c in 0..cols {
                    grad_in[r * cols + c] = g;
                }
            }

            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::SoftmaxLastDim => {
            let parent_shape = parents[0].node.borrow().shape.clone();
            let rows = parent_shape[0];
            let cols = parent_shape[1];
            let y_data = t.node.borrow().data.clone();
            let mut grad_in = vec![0.0; rows * cols];

            for r in 0..rows {
                let row_start = r * cols;
                let mut dot = 0.0;
                for c in 0..cols {
                    dot += grad_out[row_start + c] * y_data[row_start + c];
                }

                for c in 0..cols {
                    let y = y_data[row_start + c];
                    grad_in[row_start + c] = y * (grad_out[row_start + c] - dot);
                }
            }

            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::CausalMaskUpper => {
            let parent_shape = parents[0].node.borrow().shape.clone();
            let rows = parent_shape[0];
            let cols = parent_shape[1];
            let mut grad_in = vec![0.0; rows * cols];

            for r in 0..rows {
                for c in 0..cols {
                    if c <= r {
                        grad_in[r * cols + c] = grad_out[r * cols + c];
                    }
                }
            }

            add_grad_if_needed(&parents[0], &grad_in);
        }
        Op::Mean => {
            let parent_len = parents[0].node.borrow().data.len();
            let scale = grad_out[0] / parent_len as f32;
            let grad_in = vec![scale; parent_len];
            add_grad_if_needed(&parents[0], &grad_in);
        }
    }

}

fn unary_grad_from_output(
    grad_out: &[f32],
    output_data: &[f32],
    derivative_from_output: impl Fn(f32) -> f32,
) -> Vec<f32> {
    grad_out
        .iter()
        .zip(output_data.iter())
        .map(|(g, y)| *g * derivative_from_output(*y))
        .collect()
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>()
}

#[cfg(feature = "parallel")]
fn env_usize_or_default(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

#[cfg(feature = "parallel")]
#[inline]
fn should_parallel_elementwise(len: usize) -> bool {
    let min_len = *ELEMENTWISE_PAR_MIN_LEN.get_or_init(|| {
        env_usize_or_default(
            "LLM_AUTOGRAD_ELEMENTWISE_PAR_MIN_LEN",
            ELEMENTWISE_PAR_MIN_LEN_DEFAULT,
        )
    });
    len >= min_len
}

#[cfg(feature = "parallel")]
pub fn parallel_runtime_thresholds() -> (usize, usize, usize) {
    let elementwise = *ELEMENTWISE_PAR_MIN_LEN.get_or_init(|| {
        env_usize_or_default(
            "LLM_AUTOGRAD_ELEMENTWISE_PAR_MIN_LEN",
            ELEMENTWISE_PAR_MIN_LEN_DEFAULT,
        )
    });
    let broadcast_backward = *BROADCAST_BACKWARD_PAR_MIN_LEN.get_or_init(|| {
        env_usize_or_default(
            "LLM_AUTOGRAD_BROADCAST_BACKWARD_PAR_MIN_LEN",
            BROADCAST_BACKWARD_PAR_MIN_LEN_DEFAULT,
        )
    });
    let matmul_work = *MATMUL_PAR_MIN_WORK.get_or_init(|| {
        env_usize_or_default("LLM_AUTOGRAD_MATMUL_PAR_MIN_WORK", MATMUL_PAR_MIN_WORK_DEFAULT)
    });

    (elementwise, broadcast_backward, matmul_work)
}

#[cfg(feature = "parallel")]
#[inline]
fn should_parallel_broadcast_backward(len: usize) -> bool {
    let min_len = *BROADCAST_BACKWARD_PAR_MIN_LEN.get_or_init(|| {
        env_usize_or_default(
            "LLM_AUTOGRAD_BROADCAST_BACKWARD_PAR_MIN_LEN",
            BROADCAST_BACKWARD_PAR_MIN_LEN_DEFAULT,
        )
    });
    len >= min_len
}

#[cfg(feature = "parallel")]
#[inline]
fn should_parallel_matmul(m: usize, n: usize, p: usize) -> bool {
    let min_work = *MATMUL_PAR_MIN_WORK.get_or_init(|| {
        env_usize_or_default("LLM_AUTOGRAD_MATMUL_PAR_MIN_WORK", MATMUL_PAR_MIN_WORK_DEFAULT)
    });
    m.saturating_mul(n).saturating_mul(p) >= min_work
}

fn broadcast_shape(
    lhs: &[usize],
    rhs: &[usize],
    op: &'static str,
) -> Result<Vec<usize>, AutogradError> {
    let out_rank = lhs.len().max(rhs.len());
    let mut out = vec![1; out_rank];

    for axis in 0..out_rank {
        let lhs_axis = lhs_dim_aligned(lhs, out_rank, axis);
        let rhs_axis = lhs_dim_aligned(rhs, out_rank, axis);

        if lhs_axis == rhs_axis || lhs_axis == 1 || rhs_axis == 1 {
            out[axis] = lhs_axis.max(rhs_axis);
        } else {
            return Err(AutogradError::ShapeMismatch {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
                op,
            });
        }
    }

    Ok(out)
}

fn lhs_dim_aligned(shape: &[usize], out_rank: usize, out_axis: usize) -> usize {
    let leading = out_rank.saturating_sub(shape.len());
    if out_axis < leading {
        1
    } else {
        shape[out_axis - leading]
    }
}

fn broadcasted_input_linear_index_precomputed(
    out_linear: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    leading: usize,
) -> usize {
    if out_shape == input_shape {
        return out_linear;
    }

    let out_rank = out_shape.len();
    let mut rem = out_linear;
    let mut input_linear = 0usize;

    for axis in 0..out_rank {
        let coord = rem / out_strides[axis];
        rem %= out_strides[axis];

        if axis < leading {
            continue;
        }

        let in_axis = axis - leading;
        let in_dim = input_shape[in_axis];
        let mapped = if in_dim == 1 { 0 } else { coord };
        input_linear += mapped * input_strides[in_axis];
    }

    input_linear
}

fn strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn add_grad_if_needed(parent: &Tensor, contrib: &[f32]) {
    let mut n = parent.node.borrow_mut();
    if !n.requires_grad {
        return;
    }
    for (dst, src) in n.grad.iter_mut().zip(contrib.iter()) {
        *dst += *src;
    }
}

fn topo_sort_checked(root: &Tensor) -> Result<Vec<Tensor>, AutogradError> {
    let mut state: HashMap<usize, u8> = HashMap::new();
    let mut out = Vec::new();
    dfs_cycle_checked(root, &mut state, &mut out)?;
    Ok(out)
}

fn dfs_cycle_checked(
    current: &Tensor,
    state: &mut HashMap<usize, u8>,
    out: &mut Vec<Tensor>,
) -> Result<(), AutogradError> {
    let id = Rc::as_ptr(&current.node) as usize;

    if let Some(color) = state.get(&id) {
        if *color == 1 {
            return Err(AutogradError::GraphCycleDetected);
        }
        if *color == 2 {
            return Ok(());
        }
    }

    state.insert(id, 1);

    let parents = current.node.borrow().parents.clone();
    for p in parents {
        dfs_cycle_checked(&p, state, out)?;
    }

    state.insert(id, 2);
    out.push(current.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AutogradError, Op, Tensor};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn elementwise_gradients_are_correct() {
        let a = Tensor::from_data(vec![2.0, 3.0], vec![2], true);
        let b = Tensor::from_data(vec![4.0, 5.0], vec![2], true);

        let y = a.mul(&b).unwrap().add(&a).unwrap().sum();
        y.backward().unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        assert!(approx_eq(grad_a[0], 5.0, 1e-6));
        assert!(approx_eq(grad_a[1], 6.0, 1e-6));
        assert!(approx_eq(grad_b[0], 2.0, 1e-6));
        assert!(approx_eq(grad_b[1], 3.0, 1e-6));
    }

    #[test]
    fn matmul_gradients_are_correct() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let b = Tensor::from_data(vec![5.0, 6.0], vec![2, 1], true);

        let y = a.matmul(&b).unwrap().sum();
        y.backward().unwrap();

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        assert!(approx_eq(grad_a[0], 5.0, 1e-6));
        assert!(approx_eq(grad_a[1], 6.0, 1e-6));
        assert!(approx_eq(grad_a[2], 5.0, 1e-6));
        assert!(approx_eq(grad_a[3], 6.0, 1e-6));

        assert!(approx_eq(grad_b[0], 4.0, 1e-6));
        assert!(approx_eq(grad_b[1], 6.0, 1e-6));
    }

    #[test]
    fn matmul_gradients_non_square_are_correct() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], true);

        let y = a.matmul(&b).unwrap().sum();
        y.backward().unwrap();

        assert_eq!(a.grad().unwrap(), vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
        assert_eq!(b.grad().unwrap(), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn relu_masks_negative_branch() {
        let x = Tensor::from_data(vec![-2.0, 3.0, -1.0, 4.0], vec![4], true);
        let y = x.relu().sum();
        y.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn add_scalar_broadcast_works() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], true);
        let s = Tensor::from_data(vec![2.0], vec![1], true);

        let y = x.add(&s).unwrap().sum();
        y.backward().unwrap();

        assert_eq!(x.grad().unwrap(), vec![1.0, 1.0, 1.0]);
        assert_eq!(s.grad().unwrap(), vec![3.0]);
    }

    #[test]
    fn mul_scalar_broadcast_works() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], true);
        let s = Tensor::from_data(vec![2.0], vec![1], true);

        let y = x.mul(&s).unwrap().sum();
        y.backward().unwrap();

        assert_eq!(x.grad().unwrap(), vec![2.0, 2.0, 2.0]);
        assert_eq!(s.grad().unwrap(), vec![6.0]);
    }

    #[test]
    fn add_broadcast_rank_expansion_backward_reduces_axes() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3], true);

        let y = a.add(&b).unwrap();
        y.sum().backward().unwrap();

        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn mul_broadcast_multiaxis_backward_reduces_correctly() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3], true);
        let b = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], true);

        let y = a.mul(&b).unwrap();
        y.sum().backward().unwrap();

        assert_eq!(a.grad().unwrap(), vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        assert_eq!(b.grad().unwrap(), vec![21.0, 21.0, 21.0, 21.0]);
    }

    #[test]
    fn mean_backward_scales_grad() {
        let x = Tensor::from_data(vec![2.0, 4.0, 6.0, 8.0], vec![4], true);
        let y = x.mean();
        y.backward().unwrap();
        assert_eq!(x.grad().unwrap(), vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn log_domain_error_reports_index() {
        let x = Tensor::from_data(vec![1.0, 0.0, 3.0], vec![3], true);
        let err = match x.log() {
            Ok(_) => panic!("expected log domain error"),
            Err(e) => e,
        };
        assert_eq!(
            err,
            AutogradError::DomainError {
                op: "log",
                index: 1,
                value: 0.0,
            }
        );
    }

    #[test]
    fn with_data_and_with_grad_avoid_clone_api_surface() {
        let x = Tensor::from_data(vec![1.0, 2.0], vec![2], true);
        let s = x.sum();
        s.backward().unwrap();

        let n = x.with_data(|d| d.len());
        let gsum = x.with_grad(|g| g.unwrap().iter().sum::<f32>());

        assert_eq!(n, 2);
        assert!(approx_eq(gsum, 2.0, 1e-6));
    }

    #[test]
    fn backward_with_grad_supports_non_scalar_outputs() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], true);
        let y = x.mul(&x).unwrap();
        y.backward_with_grad(&[1.0, 0.0, 2.0]).unwrap();

        assert_eq!(x.grad().unwrap(), vec![2.0, 0.0, 12.0]);
    }

    #[test]
    fn backward_with_grad_rejects_mismatched_length() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], true);
        let y = x.mul(&x).unwrap();
        let err = y.backward_with_grad(&[1.0, 0.0]).unwrap_err();

        assert_eq!(
            err,
            AutogradError::BackwardGradLenMismatch {
                expected: 3,
                found: 2,
            }
        );
    }

    #[test]
    fn broadcast_incompatible_shapes_return_error() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3], true);
        let err = match a.add(&b) {
            Ok(_) => panic!("expected shape mismatch"),
            Err(e) => e,
        };

        assert_eq!(
            err,
            AutogradError::ShapeMismatch {
                lhs: vec![2, 2],
                rhs: vec![3],
                op: "add",
            }
        );
    }

    #[test]
    fn div_forward_backward_and_safety() {
        let a = Tensor::from_data(vec![2.0, 4.0, 8.0], vec![3], true);
        let b = Tensor::from_data(vec![2.0], vec![1], true);

        let y = a.div(&b).unwrap().sum();
        y.backward().unwrap();

        assert_eq!(a.grad().unwrap(), vec![0.5, 0.5, 0.5]);
        assert_eq!(b.grad().unwrap(), vec![-3.5]);

        let z = Tensor::from_data(vec![1.0, 2.0], vec![2], true);
        let zero = Tensor::from_data(vec![1.0, 0.0], vec![2], true);
        let err = match z.div(&zero) {
            Ok(_) => panic!("expected division by zero"),
            Err(e) => e,
        };
        assert_eq!(err, AutogradError::DivisionByZero { index: 1 });
    }

    #[test]
    fn sqrt_domain_and_gradient_are_safe() {
        let x = Tensor::from_data(vec![4.0, 9.0], vec![2], true);
        let y = x.sqrt().unwrap().sum();
        y.backward().unwrap();

        let grad = x.grad().unwrap();
        assert!(approx_eq(grad[0], 0.25, 1e-6));
        assert!(approx_eq(grad[1], 1.0 / 6.0, 1e-6));

        let bad = Tensor::from_data(vec![1.0, -1.0], vec![2], true);
        let err = match bad.sqrt() {
            Ok(_) => panic!("expected sqrt domain error"),
            Err(e) => e,
        };
        assert_eq!(
            err,
            AutogradError::DomainError {
                op: "sqrt",
                index: 1,
                value: -1.0,
            }
        );
    }

    #[test]
    fn data_ref_and_grad_ref_expose_borrow_views() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], true);
        let y = x.sum();
        y.backward().unwrap();

        let data_sum: f32 = x.data_ref().iter().sum();
        let grad_sum: f32 = x.grad_ref().unwrap().iter().sum();

        assert!(approx_eq(data_sum, 6.0, 1e-6));
        assert!(approx_eq(grad_sum, 3.0, 1e-6));
    }

    #[test]
    fn cycle_detection_returns_error() {
        let x = Tensor::from_data(vec![1.0], vec![1], true);
        {
            let mut n = x.node.borrow_mut();
            n.parents.push(x.clone());
            n.op = Op::Add;
        }

        let err = x.backward().unwrap_err();
        assert_eq!(err, AutogradError::GraphCycleDetected);
    }

    #[test]
    fn transpose2d_backward_is_correct() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let y = x.transpose2d().unwrap().sum();
        y.backward().unwrap();

        assert_eq!(x.grad().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn sum_last_dim_backward_broadcasts_row_grads() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let y = x.sum_last_dim().unwrap();
        y.backward_with_grad(&[2.0, 3.0]).unwrap();

        assert_eq!(x.grad().unwrap(), vec![2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn causal_mask_upper_blocks_future_gradients() {
        let x = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            true,
        );
        let y = x.causal_mask_upper(-1.0e9).unwrap().sum();
        y.backward().unwrap();

        assert_eq!(x.grad().unwrap(), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn softmax_last_dim_is_stable_and_row_normalized() {
        let x = Tensor::from_data(vec![1000.0, 1001.0, 1002.0, 0.0, 0.0, 0.0], vec![2, 3], true);
        let y = x.softmax_last_dim().unwrap();
        let out = y.data();

        let row0_sum: f32 = out[0..3].iter().sum();
        let row1_sum: f32 = out[3..6].iter().sum();

        assert!(out.iter().all(|v| v.is_finite()));
        assert!(approx_eq(row0_sum, 1.0, 1e-6));
        assert!(approx_eq(row1_sum, 1.0, 1e-6));

        assert!(approx_eq(out[0], 0.09003057, 1e-5));
        assert!(approx_eq(out[1], 0.24472848, 1e-5));
        assert!(approx_eq(out[2], 0.66524094, 1e-5));
    }

    #[test]
    fn softmax_last_dim_backward_zero_for_constant_upstream() {
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let y = x.softmax_last_dim().unwrap();
        y.backward_with_grad(&[1.0, 1.0, 1.0, 1.0]).unwrap();

        let grad = x.grad().unwrap();
        assert!(grad.iter().all(|g| g.abs() < 1e-6));
    }
}
