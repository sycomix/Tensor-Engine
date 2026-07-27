use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    InvalidShape {
        context: &'static str,
        expected_rank: usize,
        found_rank: usize,
    },
    EmptyTensor {
        context: &'static str,
    },
    Unsupported(&'static str),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuAutogradBackend;

impl CpuAutogradBackend {
    pub fn from_data(
        &self,
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<Tensor, BackendError> {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|_| BackendError::Unsupported("data length does not match tensor shape"))?;
        Ok(Tensor::new(array, requires_grad))
    }

    pub fn shape(&self, tensor: &Tensor) -> Vec<usize> {
        tensor.shape()
    }

    pub fn data(&self, tensor: &Tensor) -> Vec<f32> {
        tensor.to_vec()
    }

    pub fn grad(&self, tensor: &Tensor) -> Option<Vec<f32>> {
        tensor
            .lock()
            .grad
            .as_ref()
            .map(|grad| grad.iter().copied().collect())
    }

    pub fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        Ok(lhs.add(rhs))
    }

    pub fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        Ok(lhs.mul(rhs))
    }

    pub fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        Ok(lhs.div(rhs))
    }

    pub fn relu(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        Ok(tensor.relu())
    }

    pub fn log(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        Ok(tensor.log())
    }

    pub fn softmax_last_dim(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        let shape = tensor.shape();
        let axis = shape
            .len()
            .checked_sub(1)
            .ok_or(BackendError::EmptyTensor {
                context: "softmax_last_dim",
            })?;
        Ok(tensor.softmax(axis))
    }

    pub fn sqrt(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        Ok(tensor.sqrt())
    }

    pub fn transpose2d(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(BackendError::InvalidShape {
                context: "transpose2d",
                expected_rank: 2,
                found_rank: shape.len(),
            });
        }
        Ok(tensor.permute(vec![1, 0]))
    }

    pub fn sum_last_dim(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        if tensor.shape().is_empty() {
            return Err(BackendError::EmptyTensor {
                context: "sum_last_dim",
            });
        }
        Ok(tensor.sum_axis(-1, true))
    }

    pub fn causal_mask_upper(
        &self,
        tensor: &Tensor,
        mask_value: f32,
    ) -> Result<Tensor, BackendError> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(BackendError::InvalidShape {
                context: "causal_mask_upper",
                expected_rank: 2,
                found_rank: shape.len(),
            });
        }
        let (rows, cols) = (shape[0], shape[1]);
        let keep = ArrayD::from_shape_fn(IxDyn(&[rows, cols]), |index| {
            if index[1] > index[0] {
                0.0
            } else {
                1.0
            }
        });
        let bias = ArrayD::from_shape_fn(IxDyn(&[rows, cols]), |index| {
            if index[1] > index[0] {
                mask_value
            } else {
                0.0
            }
        });
        Ok(tensor
            .mul(&Tensor::new(keep, false))
            .add(&Tensor::new(bias, false)))
    }

    pub fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape.len() != 2 {
            return Err(BackendError::InvalidShape {
                context: "matmul(lhs)",
                expected_rank: 2,
                found_rank: lhs_shape.len(),
            });
        }
        if rhs_shape.len() != 2 {
            return Err(BackendError::InvalidShape {
                context: "matmul(rhs)",
                expected_rank: 2,
                found_rank: rhs_shape.len(),
            });
        }
        if lhs_shape[1] != rhs_shape[0] {
            return Err(BackendError::Unsupported(
                "matmul inner dimensions must match",
            ));
        }
        Ok(lhs.matmul(rhs))
    }

    pub fn mean(&self, tensor: &Tensor) -> Result<Tensor, BackendError> {
        Ok(tensor.mean())
    }

    pub fn backward(&self, tensor: &Tensor) -> Result<(), BackendError> {
        tensor.backward();
        Ok(())
    }

    pub fn scalar(&self, value: f32, requires_grad: bool) -> Result<Tensor, BackendError> {
        self.from_data(vec![value], vec![1], requires_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::CpuAutogradBackend;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn backend_uses_canonical_tensor_and_preserves_transpose_gradients() {
        fn assert_canonical(_: &crate::tensor::Tensor) {}

        let backend = CpuAutogradBackend;
        let input = backend
            .from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("canonical input");
        assert_canonical(&input);

        let transposed = backend.transpose2d(&input).expect("transpose");
        let loss = backend.mean(&transposed).expect("mean");
        backend.backward(&loss).expect("backward");

        let grad = backend.grad(&input).expect("input gradient");
        assert_eq!(grad.len(), 4);
        for value in grad {
            assert!((value - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn canonical_causal_mask_blocks_upper_triangle_gradients() {
        let backend = CpuAutogradBackend;
        let input = backend
            .from_data(vec![0.0; 9], vec![3, 3], true)
            .expect("canonical input");
        let masked = backend
            .causal_mask_upper(&input, -1.0e9)
            .expect("causal mask");

        assert_eq!(
            backend.data(&masked),
            vec![0.0, -1.0e9, -1.0e9, 0.0, 0.0, -1.0e9, 0.0, 0.0, 0.0]
        );

        let loss = backend.mean(&masked).expect("mean");
        backend.backward(&loss).expect("backward");
        let grad = backend.grad(&input).expect("input gradient");
        let scale = 1.0 / 9.0;
        assert_eq!(
            grad,
            vec![scale, 0.0, 0.0, scale, scale, 0.0, scale, scale, scale]
        );
    }

    #[test]
    fn canonical_elementwise_and_broadcast_gradients_match_contract() {
        let backend = CpuAutogradBackend;
        let a = backend
            .from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3], true)
            .expect("a");
        let b = backend
            .from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], true)
            .expect("b");

        let product = backend.mul(&a, &b).expect("broadcast multiply");
        let loss = product.sum();
        backend.backward(&loss).expect("backward");

        assert_close(&backend.grad(&a).expect("a gradient"), &[10.0; 6]);
        assert_close(&backend.grad(&b).expect("b gradient"), &[21.0; 4]);

        let x = backend
            .from_data(vec![1.0, 2.0, 3.0], vec![3], true)
            .expect("x");
        let scalar = backend.scalar(2.0, true).expect("scalar");
        let sum = backend.add(&x, &scalar).expect("broadcast add").sum();
        backend.backward(&sum).expect("backward");
        assert_close(&backend.grad(&x).expect("x gradient"), &[1.0; 3]);
        assert_close(&backend.grad(&scalar).expect("scalar gradient"), &[3.0]);
    }

    #[test]
    fn canonical_matmul_and_activation_gradients_match_contract() {
        let backend = CpuAutogradBackend;
        let a = backend
            .from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true)
            .expect("a");
        let b = backend
            .from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], true)
            .expect("b");
        let loss = backend.matmul(&a, &b).expect("matmul").sum();
        backend.backward(&loss).expect("backward");
        assert_close(
            &backend.grad(&a).expect("a gradient"),
            &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0],
        );
        assert_close(
            &backend.grad(&b).expect("b gradient"),
            &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
        );

        let x = backend
            .from_data(vec![-2.0, 3.0, -1.0, 4.0], vec![4], true)
            .expect("x");
        let relu_loss = backend.relu(&x).expect("relu").sum();
        backend.backward(&relu_loss).expect("backward");
        assert_close(
            &backend.grad(&x).expect("relu gradient"),
            &[0.0, 1.0, 0.0, 1.0],
        );
    }

    #[test]
    fn canonical_div_sqrt_and_sum_axis_gradients_match_contract() {
        let backend = CpuAutogradBackend;
        let numerator = backend
            .from_data(vec![2.0, 4.0, 8.0], vec![3], true)
            .expect("numerator");
        let denominator = backend.scalar(2.0, true).expect("denominator");
        let loss = backend
            .div(&numerator, &denominator)
            .expect("division")
            .sum();
        backend.backward(&loss).expect("backward");
        assert_close(
            &backend.grad(&numerator).expect("numerator gradient"),
            &[0.5; 3],
        );
        assert_close(
            &backend.grad(&denominator).expect("denominator gradient"),
            &[-3.5],
        );

        let x = backend.from_data(vec![4.0, 9.0], vec![2], true).expect("x");
        let sqrt_loss = backend.sqrt(&x).expect("sqrt").sum();
        backend.backward(&sqrt_loss).expect("backward");
        assert_close(
            &backend.grad(&x).expect("sqrt gradient"),
            &[0.25, 1.0 / 6.0],
        );

        let rows = backend
            .from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
            .expect("rows");
        let row_sums = backend.sum_last_dim(&rows).expect("row sums");
        let upstream = backend
            .from_data(vec![2.0, 3.0], vec![2, 1], false)
            .expect("upstream");
        let weighted = backend.mul(&row_sums, &upstream).expect("weighted").sum();
        backend.backward(&weighted).expect("backward");
        assert_close(
            &backend.grad(&rows).expect("sum-axis gradient"),
            &[2.0, 2.0, 3.0, 3.0],
        );
    }

    #[test]
    fn canonical_softmax_is_stable_and_has_zero_constant_upstream_gradient() {
        let backend = CpuAutogradBackend;
        let x = backend
            .from_data(
                vec![1000.0, 1001.0, 1002.0, 0.0, 0.0, 0.0],
                vec![2, 3],
                true,
            )
            .expect("x");
        let output = backend.softmax_last_dim(&x).expect("softmax");
        let values = backend.data(&output);
        assert!(values.iter().all(|value| value.is_finite()));
        assert_close(&[values[0] + values[1] + values[2]], &[1.0]);
        assert_close(&[values[3] + values[4] + values[5]], &[1.0]);
        assert_close(&values[0..3], &[0.09003057, 0.24472848, 0.66524094]);

        let loss = output.sum();
        backend.backward(&loss).expect("backward");
        assert_close(&backend.grad(&x).expect("softmax gradient"), &[0.0; 6]);
    }
}
