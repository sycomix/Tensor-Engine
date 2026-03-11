use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, ArrayView2, IxDyn, Axis};

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }

    fn create_from_data(&self, data: ArrayD<f32>, dtype: DType) -> Box<dyn Storage> {
        if dtype == DType::F32 {
            Box::new(TensorStorage::F32(data))
        } else {
            Box::new(TensorStorage::from_f32_array(&data, dtype))
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Box<dyn Storage> {
        let shape_ix = IxDyn(shape);
        let data = ArrayD::zeros(shape_ix);
        Box::new(TensorStorage::F32(data))
    }

    fn create_ones(&self, shape: &[usize]) -> Box<dyn Storage> {
        let shape_ix = IxDyn(shape);
        let data = ArrayD::from_elem(shape_ix, 1.0);
        Box::new(TensorStorage::F32(data))
    }

    fn matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        // Support both 2D and batched matrix multiplication
        match (a.ndim(), b.ndim()) {
            (2, 2) => self.matmul_2d(a, b),
            (3, 3) => self.matmul_3d(a, b),
            _ => None,
        }
    }

fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        let mut output = input.clone();
        let ndim = input.ndim();
        
        if axis < 0 || (axis as usize) >= ndim {
            return None;
        }
        
        let axis = axis as usize;
        
        // Compute max for numerical stability along the softmax axis
        let max_val: f32 = output.iter().cloned().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Subtract max and compute exp
        output.mapv_inplace(|x| (x - max_val).exp());
        
        // Sum along axis
        let sum: f32 = output.sum_axis(Axis(axis));
        
        // Divide by sum for each element
        for i in 0..output.len() {
            let mut flat_idx = 0;
            let mut remainder = i;
            for (j, dim) in output.shape().iter().enumerate() {
                let idx_j = remainder % *dim;
                remainder /= *dim;
                flat_idx += idx_j * output.strides()[j];
            }
            
            if sum > 0.0 {
                output[[flat_idx]] /= sum;
            }
        }
        
        Some(output)
    }

    fn memory_info(&self) -> (usize, usize) {
        // CPU backend - estimate based on system memory
        let total = 16 * 1024 * 1024 * 1024; // Assume 16GB
        let used = 2 * 1024 * 1024 * 1024; // Estimate 2GB used
        (used, total)
    }

    fn synchronize(&self) {
        // CPU is synchronous by nature
    }
}

impl CpuBackend {
fn matmul_2d(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0] {
            log::warn!("matmul_2d: Shape mismatch");
            return None;
        }

        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let mut c = ArrayD::<f32>::zeros(IxDyn(&[m, n]));

        // Use ndarray's dot product for simplicity
        let a_2d: ArrayView2<f32> = match ndarray::ArrayView2::from_shape((m, k), a.as_slice()?) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let b_2d: ArrayView2<f32> = match ndarray::ArrayView2::from_shape((k, n), b.as_slice()?) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let result = a_2d.dot(&b_2d);
        c.assign(&result.into_dyn());

        Some(c)
    }

    fn matmul_3d(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
        if a.ndim() != 3 || b.ndim() != 3 {
            return None;
        }

        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];

        if b.shape()[1] != k {
            log::warn!("matmul_3d: Shape mismatch");
            return None;
        }

        let mut c = ArrayD::<f32>::zeros(IxDyn(&[batch, m, n]));

        for i in 0..batch {
            let a_i = a.index_axis(Axis(0), i);
            let b_i = b.index_axis(Axis(0), i);

            if let (Ok(a_2d), Ok(b_2d)) = (a_i.into_dimensionality::<ndarray::Ix2>(), b_i.into_dimensionality::<ndarray::Ix2>()) {
                let result = a_2d.dot(&b_2d);
                c.index_axis_mut(Axis(0), i).assign(&result.into_dyn());
            } else {
                return None;
            }
        }

        Some(c)
    }
}
