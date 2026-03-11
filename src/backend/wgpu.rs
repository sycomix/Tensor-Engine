use crate::backend::traits::{Backend, Storage};
use crate::dtype::{DType, TensorStorage};
use ndarray::{ArrayD, Axis, IxDyn};

pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok_or("Failed to find an appropriate adapter".to_string())?;

            log::info!("WGPU Adapter: {:?}", adapter.get_info());

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("TensorEngine WGPU Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("Failed to create device: {}", e))?;

            Ok(Self { device, queue })
        })
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
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
        log::warn!("WGPU matmul not yet implemented - falling back to CPU");
        
        if a.len() == 0 || b.len() == 0 {
            return None;
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            log::error!("MatMul requires 2D arrays");
            return None;
        }

        if a_shape[1] != b_shape[0] {
            log::error!(
                "Matrix dimensions incompatible: {:?} x {:?} cannot be multiplied",
                a_shape,
                b_shape
            );
            return None;
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let mut result = ArrayD::zeros(IxDyn(&[m, n]));

        for i in 0..m {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Some(result)
    }

    fn softmax(&self, input: &ArrayD<f32>, axis: isize) -> Option<ArrayD<f32>> {
        if input.len() == 0 {
            return None;
        }

        let mut result = input.to_owned();
        
        let shape = input.shape();
        let ndim = shape.len();

        if ndim < 1 || axis < -(ndim as isize) || axis >= ndim as isize {
            log::error!("Invalid softmax axis: {} for array with {} dimensions", axis, ndim);
            return None;
        }

        let actual_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };

        // Find max along the specified axis using fold_axis
        let max_val: f32 = result.fold_axis(Axis(actual_axis), f32::NEG_INFINITY, |a, &b| a.max(b));

        // Subtract max for numerical stability and compute exp
        let exp_vals: ArrayD<f32> = result.mapv(|x| (x - max_val).exp());

        // Compute sum along the axis using fold_axis
        let sum_exp: ArrayD<f32> = exp_vals.fold_axis(
            Axis(actual_axis), 
            0.0f32, 
            |a, &b| a + b
        );

        // Normalize each element along the axis by dividing by sum using indexed_iter_mut
        let mut idx: Vec<usize> = vec![0; ndim];
        
        for flat_idx in 0..shape.len() {
            let mut temp = flat_idx;
            
            for dim in (0..ndim).rev() {
                if dim > 0 {
                    idx[dim] = temp % shape[dim];
                    temp /= shape[dim];
                } else {
                    idx[0] = temp;
                }
            }

            let val = exp_vals[[&idx[..]]];
            let sum_val = sum_exp[[&idx[..]]];
            
            if sum_val > 0.0 {
                result[[&idx[..]]] = val / sum_val;
            } else {
                log::warn!("Sum of exponentials is zero, setting to uniform distribution");
                result[[&idx[..]]] = 1.0 / shape[actual_axis] as f32;
            }
        }

        Some(result)
    }

    fn memory_info(&self) -> (usize, usize) {
        let total = 8 * 1024 * 1024 * 1024;
        let used = 512 * 1024 * 1024;
        (used, total)
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}
