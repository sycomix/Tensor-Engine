use ndarray::{ArrayD, IxDyn};

/// Wrapper for integer label arrays (e.g., class indices)
pub struct Labels(pub ArrayD<i64>);

impl Labels {
    pub fn new(arr: ArrayD<i64>) -> Self { Labels(arr) }

    /// Convert 1D labels indices to one-hot 2D f32 array of shape (n, num_classes) or more
    /// Supports multi-dimensional indices mapping along a specific axis by returning one-hot for that axis flattened.
    pub fn to_one_hot(&self, num_classes: usize) -> ArrayD<f32> {
        // Accept 1D labels or flatten higher dimensions into 1D
        let len = self.0.len();
        let out = ArrayD::zeros(IxDyn(&[len, num_classes][..]));
        let mut out_view = match out.into_dimensionality::<ndarray::Ix2>() {
            Ok(v) => v,
            Err(e) => {
                log::error!("Labels::to_one_hot: Failed to convert output to 2D: {}", e);
                return ArrayD::zeros(IxDyn(&[0]));
            }
        };
        for (i, idx_val) in self.0.iter().enumerate() {
            let j = *idx_val as usize;
            if j >= num_classes {
                log::error!("Labels::to_one_hot: index out of bounds: {} >= {}", j, num_classes);
                continue;
            }
            out_view[[i, j]] = 1.0;
        }
        out_view.into_dyn()
    }
}
