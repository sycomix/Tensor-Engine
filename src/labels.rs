use ndarray::{ArrayD, IxDyn};

/// Wrapper for integer label arrays (e.g., class indices)
pub struct Labels(pub ArrayD<i64>);

impl Labels {
    pub fn new(arr: ArrayD<i64>) -> Self { Labels(arr) }

    /// Convert 1D labels indices to one-hot 2D f32 array of shape (n, num_classes) or more
    /// Supports multi-dimensional indices mapping along a specific axis by returning one-hot for that axis flattened.
    pub fn to_one_hot(&self, num_classes: usize) -> ArrayD<f32> {
        // expects 1D labels for now
        assert!(self.0.ndim() == 1, "Labels::to_one_hot currently only accepts 1D index array");
        let len = self.0.shape()[0];
        let out = ArrayD::zeros(IxDyn(&[len, num_classes]));
        let mut out_view = out.into_dimensionality::<ndarray::Ix2>().unwrap();
        for (i, &idx) in self.0.clone().into_dimensionality::<ndarray::Ix1>().unwrap().iter().enumerate() {
            let j = idx as usize;
            out_view[[i, j]] = 1.0;
        }
        out_view.into_dyn()
    }
}
