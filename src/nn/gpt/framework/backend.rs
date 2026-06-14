use super::autograd::{AutogradError, Tensor};

#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    Autograd(AutogradError),
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

impl From<AutogradError> for BackendError {
    fn from(value: AutogradError) -> Self {
        Self::Autograd(value)
    }
}

pub trait TensorBackend {
    type Tensor;

    fn from_data(
        &self,
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<Self::Tensor, BackendError>;

    fn shape(&self, tensor: &Self::Tensor) -> Vec<usize>;
    fn data(&self, tensor: &Self::Tensor) -> Vec<f32>;
    fn grad(&self, tensor: &Self::Tensor) -> Option<Vec<f32>>;
    fn zero_grad(&self, tensor: &Self::Tensor);

    fn add(&self, lhs: &Self::Tensor, rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn mul(&self, lhs: &Self::Tensor, rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn div(&self, lhs: &Self::Tensor, rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn relu(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn exp(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn log(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn softmax_last_dim(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn sqrt(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn transpose2d(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn sum_last_dim(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn causal_mask_upper(
        &self,
        tensor: &Self::Tensor,
        mask_value: f32,
    ) -> Result<Self::Tensor, BackendError>;
    fn matmul(
        &self,
        lhs: &Self::Tensor,
        rhs: &Self::Tensor,
    ) -> Result<Self::Tensor, BackendError>;
    fn mean(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn backward(&self, tensor: &Self::Tensor) -> Result<(), BackendError>;

    fn scalar(&self, value: f32, requires_grad: bool) -> Result<Self::Tensor, BackendError> {
        self.from_data(vec![value], vec![1], requires_grad)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuAutogradBackend;

impl TensorBackend for CpuAutogradBackend {
    type Tensor = Tensor;

    fn from_data(
        &self,
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<Self::Tensor, BackendError> {
        Ok(Tensor::from_data(data, shape, requires_grad))
    }

    fn shape(&self, tensor: &Self::Tensor) -> Vec<usize> {
        tensor.shape()
    }

    fn data(&self, tensor: &Self::Tensor) -> Vec<f32> {
        tensor.data()
    }

    fn grad(&self, tensor: &Self::Tensor) -> Option<Vec<f32>> {
        tensor.grad()
    }

    fn zero_grad(&self, tensor: &Self::Tensor) {
        tensor.zero_grad();
    }

    fn add(&self, lhs: &Self::Tensor, rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(lhs.add(rhs)?)
    }

    fn mul(&self, lhs: &Self::Tensor, rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(lhs.mul(rhs)?)
    }

    fn div(&self, lhs: &Self::Tensor, rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(lhs.div(rhs)?)
    }

    fn relu(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.relu())
    }

    fn exp(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.exp())
    }

    fn log(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.log()?)
    }

    fn softmax_last_dim(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.softmax_last_dim()?)
    }

    fn sqrt(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.sqrt()?)
    }

    fn transpose2d(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.transpose2d()?)
    }

    fn sum_last_dim(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.sum_last_dim()?)
    }

    fn causal_mask_upper(
        &self,
        tensor: &Self::Tensor,
        mask_value: f32,
    ) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.causal_mask_upper(mask_value)?)
    }

    fn matmul(
        &self,
        lhs: &Self::Tensor,
        rhs: &Self::Tensor,
    ) -> Result<Self::Tensor, BackendError> {
        Ok(lhs.matmul(rhs)?)
    }

    fn mean(&self, tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor.mean())
    }

    fn backward(&self, tensor: &Self::Tensor) -> Result<(), BackendError> {
        Ok(tensor.backward()?)
    }
}

#[cfg(feature = "tch-backend")]
#[derive(Debug, Default, Clone, Copy)]
pub struct TchBackend;

#[cfg(feature = "tch-backend")]
impl TensorBackend for TchBackend {
    type Tensor = ();

    fn from_data(
        &self,
        _data: Vec<f32>,
        _shape: Vec<usize>,
        _requires_grad: bool,
    ) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn shape(&self, _tensor: &Self::Tensor) -> Vec<usize> {
        Vec::new()
    }

    fn data(&self, _tensor: &Self::Tensor) -> Vec<f32> {
        Vec::new()
    }

    fn grad(&self, _tensor: &Self::Tensor) -> Option<Vec<f32>> {
        None
    }

    fn zero_grad(&self, _tensor: &Self::Tensor) {}

    fn add(&self, _lhs: &Self::Tensor, _rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn mul(&self, _lhs: &Self::Tensor, _rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn div(&self, _lhs: &Self::Tensor, _rhs: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn relu(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn exp(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn log(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn softmax_last_dim(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn sqrt(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn transpose2d(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn sum_last_dim(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn causal_mask_upper(
        &self,
        _tensor: &Self::Tensor,
        _mask_value: f32,
    ) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn matmul(
        &self,
        _lhs: &Self::Tensor,
        _rhs: &Self::Tensor,
    ) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn mean(&self, _tensor: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }

    fn backward(&self, _tensor: &Self::Tensor) -> Result<(), BackendError> {
        Err(BackendError::Unsupported("tch backend scaffold not yet implemented"))
    }
}
