use crate::tensor::Tensor;
use ndarray::IxDyn;
use std::sync::Arc;
use super::Module;
use std::any::Any;

/// 3D convolution layer (NCDHW)
#[derive(Clone)]
pub struct Conv3D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_d: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[
            out_channels,
            in_channels,
            kernel_d,
            kernel_h,
            kernel_w,
        ]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(IxDyn(&[out_channels])),
                true,
            ))
        } else {
            None
        };
        Conv3D {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for Conv3D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(
            Arc::new(crate::ops::Conv3D::new(self.stride, self.padding)),
            &inputs,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Depthwise Separable Conv2D Module
#[derive(Clone)]
pub struct DepthwiseSeparableConv2D {
    pub depthwise_weight: Tensor,
    pub pointwise_weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl DepthwiseSeparableConv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let dw = ndarray::Array::zeros(IxDyn(&[in_channels, 1, kernel_size, kernel_size]));
        let pw = ndarray::Array::zeros(IxDyn(&[out_channels, in_channels, 1, 1]));
        DepthwiseSeparableConv2D {
            depthwise_weight: Tensor::new(dw, true),
            pointwise_weight: Tensor::new(pw, true),
            bias: if bias {
                Some(Tensor::new(
                    ndarray::Array::zeros(IxDyn(&[out_channels])),
                    true,
                ))
            } else {
                None
            },
            stride,
            padding,
        }
    }
}

impl Module for DepthwiseSeparableConv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![
            input.clone(),
            self.depthwise_weight.clone(),
            self.pointwise_weight.clone(),
        ];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(
            Arc::new(crate::ops::DepthwiseSeparableConv2D::new(
                self.stride,
                self.padding,
            )),
            &inputs,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.depthwise_weight.clone(), self.pointwise_weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// ConvTranspose2D Module
#[derive(Clone)]
pub struct ConvTranspose2D {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvTranspose2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let weight_data = ndarray::Array::zeros(IxDyn(&[
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]));
        let weight = Tensor::new(weight_data, true);
        let bias = if bias {
            Some(Tensor::new(
                ndarray::Array::zeros(IxDyn(&[out_channels])),
                true,
            ))
        } else {
            None
        };
        ConvTranspose2D {
            weight,
            bias,
            stride,
            padding,
        }
    }
}

impl Module for ConvTranspose2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            inputs.push(b.clone());
        }
        Tensor::apply(
            Arc::new(crate::ops::ConvTranspose2D::new(self.stride, self.padding)),
            &inputs,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Average Pooling 2D layer wrapper
#[derive(Clone)]
pub struct AvgPool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2D {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        AvgPool2D {
            kernel_size,
            stride,
        }
    }
}

impl Module for AvgPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::AvgPool2D {
                kernel_size: self.kernel_size,
                stride: self.stride,
            }),
            &[input.clone()],
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Adaptive average pooling 2D layer wrapper
#[derive(Clone)]
pub struct AdaptiveAvgPool2D {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveAvgPool2D {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        AdaptiveAvgPool2D { out_h, out_w }
    }
}

impl Module for AdaptiveAvgPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::apply(
            Arc::new(crate::ops::AdaptiveAvgPool2D::new(self.out_h, self.out_w)),
            &[input.clone()],
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
