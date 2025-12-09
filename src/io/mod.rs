#[cfg(feature = "safe_tensors")]
pub mod safetensors_loader;

#[cfg(not(feature = "safe_tensors"))]
// placeholder module to avoid missing module issues when feature disabled
mod safetensors_loader {}
#[cfg(feature = "with_tokenizers")]
pub mod tokenizers;
#[cfg(not(feature = "with_tokenizers"))]
mod tokenizers {}

#[cfg(feature = "with_tch")]
pub mod pytorch_loader;
#[cfg(not(feature = "with_tch"))]
mod pytorch_loader {}

#[cfg(feature = "vision")]
pub mod image;
#[cfg(not(feature = "vision"))]
mod image {}

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(not(feature = "audio"))]
mod audio {}
#[cfg(feature = "audio")]
pub mod dataloader;
#[cfg(not(feature = "audio"))]
mod dataloader {}
