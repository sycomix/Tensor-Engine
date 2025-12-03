#[cfg(feature = "safe_tensors")]
pub mod safetensors_loader;

#[cfg(not(feature = "safe_tensors"))]
// placeholder module to avoid missing module issues when feature disabled
mod safetensors_loader {}
