#[cfg(feature = "safe_tensors")]
pub mod safetensors_loader;

#[cfg(not(feature = "safe_tensors"))]
mod safetensors_loader {}

pub mod tokenizers;
pub mod streaming_dataloader;
pub use streaming_dataloader::{
    StreamingDataLoader, StreamingDataLoaderConfig, StreamingCSVParser, StreamingJSONLParser,
};

pub mod pytorch_loader;

pub mod image;

#[cfg(feature = "vision")]
pub mod image_text_dataloader;
#[cfg(not(feature = "vision"))]
mod image_text_dataloader {}

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(not(feature = "audio"))]
mod audio {}

#[cfg(feature = "audio")]
pub mod dataloader;
#[cfg(not(feature = "audio"))]
mod dataloader {}
