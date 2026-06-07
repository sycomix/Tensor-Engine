use super::huggingface_loader;
use super::huggingface_loader::HugginfaceModel;
use super::unpickler;
use super::unpickler::Value;
use crate::compat::engine::tensor::{TensorBuilder, TensorDType};
use ouroboros::self_referencing;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataSourceError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Unpickling error: {0}")]
    UnpicklingError(#[from] unpickler::UnpicklingError),
    #[error("HuggingFace error: {0}")]
    HuggingFaceError(#[from] huggingface_loader::HugginfaceModelError),
    #[error("Unknown source")]
    UnknownSource,
    #[error("Safetensor error: {0}")]
    SafeTensorError(String),
}

#[derive(Clone, Debug)]
pub struct SafeTensorMeta {
    pub dtype: TensorDType,
    pub rows: i64,
    pub cols: i64,
    pub offset: u64,
    pub length: u64,
    pub is_bf16: bool,
}

pub type SafeTensorIndex = HashMap<String, SafeTensorMeta>;

// This is cloned a lot in transformers.rs, keep it cheap to clone
#[derive(Clone)]
pub enum DataSource {
    // The format used by original LLaMA release, unzipped manually
    // instructions
    LLaMASource(PathBuf, Arc<Vec<Value>>),
    // The huggingface format used by Vicuna-13B
    VicunaSource(PathBuf, Arc<HugginfaceModel>, Arc<Vec<Value>>),
    // Safetensor format (model.safetensors)
    SafeTensorSource(PathBuf, PathBuf, u64, Arc<SafeTensorIndex>),
}

pub struct DataSourceFile {
    reader: Box<dyn ReadSeek>,
}

trait ReadSeek: Read + Seek {}

impl ReadSeek for std::fs::File {}
impl ReadSeek for std::io::Cursor<Vec<u8>> {}
impl ReadSeek for ZipFileSeekWrap {}

#[self_referencing]
struct ZipFileSeekWrap {
    zipfile: PathBuf,
    name: String,
    archive: zip::ZipArchive<std::io::BufReader<std::fs::File>>,
    #[borrows(mut archive)]
    #[not_covariant]
    reader: zip::read::ZipFile<'this, std::io::BufReader<std::fs::File>>,
}

impl Read for ZipFileSeekWrap {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.with_mut(|s| s.reader.read(buf))
    }
}

impl Seek for ZipFileSeekWrap {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.with_mut(|mut s| {
            let reader = &mut s.reader;
            match pos {
                std::io::SeekFrom::Start(_pos) => {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "ZipFileSeekWrap does not support SeekFrom::Start - zip archives require sequential reads",
                    ))
                }
                std::io::SeekFrom::End(_pos) => {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "ZipFileSeekWrap does not support SeekFrom::End - zip archives require sequential reads",
                    ))
                }
                std::io::SeekFrom::Current(pos) => {
                    std::io::copy(&mut reader.by_ref().take(pos as u64), &mut std::io::sink())
                }
            }
        })
    }
}

impl Read for DataSourceFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

impl Seek for DataSourceFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }
}

impl DataSource {
    pub fn unpickled(&self) -> &[unpickler::Value] {
        match self {
            DataSource::LLaMASource(_path, unpickled) => unpickled,
            DataSource::VicunaSource(_path, _model, unpickled) => unpickled,
            DataSource::SafeTensorSource(_, _, _, _) => &[],
        }
    }

    pub fn get_tensor_builder(&self, name: &str) -> Option<TensorBuilder> {
        match self {
            DataSource::SafeTensorSource(_, _, _, index) => {
                let meta = index.get(name)?;
                let cols = meta.cols;
                Some(TensorBuilder {
                    src_path: PathBuf::new(),
                    tensor_name: name.to_string(),
                    dtype: meta.dtype,
                    stride: cols,
                    rows: meta.rows,
                    cols,
                    nitems: meta.rows * meta.cols,
                    offset: 0,
                })
            }
            _ => None,
        }
    }

    pub fn open<S: AsRef<str>, P: AsRef<Path>>(
        &self,
        name: P,
        tensor_name: S,
        shard: usize,
    ) -> Result<DataSourceFile, std::io::Error> {
        let name: &Path = name.as_ref();
        match self {
            DataSource::LLaMASource(path, _) => {
                let base = PathBuf::from(format!("consolidated.{:02}", shard));
                let path = path.join(base).join(name);
                let reader = std::fs::File::open(path)?;
                Ok(DataSourceFile {
                    reader: Box::new(reader),
                })
            }
            DataSource::VicunaSource(path, model, _) => {
                if shard != 0 {
                    panic!("Vicuna loader does not support shards");
                }
                // Performance consideration: Multiple tensors from same zip file cause repeated decompression the
                // same data, if multiple tensors are in the same file.
                //
                // Archive format limitations (no seek) require decompression-based approach
                for (zipfile_name, contents, tensors) in model.zip_file_contents.iter() {
                    let name_str: &str = name.to_str().unwrap();
                    if contents.contains(name_str) && tensors.contains(tensor_name.as_ref()) {
                        let reader = std::io::BufReader::new(std::fs::File::open(zipfile_name)?);
                        let mut archive = zip::ZipArchive::new(reader)?;
                        let archive_len = archive.len();
                        let mut idx: usize = archive_len;
                        for i in 0..archive_len {
                            let file = archive.by_index(i)?;
                            let file = huggingface_loader::remove_first_directory(file.name());
                            if file == name {
                                idx = i;
                                break;
                            }
                        }
                        if idx == archive_len {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!("file not found: {:?}", name),
                            ));
                        }
                        return Ok(DataSourceFile {
                            reader: Box::new(
                                ZipFileSeekWrapBuilder {
                                    zipfile: zipfile_name.clone(),
                                    name: name.to_str().unwrap().to_string(),
                                    archive,
                                    reader_builder: move |archive| archive.by_index(idx).unwrap(),
                                }
                                .build(),
                            ),
                        });
                    }
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("file not found: {:?}", path),
                ));
            }
            DataSource::SafeTensorSource(_path, safetensor_path, data_start, index) => {
                let tensor_name = tensor_name.as_ref();
                let meta = index.get(tensor_name).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("tensor not found in safetensor: {}", tensor_name),
                    )
                })?;
                let file_offset = data_start + meta.offset;
                if meta.is_bf16 {
                    let mut file = std::fs::File::open(safetensor_path)?;
                    file.seek(std::io::SeekFrom::Start(file_offset))?;
                    let mut raw_data = vec![0u8; meta.length as usize];
                    file.read_exact(&mut raw_data)?;
                    let n = raw_data.len() / 2;
                    let mut f32_data = vec![0u8; n * 4];
                    for i in 0..n {
                        let bf16_bits = u16::from_le_bytes([raw_data[i * 2], raw_data[i * 2 + 1]]);
                        let f32_bits = (bf16_bits as u32) << 16;
                        f32_data[i * 4..(i + 1) * 4].copy_from_slice(&f32_bits.to_le_bytes());
                    }
                    Ok(DataSourceFile {
                        reader: Box::new(std::io::Cursor::new(f32_data)),
                    })
                } else {
                    let mut file = std::fs::File::open(safetensor_path)?;
                    file.seek(std::io::SeekFrom::Start(file_offset))?;
                    Ok(DataSourceFile {
                        reader: Box::new(file),
                    })
                }
            }
        }
    }

    pub fn from_llama_source<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path = path.as_ref();
        let mut unpickle_results: Vec<Value> = vec![];
        let mut part: usize = 0;
        loop {
            let model_path: PathBuf = path.into();
            let base_path = model_path.join(format!("consolidated.{:02}", part));
            // The data file is in consolidated.XX/data.pkl where XX is the part number.
            let full_path = base_path.join("data.pkl");
            let mut fs = match std::fs::File::open(&full_path) {
                Ok(fs) => fs,
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::NotFound {
                        break;
                    } else {
                        return Err(err.into());
                    }
                }
            };
            let mut bs = Vec::new();
            fs.read_to_end(&mut bs)?;
            std::mem::drop(fs);
            let result = unpickler::unpickle(&bs)?;
            unpickle_results.push(result);
            part += 1;
        }
        Ok(Self::LLaMASource(
            path.to_path_buf(),
            Arc::new(unpickle_results),
        ))
    }

    pub fn from_inferred_source<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path = path.as_ref();
        let params_path = path.join(crate::config::filenames::PARAMS_JSON);
        let pytorch_model_path = path.join("pytorch_model.bin.index.json");
        let safetensor_path = path.join("model.safetensors");
        if params_path.exists() {
            Self::from_llama_source(path)
        } else if pytorch_model_path.exists() {
            Self::from_vicuna_source(path)
        } else if safetensor_path.exists() {
            Self::from_safetensor_source(path)
        } else {
            Err(DataSourceError::UnknownSource)
        }
    }

    pub fn from_safetensor_source<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path = path.as_ref();
        let safetensor_path = path.join("model.safetensors");
        let mut file = std::fs::File::open(&safetensor_path)
            .map_err(|e| DataSourceError::SafeTensorError(format!("Failed to open safetensor: {}", e)))?;

        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)
            .map_err(|e| DataSourceError::SafeTensorError(format!("Failed to read header length: {}", e)))?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;

        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)
            .map_err(|e| DataSourceError::SafeTensorError(format!("Failed to read header: {}", e)))?;

        let header_str = std::str::from_utf8(&header_buf)
            .map_err(|e| DataSourceError::SafeTensorError(format!("Invalid UTF-8 in header: {}", e)))?;

        let parsed: serde_json::Value = serde_json::from_str(header_str)
            .map_err(|e| DataSourceError::SafeTensorError(format!("Invalid JSON in header: {}", e)))?;

        let obj = parsed.as_object()
            .ok_or_else(|| DataSourceError::SafeTensorError("Header is not a JSON object".to_string()))?;

        let mut index = SafeTensorIndex::new();

        for (tensor_name, tensor_info) in obj {
            if tensor_name == "__metadata__" {
                continue;
            }
            let info = tensor_info.as_object()
                .ok_or_else(|| DataSourceError::SafeTensorError(format!("Invalid tensor info for {}", tensor_name)))?;

            let dtype_str = info.get("dtype").and_then(|v| v.as_str())
                .ok_or_else(|| DataSourceError::SafeTensorError(format!("Missing dtype for {}", tensor_name)))?;
            let shape = info.get("shape").and_then(|v| v.as_array())
                .ok_or_else(|| DataSourceError::SafeTensorError(format!("Missing shape for {}", tensor_name)))?;
            let data_offsets = info.get("data_offsets").and_then(|v| v.as_array())
                .ok_or_else(|| DataSourceError::SafeTensorError(format!("Missing data_offsets for {}", tensor_name)))?;

            let dtype = match dtype_str {
                "F32" => TensorDType::Float32,
                "F16" => TensorDType::Float16,
                "BF16" => TensorDType::Float16,
                _ => return Err(DataSourceError::SafeTensorError(
                    format!("Unsupported dtype: {}", dtype_str),
                )),
            };

            let (rows, cols) = if shape.len() == 1 {
                (1, shape[0].as_i64().unwrap_or(1))
            } else if shape.len() >= 2 {
                let mut r = 1i64;
                for i in 0..shape.len() - 1 {
                    r *= shape[i].as_i64().unwrap_or(1);
                }
                (r, shape[shape.len() - 1].as_i64().unwrap_or(1))
            } else {
                (1, 1)
            };

            let offset = data_offsets[0].as_u64().unwrap_or(0);
            let end = data_offsets[1].as_u64().unwrap_or(0);
            let length = end - offset;

            let is_bf16 = dtype_str == "BF16";
            let dtype = if is_bf16 { TensorDType::Float32 } else { dtype };

            index.insert(tensor_name.clone(), SafeTensorMeta {
                dtype,
                rows,
                cols,
                offset,
                length,
                is_bf16,
            });
        }

        let data_start = 8u64 + header_len as u64;

        Ok(DataSource::SafeTensorSource(
            path.to_path_buf(),
            safetensor_path,
            data_start,
            Arc::new(index),
        ))
    }

    pub fn from_vicuna_source<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path = path.as_ref();
        let model = HugginfaceModel::unpickle(path)?;
        let unpickled: Vec<unpickler::Value> = vec![model.unpickles_flattened.clone()];
        Ok(DataSource::VicunaSource(
            path.to_path_buf(),
            Arc::new(model),
            Arc::new(unpickled),
        ))
    }

    pub fn need_to_do_antitranspose(&self) -> bool {
        match self {
            Self::LLaMASource(_, _) => false,
            Self::VicunaSource(_, _, _) => true,
            Self::SafeTensorSource(_, _, _, _) => false,
        }
    }
}
