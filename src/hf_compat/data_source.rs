use super::huggingface_loader;
use super::huggingface_loader::HugginfaceModel;
use super::unpickler;
use super::unpickler::Value;
use std::io::{Cursor, Read, Seek};
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
    HuggingFaceError(#[from] crate::huggingface_loader::HugginfaceModelError),
    #[error("Unknown source")]
    UnknownSource,
}

// This is cloned a lot in transformers.rs, keep it cheap to clone
#[derive(Clone)]
pub enum DataSource {
    // The format used by original LLaMA release (consolidated.* folders)
    LLaMASource(PathBuf, Arc<Vec<Value>>),
    // The huggingface format used by some models: directory with config.json + .bin zip files
    VicunaSource(PathBuf, Arc<HugginfaceModel>, Arc<Vec<Value>>),
}

pub struct DataSourceFile {
    reader: Box<dyn ReadSeek>,
}

trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

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
                Ok(DataSourceFile { reader: Box::new(reader) })
            }
            DataSource::VicunaSource(path, model, _) => {
                if shard != 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Vicuna loader does not support shards",
                    ));
                }
                for (zipfile_name, contents, tensors) in model.zip_file_contents.iter() {
                    let name_str: &str = name.to_str().unwrap();
                    if contents.contains(name_str) && tensors.contains(tensor_name.as_ref()) {
                        let reader = std::fs::File::open(zipfile_name)?;
                        let mut archive = zip::ZipArchive::new(std::io::BufReader::new(reader))?;
                        // Find entry index
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
                        // Read the file into memory and return a cursor (Read + Seek)
                        let mut file = archive.by_index(idx)?;
                        let mut buf: Vec<u8> = Vec::with_capacity(file.size() as usize);
                        file.read_to_end(&mut buf)?;
                        return Ok(DataSourceFile { reader: Box::new(Cursor::new(buf)) });
                    }
                }
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("file not found: {:?}", path),
                ))
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
            let result = unpickler::unpickle(&bs)?;
            unpickle_results.push(result);
            part += 1;
        }
        Ok(Self::LLaMASource(path.to_path_buf(), Arc::new(unpickle_results)))
    }

    pub fn from_inferred_source<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path = path.as_ref();
        let params_path = path.join("params.json");
        let pytorch_model_path = path.join("pytorch_model.bin.index.json");
        if params_path.exists() {
            Self::from_llama_source(path)
        } else if pytorch_model_path.exists() {
            Self::from_vicuna_source(path)
        } else {
            Err(DataSourceError::UnknownSource)
        }
    }

    pub fn from_vicuna_source<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path = path.as_ref();
        let model = HugginfaceModel::unpickle(path)?;
        let unpickled: Vec<unpickler::Value> = vec![model.unpickles_flattened.clone()];
        Ok(DataSource::VicunaSource(path.to_path_buf(), Arc::new(model), Arc::new(unpickled)))
    }

    /// Construct a `DataSource` directly from a `HugginfaceModel` instance (useful in tests).
    pub fn from_hf_model(model: HugginfaceModel) -> Self {
        let unpickled: Vec<unpickler::Value> = vec![model.unpickles_flattened.clone()];
        DataSource::VicunaSource(PathBuf::from("."), Arc::new(model), Arc::new(unpickled))
    }

    pub fn need_to_do_antitranspose(&self) -> bool {
        match self {
            Self::LLaMASource(_, _) => false,
            Self::VicunaSource(_, _, _) => true,
        }
    }
}
