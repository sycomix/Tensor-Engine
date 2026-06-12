/*
 * Understands HuggingFace format for models, or well at least as much as we need to.
 */

use super::unpickler;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HugginfaceModelError {
    #[error("Error parsing JSON: {0}")]
    JSONError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("ZIP error: {0}")]
    ZIPError(#[from] zip::result::ZipError),
    #[error("Unpickler error: {0}")]
    UnpicklingError(#[from] unpickler::UnpicklingError),
}

#[allow(dead_code)]
pub struct HugginfaceModel {
    pub(crate) unpickles: Vec<(unpickler::Value, PathBuf)>,
    // (path, files, tensors)
    pub(crate) zip_file_contents: Vec<(PathBuf, BTreeSet<String>, BTreeSet<String>)>,
    pub(crate) unpickles_flattened: unpickler::Value,
    pub(crate) index: HugginfaceIndex,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HugginfaceConfig {
    #[serde(flatten)]
    pub text_config: Option<TextConfig>,
    
    // Flat structure fields (for models without nested config)
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f32>,
    pub architectures: Option<Vec<String>>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub torch_dtype: Option<String>,
    
    // Additional fields for multimodal models
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextConfig {
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f32>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<serde_json::Value>,
}

impl HugginfaceConfig {
    pub fn get_vocab_size(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.vocab_size).or(self.vocab_size)
    }
    
    pub fn get_hidden_size(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.hidden_size).or(self.hidden_size)
    }
    
    pub fn get_intermediate_size(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.intermediate_size).or(self.intermediate_size)
    }
    
    pub fn get_num_hidden_layers(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.num_hidden_layers).or(self.num_hidden_layers)
    }
    
    pub fn get_num_attention_heads(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.num_attention_heads).or(self.num_attention_heads)
    }
    
    pub fn get_max_position_embeddings(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.max_position_embeddings).or(self.max_position_embeddings)
    }
    
    pub fn get_rms_norm_eps(&self) -> Option<f32> {
        self.text_config.as_ref().and_then(|tc| tc.rms_norm_eps).or(self.rms_norm_eps)
    }
    
    pub fn get_num_key_value_heads(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.num_key_value_heads).or(self.num_key_value_heads)
    }
    
    pub fn get_head_dim(&self) -> Option<usize> {
        self.text_config.as_ref().and_then(|tc| tc.head_dim).or(self.head_dim)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HugginfaceIndex {
    metadata: HugginfaceIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HugginfaceIndexMetadata {
    total_size: usize,
}

impl HugginfaceModel {
    pub fn unpickle<P: AsRef<Path>>(path: P) -> Result<Self, HugginfaceModelError> {
        let path: &Path = path.as_ref();

        let mut unpickles = vec![];

        // Read config,json
        let config_json_path: PathBuf = path.join(crate::config::filenames::CONFIG_JSON);
        let config_json = std::fs::read_to_string(config_json_path)?;
        let _config: HugginfaceConfig = serde_json::from_str(&config_json)?;

        let index_json_path: PathBuf = path.join("pytorch_model.bin.index.json");
        let index_json = std::fs::read_to_string(index_json_path)?;
        let index: HugginfaceIndex = serde_json::from_str(&index_json)?;

        // List all .bin files that contain the weights.
        let mut weight_files: Vec<PathBuf> = vec![];
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().unwrap_or_default() == "bin" {
                weight_files.push(path);
            }
        }

        // List all files in said zips
        let mut unpickles2 = vec![];
        let mut zip_file_contents = vec![];
        for file in weight_files.iter() {
            let mut files_in_zip = BTreeSet::new();
            let mut tensors_in_zip = BTreeSet::new();
            let reader = std::io::BufReader::new(std::fs::File::open(file)?);
            let mut archive = zip::ZipArchive::new(reader)?;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                // Remove the first directory.
                let file2 = remove_first_directory(file.name());
                files_in_zip.insert(file2.to_str().unwrap().to_string());
                // data.pkl
                if file.name().ends_with("data.pkl") {
                    let mut data_unzipped: Vec<u8> = vec![];
                    file.read_to_end(&mut data_unzipped)?;
                    let unpickled = unpickler::unpickle(&data_unzipped)?;
                    for tensor in unpickled.keys() {
                        tensors_in_zip.insert(tensor.to_string());
                    }
                    unpickles2.push(unpickled.clone());
                    unpickles.push((unpickled, file.name().to_string().into()))
                }
            }
            zip_file_contents.push((file.clone(), files_in_zip, tensors_in_zip));
        }
        let unpickles2: Vec<unpickler::Value> = unpickles.iter().map(|(v, _)| v.clone()).collect();
        let unpickles_flattened = unpickler::Value::merge_dicts(&unpickles2);

        Ok(HugginfaceModel {
            unpickles,
            unpickles_flattened,
            zip_file_contents,
            index,
        })
    }
}

pub fn remove_first_directory<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    let mut components = vec![];
    for component in path.components().skip(1) {
        components.push(component);
    }
    PathBuf::from(components.iter().collect::<PathBuf>())
}
