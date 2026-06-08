use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelParams {
    #[serde(alias = "hidden_size")]
    pub dim: usize,
    #[serde(alias = "num_attention_heads")]
    pub n_heads: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(alias = "rms_norm_eps")]
    pub norm_eps: f64,
    pub vocab_size: i64,
    #[serde(alias = "num_key_value_heads")]
    pub n_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    #[serde(alias = "rope_theta")]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl ModelParams {
    pub fn eos_token_ids(&self) -> Vec<i64> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_i64().unwrap_or(0)]
            }
            Some(serde_json::Value::Array(arr)) => {
                arr.iter()
                    .filter_map(|v| v.as_i64())
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}
