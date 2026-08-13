//! Hugging Face supported-model matrix.
//!
//! This module is the single source of truth for which Hugging Face model
//! families Tensor Engine claims to support. The canonical artifact lives in
//! `conformance/hf_model_matrix.json` and is embedded at compile time so that
//! server responses and documentation always reflect the same, version-pinned
//! contract.
//!
//! Per the HF readiness plan (Phase 0): every model claim must be explicit,
//! version-pinned, and testable. Unknown `model_type` values are rejected by
//! the config/loader instead of silently defaulting to another architecture.

use serde::{Deserialize, Serialize};

/// Serialized contents of `conformance/hf_model_matrix.json`.
pub const HF_MODEL_MATRIX_JSON: &str = include_str!("../conformance/hf_model_matrix.json");

/// Valid status values for matrix entries.
pub const STATUS_NOT_SUPPORTED: &str = "not_supported";
pub const STATUS_PARTIAL: &str = "partial";
pub const STATUS_SUPPORTED: &str = "supported";

/// The parsed model matrix as served by the server and consumed by tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelMatrix {
    pub schema_version: u32,
    pub scope: String,
    #[serde(default)]
    pub description: String,
    pub architectures: Vec<HfModelMatrixEntry>,
}

/// A single supported model family claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelMatrixEntry {
    /// Stable identifier used in fixture IDs and reports.
    pub id: String,
    /// The `model_type` value accepted from `config.json`.
    pub architecture: String,
    /// Upstream transformers architecture names the model may claim.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Pinned upstream config schema/version the claim is made against.
    pub upstream_config_version: String,
    /// Tokenizer formats covered by the claim.
    pub tokenizer_types: Vec<String>,
    /// Dtypes the weights may be stored in and are executed as.
    pub supported_dtypes: Vec<String>,
    /// Attention variants implemented for the family.
    pub attention_variants: Vec<String>,
    /// RoPE/positional semantics covered by the claim.
    pub rope: String,
    pub tied_embeddings: String,
    /// One of `not_supported`, `partial`, `supported`.
    pub training_status: String,
    /// One of `not_supported`, `partial`, `supported`.
    pub inference_status: String,
    /// IDs of fixtures that carry compatibility-contract evidence.
    pub fixture_ids: Vec<String>,
}

impl HfModelMatrix {
    /// Parse the embedded matrix; panics only if the checked-in artifact is malformed.
    pub fn parse() -> Result<Self, String> {
        serde_json::from_str(HF_MODEL_MATRIX_JSON)
            .map_err(|error| format!("invalid conformance/hf_model_matrix.json: {}", error))
    }

    /// The set of `model_type` values the matrix claims support for.
    pub fn model_types(&self) -> Vec<String> {
        self.architectures
            .iter()
            .map(|entry| entry.architecture.clone())
            .collect()
    }

    /// Validate the embedded matrix against the compatibility contract:
    /// required fields present, statuses restricted, and unique ids/model types.
    pub fn validate(&self) -> Result<(), String> {
        const STATUSES: [&str; 3] = [STATUS_NOT_SUPPORTED, STATUS_PARTIAL, STATUS_SUPPORTED];

        if self.architectures.is_empty() {
            return Err("matrix must declare at least one architecture".to_string());
        }
        let mut ids = std::collections::HashSet::new();
        let mut model_types = std::collections::HashSet::new();
        for entry in &self.architectures {
            if entry.id.trim().is_empty() {
                return Err("matrix entry has an empty 'id'".to_string());
            }
            if !ids.insert(&entry.id) {
                return Err(format!("duplicate matrix entry id '{}'", entry.id));
            }
            if !model_types.insert(&entry.architecture) {
                return Err(format!(
                    "duplicate matrix architecture '{}'",
                    entry.architecture
                ));
            }
            if !STATUSES.contains(&entry.training_status.as_str()) {
                return Err(format!(
                    "entry '{}' has invalid training_status '{}'",
                    entry.id, entry.training_status
                ));
            }
            if !STATUSES.contains(&entry.inference_status.as_str()) {
                return Err(format!(
                    "entry '{}' has invalid inference_status '{}'",
                    entry.id, entry.inference_status
                ));
            }
            for field in [&entry.architecture, &entry.upstream_config_version] {
                if field.trim().is_empty() {
                    return Err(format!("entry '{}' has an empty required field", entry.id));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_matrix_parses() {
        let matrix = HfModelMatrix::parse().expect("embedded matrix must parse");
        assert_eq!(matrix.schema_version, 1);
        assert!(!matrix.architectures.is_empty());
    }

    #[test]
    fn embedded_matrix_passes_validation() {
        let matrix = HfModelMatrix::parse().unwrap();
        matrix
            .validate()
            .expect("embedded matrix must satisfy the contract");
    }

    #[test]
    fn matrix_covers_the_plan_families() {
        let matrix = HfModelMatrix::parse().unwrap();
        let model_types = matrix.model_types();
        for required in ["llama", "mistral", "qwen2", "qwen3", "gemma2", "phi3"] {
            assert!(
                model_types.iter().any(|m| m == required),
                "matrix is missing required decoder-only family '{}'",
                required
            );
        }
    }
}
