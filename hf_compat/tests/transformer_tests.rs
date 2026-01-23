#[cfg(test)]
mod tests {
    use hf_compat::transformer::{Transformer, DataSettings};
    use hf_compat::embedding::Embedding;

    #[test]
    fn test_transformer_make_caches() {
        let emb = Embedding { rows: 4, cols: 8, data: vec![0.0f32; 32] };
        let ds = DataSettings::new();
        let t = Transformer::new(emb, 8, 2, 2, 512, ds);
        let caches = t.make_caches();
        assert_eq!(caches.layer_caches.len(), 2);
    }

    #[test]
    fn test_transformer_from_unpickled_minimal() {
        let emb = Embedding { rows: 4, cols: 8, data: vec![0.0f32; 32] };
        let ds = DataSettings::new();
        // Use a dummy path and rely on the minimal implementation
        let dummy_path = std::path::Path::new(".");
        let _data_source = hf_compat::data_source::DataSource::from_inferred_source(dummy_path);
        // This will error for missing files but our from_unpickled ignores that and returns Ok
        let tf = Transformer::from_unpickled(emb, ds, hf_compat::data_source::DataSource::from_inferred_source(".").unwrap_or_else(|_| hf_compat::data_source::DataSource::from_llama_source(".").unwrap_or_else(|_| panic!())));
        match tf {
            Ok(t) => {
                assert_eq!(t.dim, 8);
            }
            Err(_) => panic!("from_unpickled failed"),
        }
    }

    #[test]
    fn test_load_weights_respects_config() {
        let emb = Embedding { rows: 4, cols: 8, data: vec![0.0f32; 32] };
        let ds = DataSettings::new();
        let mut t = Transformer::new(emb, 8, 1, 1, 512, ds);

        let cfg = hf_compat::huggingface_loader::HugginfaceConfig {
            vocab_size: 1,
            hidden_size: 16,
            intermediate_size: 64,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            max_position_embeddings: 128,
            rms_norm_eps: 1.0,
            architectures: Vec::new(),
            bos_token_id: 0,
            eos_token_id: 2,
            torch_dtype: "float32".to_string(),
        };

        let model = hf_compat::huggingface_loader::HugginfaceModel::from_config_for_tests(cfg);
        let dsrc = hf_compat::data_source::DataSource::from_hf_model(model);
        let _ = t.load_weights_from_datasource(dsrc);

        assert_eq!(t.n_layers, 3usize);
        assert_eq!(t.n_heads, 4usize);
        assert_eq!(t.dim, 16usize);
        assert_eq!(t.head_dim, 4usize);
    }
}

