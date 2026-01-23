#[cfg(test)]
mod tests {
    use hf_compat::data_source::DataSource;
    // no extra imports needed
    use tempfile::tempdir;

    #[test]
    fn test_from_inferred_vicuna() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        let config = r#"{ "vocab_size": 1, "hidden_size": 1, "intermediate_size": 1, "num_hidden_layers": 1, "num_attention_heads": 1, "max_position_embeddings": 1, "rms_norm_eps": 1.0, "architectures": ["Test"], "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32" }"#;
        std::fs::write(path.join("config.json"), config).unwrap();
        let index = r#"{ "metadata": { "total_size": 0 }, "weight_map": {} }"#;
        std::fs::write(path.join("pytorch_model.bin.index.json"), index).unwrap();

        let ds = DataSource::from_inferred_source(path).unwrap();
        // Should be vicuna
        match ds {
            DataSource::VicunaSource(_, _, _) => {}
            _ => panic!("expected vicuna source"),
        }
    }
}
