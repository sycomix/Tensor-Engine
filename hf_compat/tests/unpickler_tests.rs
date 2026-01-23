#[cfg(test)]
mod tests {
    use hf_compat::unpickler;

    #[test]
    fn test_unpickle_empty_dict() {
        // Pickle protocol 2 empty dict: \x80\x02}\x2e
        let data = b"\x80\x02}.";
        let val = unpickler::unpickle(data).expect("unpickle failed");
        match val {
            unpickler::Value::Dict(d) => assert!(d.is_empty()),
            _ => panic!("Expected dict"),
        }
    }

    #[test]
    fn test_hf_loader_minimal_zip() {
        use std::fs::File;
        use std::io::Write;
        use tempfile::tempdir;

        // create temporary dir with config.json and index.json and a zipped weight file
        let dir = tempdir().unwrap();
        let path = dir.path();

        let config = r#"{ "vocab_size": 1, "hidden_size": 1, "intermediate_size": 1, "num_hidden_layers": 1, "num_attention_heads": 1, "max_position_embeddings": 1, "rms_norm_eps": 1.0, "architectures": ["Test"], "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32" }"#;
        std::fs::write(path.join("config.json"), config).unwrap();

        let index = r#"{ "metadata": { "total_size": 0 }, "weight_map": {} }"#;
        std::fs::write(path.join("pytorch_model.bin.index.json"), index).unwrap();

        // create a zip file named model-00001-of-00001.bin with a data.pkl that is empty dict
        let zip_path = path.join("pytorch_model-00001-of-00001.bin");
        let f = File::create(&zip_path).unwrap();
        let mut zip = zip::ZipWriter::new(f);
        let options = zip::write::FileOptions::default();
        zip.start_file("0/data.pkl", options).unwrap();
        zip.write_all(b"\x80\x02}.").unwrap();
        zip.finish().unwrap();

        let hm = hf_compat::huggingface_loader::HugginfaceModel::unpickle(path).unwrap();
        // Should have found one zipped file entry
        assert!(hm.zip_file_count() > 0);
    }
}
