#[cfg(test)]
#[cfg(feature = "hf_compat")]
mod tests {
    use tensor_engine::hf_bridge::embedding_to_tensor;

    #[test]
    fn test_embedding_conversion() {
        let rows = 2usize;
        let cols = 3usize;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let he = tensor_engine::hf_compat::Embedding { rows, cols, data };
        let t = embedding_to_tensor(&he).expect("conversion failed");
        assert_eq!(t.lock().storage.shape(), vec![rows, cols]);
        let arr = t.lock().storage.to_f32_array();
        assert_eq!(arr[[0, 0]], 1.0f32);
        assert_eq!(arr[[1, 2]], 6.0f32);
    }
}
