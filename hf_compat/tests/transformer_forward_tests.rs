#[cfg(test)]
mod tests {
    use hf_compat::transformer::{Transformer, DataSettings};
    use hf_compat::embedding::Embedding;

    #[test]
    fn test_transformer_forward() {
        // Make tiny embedding: vocab 4, dim 8
        let rows = 4usize;
        let cols = 8usize;
        let mut data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[r * cols + c] = (r as f32) * 0.1 + (c as f32) * 0.01;
            }
        }
        let emb = Embedding { rows, cols, data };
        let ds = DataSettings::new();
        let t = Transformer::new(emb, cols, 2, 2, 16, ds);
        let mut caches = t.make_caches();
        let tokens = vec![0usize, 1usize, 2usize];
        let out = t.forward(&tokens, &mut caches);
        // out should be (1, dim)
        assert_eq!(out.nrows(), 1);
        assert_eq!(out.ncols(), cols);
    }
}
