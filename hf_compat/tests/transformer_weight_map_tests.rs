#[cfg(test)]
mod tests {
    use hf_compat::transformer::Transformer;
    use hf_compat::embedding::Embedding;
    use ndarray::Array2;

    #[test]
    fn test_apply_weight_map() {
        let rows = 4usize;
        let cols = 8usize;
        let data = vec![0.0f32; rows * cols];
        let emb = Embedding { rows, cols, data };
        let mut t = Transformer::new(emb, cols, 2, 2, 16, hf_compat::transformer::DataSettings::new());

        // create weight matrices for layer 0 q,k,v,o and fc1/fc2
        let mut map = std::collections::HashMap::new();
        let w = Array2::from_elem((cols, cols), 1.234f32);
        map.insert("model.layers.0.self_attn.q_proj.weight".to_string(), w.clone());
        map.insert("model.layers.0.self_attn.k_proj.weight".to_string(), w.clone());
        map.insert("model.layers.0.self_attn.v_proj.weight".to_string(), w.clone());
        map.insert("model.layers.0.self_attn.o_proj.weight".to_string(), w.clone());
        let wf1 = Array2::from_elem((cols * 4, cols), 2.0f32);
        map.insert("model.layers.0.mlp.fc1.weight".to_string(), wf1.clone());
        let wf2 = Array2::from_elem((cols, cols * 4), 3.0f32);
        map.insert("model.layers.0.mlp.fc2.weight".to_string(), wf2.clone());

        t.apply_weight_map(&map);

        // Check that q_proj weight was updated
        let qw = &t.blocks[0].attn.wq.weight;
        assert_eq!(qw[(0,0)], 1.234f32);
        let w1 = &t.blocks[0].ffn.w1.weight;
        assert_eq!(w1.shape(), wf1.shape());

        // Test bias assignment
        let mut map2 = map.clone();
        let b = ndarray::Array2::from_elem((cols, 1), 0.1f32);
        map2.insert("model.layers.0.mlp.fc1.bias".to_string(), b.clone());
        t.apply_weight_map(&map2);
        assert!(t.blocks[0].ffn.w1.bias.is_some());

        // Test alternative naming and c_attn concatenated splitting
        let mut big = ndarray::Array2::from_elem((cols * 3, cols), 0.5f32);
        let mut map3 = std::collections::HashMap::new();
        map3.insert("transformer.h.0.attn.c_attn.weight".to_string(), big.clone());
        t.apply_weight_map(&map3);
        // q,w should be set
        assert_eq!(t.blocks[0].attn.wq.weight[(0,0)], 0.5f32);
    }
}
