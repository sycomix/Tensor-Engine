#[cfg(test)]
mod tests {
    use hf_compat::transformer::read_builder_matrix;
    use hf_compat::unpickler::{TensorBuilder, TensorDType};
    use hf_compat::huggingface_loader::HugginfaceModel;
    use hf_compat::data_source::DataSource;
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use zip::write::FileOptions;
    use std::io::Write;

    #[test]
    fn test_read_builder_matrix_from_zip() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        let zip_path = path.join("pytorch_model-00001-of-00001.bin");
        // Instead create a LLaMA-style consolidated folder so DataSource::open uses filesystem directly
        let consolidated = path.join("consolidated.00");
        std::fs::create_dir_all(consolidated.join("data")).unwrap();
        let weights_path = consolidated.join("data").join("weights.bin");
        let mut wf = std::fs::File::create(&weights_path).unwrap();
        let rows = 2usize;
        let cols = 3usize;
        for r in 0..rows {
            for c in 0..cols {
                let v: f32 = (r * cols + c) as f32 + 0.5;
                wf.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        let ds = DataSource::LLaMASource(path.to_path_buf(), std::sync::Arc::new(vec![]));

        let builder = TensorBuilder::new(PathBuf::from("weights.bin"), "my_tensor", TensorDType::Float32, 3, 2, 3, 6, 0);

        let mat = read_builder_matrix(&builder, ds).expect("read failed");
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        assert_eq!(mat[(0,0)], 0.5f32);
        assert_eq!(mat[(1,2)], 5.5f32);
    }

    #[test]
    fn test_decode_k4_buffer() {
        // Prepare bytes: two rows, 3 cols each -> each row needs ceil(3/2)=2 bytes
        let buf: Vec<u8> = vec![0x21, 0x43, 0x65, 0x87]; // arbitrary nibble values
        let m = hf_compat::transformer::decode_k4_buffer_to_matrix(&buf, 2, 3).unwrap();
        assert_eq!(m.shape(), &[2,3]);
    }

    #[test]
    fn test_decode_k4_with_scales() {
        // rows=2, cols=4 (group_size=4 -> one group per row)
        // row0 nibbles: 1,2,3,4 -> bytes: 0x21,0x43
        // row1 nibbles: 5,6,7,8 -> bytes: 0x65,0x87
        let buf: Vec<u8> = vec![0x21,0x43,0x65,0x87];
        // scales as (rows,1) - one scalar per row
        let mut s = ndarray::Array2::<f32>::zeros((2,1));
        s[(0,0)] = 0.5f32;
        s[(1,0)] = 0.25f32;
        let m = hf_compat::transformer::decode_k4_buffer_to_matrix_with_scales(&buf, 2, 4, Some(&s), None).unwrap();
        assert_eq!(m.shape(), &[2,4]);
        // Check first element: q=1 -> (1 - 0) * 0.5 = 0.5
        assert_eq!(m[(0,0)], 0.5f32);
        // Second row first element: q=5 -> 5 * 0.25 = 1.25
        assert_eq!(m[(1,0)], 1.25f32);
    }
}

