use crate::data_source::DataSource;
use crate::unpickler::TensorBuilder;
use crate::unpickler::TensorDType;
use crate::unpickler::UnpicklingError;
use half::f16;
use std::io::Read;

#[derive(Debug)]
pub struct Embedding {
    pub rows: usize,
    pub cols: usize,
    // row-major f32 matrix (rows * cols)
    pub data: Vec<f32>,
}

impl Embedding {
    pub fn from_unpickled(data_source: DataSource) -> Result<Self, UnpicklingError> {
        let mut builders: Vec<TensorBuilder> = vec![];
        let unpickled = data_source.unpickled();
        for unpickle in unpickled.iter() {
            let (name, val) = match unpickle.get_str_key2("tok_embeddings.weight", "model.embed_tokens.weight") {
                Some(v) => v,
                None => {
                    return Err(UnpicklingError::MissingField("tok_embeddings.weight/model.embed_tokens.weight".to_string()))
                }
            };
            builders.push(val.to_tensor_builder(name).ok_or(UnpicklingError::InvalidTensorData)?);
        }

        // We support only the simple case: single builder or concatenated columns
        // Compute total cols and rows
        if builders.is_empty() {
            return Err(UnpicklingError::MissingField("no embedding builders".to_string()));
        }
        let mut total_cols: i64 = 0;
        let expected_rows = builders[0].rows;
        let expected_dtype = builders[0].dtype;
        for b in builders.iter() {
            total_cols += b.cols;
            if b.rows != expected_rows {
                return Err(UnpicklingError::UnpicklingError("inconsistent builder rows".to_string()));
            }
            if b.dtype != expected_dtype {
                return Err(UnpicklingError::UnpicklingError("inconsistent builder dtype".to_string()));
            }
        }

        let rows = expected_rows as usize;
        let cols = total_cols as usize;
        let mut data: Vec<f32> = vec![0.0; rows * cols];

        let mut col_offset = 0usize;
        for (idx, builder) in builders.iter().enumerate() {
            let path = std::path::PathBuf::from("data").join(&builder.src_path);
            let mut f = data_source.open(path.clone(), &builder.tensor_name, idx).map_err(|e| UnpicklingError::UnpicklingError(format!("IO: {}", e)))?;
            // Read once per row
            let nbytes = builder.dtype.bytes_for_nvalues(builder.cols as usize);
            let mut buf: Vec<u8> = vec![0u8; nbytes];
            // Seek to offset (already handled by DataSource open which returns a cursor at start)
            use std::io::Seek;
            let offset_bytes: i64 = builder.offset * builder.dtype.bytes_for_nvalues(1) as i64;
            f.seek(std::io::SeekFrom::Current(offset_bytes)).map_err(|e| UnpicklingError::UnpicklingError(format!("seek error: {}", e)))?;
            for r in 0..builder.rows as usize {
                f.read_exact(&mut buf).map_err(|e| UnpicklingError::UnpicklingError(format!("read error: {}", e)))?;
                match builder.dtype {
                    TensorDType::Float16 => {
                        // interpret buf as contiguous f16 values
                        let mut off = 0usize;
                        for c in 0..(builder.cols as usize) {
                            let v = f16::from_bits(u16::from_le_bytes([buf[off], buf[off+1]]));
                            data[r * cols + col_offset + c] = v.to_f32();
                            off += 2;
                        }
                    }
                    TensorDType::Float32 => {
                        let mut off = 0usize;
                        for c in 0..(builder.cols as usize) {
                            let v = f32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]);
                            data[r * cols + col_offset + c] = v;
                            off += 4;
                        }
                    }
                    _ => {
                        return Err(UnpicklingError::UnpicklingError("unsupported dtype in embedding".to_string()));
                    }
                }
            }
            col_offset += builder.cols as usize;
        }

        Ok(Embedding { rows, cols, data })
    }

    pub fn get_embedding(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.rows { return None; }
        Some(&self.data[idx * self.cols..(idx+1) * self.cols])
    }
}