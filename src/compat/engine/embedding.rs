use super::data_source::DataSource;
use super::tensor::{FromPiecesDirection, Tensor, TensorBuilder};
use super::unpickler::UnpicklingError;

use std::collections::BTreeMap;

pub struct Embedding {
    wgts: BTreeMap<usize, Tensor>,
}

impl Embedding {
    pub fn from_unpickled(data_source: DataSource) -> Result<Self, UnpicklingError> {
        let mut builders: Vec<TensorBuilder> = vec![];

        let safetensor_builder = data_source
            .get_tensor_builder("tok_embeddings.weight")
            .or_else(|| data_source.get_tensor_builder("model.embed_tokens.weight"));

        if let Some(builder) = safetensor_builder {
            builders.push(builder);
        } else {
            let unpickled = data_source.unpickled();
            for unpickle in unpickled.iter() {
                let (name, val) = match unpickle
                    .get_str_key2("tok_embeddings.weight", "model.embed_tokens.weight")
                {
                    Some(val) => val,
                    None => {
                        return Err(UnpicklingError::MissingField(
                            "tok_embeddings.weight/model.embed_tokens.weight".to_string(),
                        ))
                    }
                };
                builders.push(
                    val.to_tensor_builder(name)
                        .ok_or(UnpicklingError::InvalidTensorData)?,
                );
            }
        }

        let tensor = TensorBuilder::load_from_pieces2(
            &builders,
            "tok_embeddings.weight",
            "model.embed_tokens.weight",
            data_source.clone(),
            FromPiecesDirection::Cols,
        )?;
        let num_embeddings = tensor.rows();

        let mut table: BTreeMap<usize, Tensor> = BTreeMap::new();
        for key in 0..num_embeddings {
            let row = tensor.row(key);
            table.insert(key as usize, row);
        }

        Ok(Self { wgts: table })
    }

    pub fn get_embedding(&self, idx: usize) -> &Tensor {
        self.wgts.get(&idx).unwrap()
    }
}
