use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use thiserror::Error;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TensorBuilder {
    pub(crate) src_path: PathBuf,
    pub(crate) tensor_name: String,
    pub(crate) dtype: TensorDType,
    pub(crate) stride: i64,
    pub(crate) rows: i64,
    pub(crate) cols: i64,
    pub(crate) nitems: i64,
    pub(crate) offset: i64,
}

impl TensorBuilder {
    /// Test-friendly constructor
    pub fn new<S: Into<PathBuf>, T: Into<String>>(
        src_path: S,
        tensor_name: T,
        dtype: TensorDType,
        stride: i64,
        rows: i64,
        cols: i64,
        nitems: i64,
        offset: i64,
    ) -> Self {
        TensorBuilder {
            src_path: src_path.into(),
            tensor_name: tensor_name.into(),
            dtype,
            stride,
            rows,
            cols,
            nitems,
            offset,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TensorDType {
    K4BitQuantization,
    Float16,
    Float32,
}

impl TensorDType {
    pub fn bytes_for_nvalues(&self, nvalues: usize) -> usize {
        match self {
            TensorDType::K4BitQuantization => {
                if nvalues % 2 == 1 {
                    nvalues / 2 + 1
                } else {
                    nvalues / 2
                }
            }
            TensorDType::Float16 => nvalues * 2,
            TensorDType::Float32 => nvalues * 4,
        }
    }
}

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Invalid stride: {0}")]
    InvalidStride(i64),
}

#[derive(Error, Debug)]
pub enum UnpicklingError {
    #[error("Unpickling error: {0}")]
    UnpicklingError(String),
    #[error("UTF-8 decoding error")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[error("Missing field")]
    MissingField(String),
    #[error("Tensor conversion operation failed")]
    TensorError(#[from] TensorError),
    #[error("Data has incorrect format to be converted to a tensor")]
    InvalidTensorData,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Value {
    Mark(usize),
    String(String),
    Global(String, String), // module name, attribute name
    Integer64(i64),
    Tuple(Vec<Value>),
    PersistentId(Box<Value>),
    Bool(bool),
    Reduce(Box<Value>, Box<Value>),
    Dict(BTreeMap<Value, Value>),
}

impl Value {
    pub fn get(&self, key: &Value) -> Option<&Value> {
        match self {
            Value::Dict(d) => d.get(key),
            _ => None,
        }
    }

    pub fn get_str_key<S: AsRef<str>>(&self, key: S) -> Option<&Value> {
        self.get(&Value::String(key.as_ref().to_string()))
    }

    pub fn get_str_key2<S: AsRef<str>, S2: AsRef<str>>(
        &self,
        key: S,
        key2: S2,
    ) -> Option<(String, &Value)> {
        let key = key.as_ref();
        let key2 = key2.as_ref();
        match self.get_str_key(key) {
            Some(v) => Some((key.to_string(), v)),
            None => match self.get_str_key(key2) {
                Some(v) => Some((key2.to_string(), v)),
                None => None,
            },
        }
    }

    pub fn keys(&self) -> BTreeSet<String> {
        match self {
            Value::Dict(d) => {
                let mut result = BTreeSet::new();
                for (k, _v) in d.iter() {
                    if let Value::String(s) = k {
                        result.insert(s.clone());
                    }
                }
                result
            }
            _ => BTreeSet::new(),
        }
    }

    pub fn merge_dicts(dicts: &[Self]) -> Self {
        if dicts.is_empty() {
            return Value::Dict(BTreeMap::new());
        }
        let mut result = dicts[0].clone();
        for dict in dicts.iter().skip(1) {
            match (&result, dict) {
                (Value::Dict(_), Value::Dict(d2)) => {
                    if let Value::Dict(ref mut d1) = result {
                        for (k, v) in d2 {
                            d1.insert(k.clone(), v.clone());
                        }
                    }
                }
                _ => log::warn!("merge_dicts: encountered non-dict value, skipping"),
            }
        }
        result
    }

    pub fn get_global(&self) -> Option<(&str, &str)> {
        match self {
            Value::Global(module_name, attribute_name) => Some((module_name, attribute_name)),
            _ => None,
        }
    }

    pub fn get_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn get_int64(&self) -> Option<i64> {
        match self {
            Value::Integer64(i) => Some(*i),
            _ => None,
        }
    }

    pub fn get_persistent_id(&self) -> Option<&Value> {
        match self {
            Value::PersistentId(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_tuple(&self) -> Option<&[Value]> {
        match self {
            Value::Tuple(v) => Some(v),
            _ => None,
        }
    }

    pub fn to_tensor_builder(&self, tensor_name: String) -> Option<TensorBuilder> {
        match self {
            Value::Reduce(call, args) => match **call {
                Value::Global(ref module_name, ref attribute_name) => {
                    if module_name == "torch._utils" && attribute_name == "_rebuild_tensor_v2" {
                        match **args {
                            Value::Tuple(ref args) => self.to_tensor_builder2(tensor_name, args),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn to_tensor_builder2(&self, tensor_name: String, args: &[Value]) -> Option<TensorBuilder> {
        if args.len() == 6 {
            Self::to_tensor_builder2_6items(tensor_name, args)
        } else {
            None
        }
    }

    fn to_tensor_builder2_6items(tensor_name: String, args: &[Value]) -> Option<TensorBuilder> {
        let storagev: &Value = args[0].get_persistent_id()?;
        let storage_args: &[Value] = storagev.get_tuple()?;
        let storage_mark: &str = storage_args[0].get_str()?;
        if storage_mark != "storage" {
            return None;
        }

        let (storage_module, storage_type) = storage_args[1].get_global()?;
        if storage_module != "torch" {
            return None;
        }
        let dtype: TensorDType = match storage_type {
            "HalfStorage" => TensorDType::Float16,
            _ => {
                return None;
            }
        };
        let storage_filename: &str = storage_args[2].get_str()?;
        let nitems: i64 = storage_args[4].get_int64()?;

        let offset: i64 = args[1].get_int64()?;

        let shape: &[Value] = args[2].get_tuple()?;
        let stride: &[Value] = args[3].get_tuple()?;

        if shape.len() != 2 && shape.len() != 1 {
            return None;
        }
        if stride.len() != 2 && stride.len() != 1 {
            return None;
        }

        let (rows, cols) = if shape.len() == 2 {
            (shape[0].get_int64()?, shape[1].get_int64()?)
        } else {
            let cols = shape[0].get_int64()?;
            (1, cols)
        };

        let (row_stride, col_stride) = if stride.len() == 1 {
            let (r, c) = (stride[0].get_int64()?, 1);
            if r != 1 {
                return None;
            }
            (r, c)
        } else {
            (stride[0].get_int64()?, stride[1].get_int64()?)
        };

        if col_stride != 1 {
            return None;
        }
        if row_stride != cols && stride.len() == 2 {
            return None;
        }

        Some(TensorBuilder {
            src_path: PathBuf::from(storage_filename),
            tensor_name,
            dtype,
            stride: row_stride,
            rows,
            cols,
            nitems,
            offset,
        })
    }

    pub fn debug_print(&self) {
        self.debug_print_go(0);
    }

    fn debug_print_go(&self, indent: usize) {
        if indent > 0 {
            // log::debug!("{:indent$}", "", indent = indent);
        }
        match self {
            Value::Mark(_) => {
                // log::debug!("MARK");
            }
            Value::String(_s) => {
                // log::debug!("STRING {:?}", _s);
            }
            Value::Global(_module_name, _attribute_name) => {
                // log::debug!("GLOBAL {:?} {:?}", _module_name, _attribute_name);
            }
            Value::Integer64(_i) => {
                // log::debug!("INTEGER {:?}", _i);
            }
            Value::Tuple(v) => {
                // log::debug!("TUPLE");
                for i in v {
                    i.debug_print_go(indent + 2);
                }
            }
            Value::PersistentId(v) => {
                // log::debug!("PERSISTENT_ID");
                v.debug_print_go(indent + 2);
            }
            Value::Bool(_b) => {
                // log::debug!("BOOL {:?}", _b);
            }
            Value::Reduce(v1, v2) => {
                // log::debug!("REDUCE");
                v1.debug_print_go(indent + 2);
                v2.debug_print_go(indent + 2);
            }
            Value::Dict(d) => {
                // log::debug!("DICT");
                for (k, v) in d {
                    k.debug_print_go(indent + 2);
                    v.debug_print_go(indent + 2);
                }
            }
        }
    }
}

pub fn unpickle(bytes: &[u8]) -> Result<Value, UnpicklingError> {
    if bytes.len() < 2 {
        return Err(UnpicklingError::UnpicklingError(
            "Data is too short to be a pickle".to_string(),
        ));
    }

    if bytes[0] != 128 || bytes[1] != 2 {
        return Err(UnpicklingError::UnpicklingError(
            "No magic header using Pickle 2 protocol".to_string(),
        ));
    }

    let mut memo: BTreeMap<u32, Value> = BTreeMap::new();
    let mut stack: Vec<Value> = vec![];

    let mut bytes: &[u8] = &bytes[2..];
    while !bytes.is_empty() {
        let frame_opcode = bytes[0];
        if frame_opcode == 125 {
            stack.push(Value::Dict(BTreeMap::new()));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 113 {
            if bytes.len() < 2 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINPUT".to_string(),
                ));
            }
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling BINPUT".to_string(),
                ));
            }
            let key = bytes[1];
            memo.insert(key as u32, stack.last().unwrap().clone());
            bytes = &bytes[2..];
            continue;
        }
        if frame_opcode == 40 {
            stack.push(Value::Mark(stack.len()));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 88 {
            if bytes.len() < 5 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINUNICODE".to_string(),
                ));
            }
            let len = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
            if bytes.len() < 5 + len as usize {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINUNICODE".to_string(),
                ));
            }
            let string = std::str::from_utf8(&bytes[5..5 + len as usize])?;
            stack.push(Value::String(string.to_string()));
            bytes = &bytes[5 + len as usize..];
            continue;
        }
        if frame_opcode == 99 {
            bytes = &bytes[1..];
            let mut module_name = String::new();
            while !bytes.is_empty() && bytes[0] != 10 {
                module_name.push(bytes[0] as char);
                bytes = &bytes[1..];
                if bytes.is_empty() {
                    return Err(UnpicklingError::UnpicklingError(
                        "Unexpected end of data while handling GLOBAL".to_string(),
                    ));
                }
            }
            bytes = &bytes[1..];
            let mut attribute_name = String::new();
            while !bytes.is_empty() && bytes[0] != 10 {
                attribute_name.push(bytes[0] as char);
                bytes = &bytes[1..];
                if bytes.is_empty() {
                    return Err(UnpicklingError::UnpicklingError(
                        "Unexpected end of data while handling GLOBAL".to_string(),
                    ));
                }
            }
            bytes = &bytes[1..];
            stack.push(Value::Global(module_name, attribute_name));
            continue;
        }
        if frame_opcode == 78 {
            stack.push(Value::Tuple(Vec::new()));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 46 {
            // STOP
            break;
        }

        // For the purposes of this compat module, support only a small subset of opcodes
        // that we need for simple picks (empty dicts and very small structures).
        // If we encounter unknown opcodes, return an error to be explicit.
        return Err(UnpicklingError::UnpicklingError(format!(
            "Unsupported opcode encountered: {}",
            frame_opcode
        )));
    }

    if stack.is_empty() {
        return Err(UnpicklingError::UnpicklingError(
            "Pickle parsed to empty stack".to_string(),
        ));
    }

    Ok(stack.pop().unwrap())
}
