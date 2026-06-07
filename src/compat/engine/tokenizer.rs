use super::protomodels::sentencepiece_model::model_proto::sentence_piece;
use super::protomodels::sentencepiece_model::ModelProto;
use protobuf::Message;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::Path;
use thiserror::Error;

#[cfg(feature = "with_tokenizers")]
use tokenizers::Tokenizer as HFTokenizer;

pub type TokenId = i32;

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pieces: BTreeMap<String, Piece>,
    #[cfg(feature = "with_tokenizers")]
    hf: Option<HFTokenizer>,
    bpe_merges: Option<Vec<(String, String)>>,
    byte_decoder: Option<HashMap<u32, u8>>,
    byte_encoder: Option<HashMap<u8, u32>>,
}

#[derive(Clone, Debug, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
}

#[derive(Clone, Debug)]
pub struct Piece {
    _tp: PieceType,
    _score: f32,
    idx: usize,
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error")]
    IoError(#[from] std::io::Error),
    #[error("Protobuf error")]
    ProtobufError(#[from] protobuf::Error),
    #[error("Unknown piece type")]
    UnknownPieceType(String),
    #[error("HuggingFace tokenizer error: {0}")]
    HFTokenizerError(String),
}

impl Tokenizer {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Tokenizer, TokenizerError> {
        let path = path.as_ref();
        let file_path = if path.is_dir() {
            let json = path.join("tokenizer.json");
            if json.exists() {
                json
            } else {
                let model = path.join("tokenizer.model");
                if model.exists() {
                    model
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("No tokenizer.json or tokenizer.model found in directory {:?}", path),
                    ).into());
                }
            }
        } else {
            path.to_path_buf()
        };
        let mut buffer = Vec::new();
        std::fs::File::open(&file_path)?.read_to_end(&mut buffer)?;

        if let Ok(model) = ModelProto::parse_from_bytes(&buffer) {
            let mut pieces = BTreeMap::new();
            for (idx, piece) in model.pieces.iter().enumerate() {
                let piece_str = match piece.piece.clone() {
                    None => continue,
                    Some(s) => s,
                };
                let piece_type = match piece.type_ {
                    None => sentence_piece::Type::NORMAL,
                    Some(v) => match v.enum_value() {
                        Err(_) => return Err(TokenizerError::UnknownPieceType(piece_str)),
                        Ok(v) => v,
                    },
                };

                let score = piece.score.unwrap_or(0.0);
                let tp = if piece_type == sentence_piece::Type::NORMAL {
                    PieceType::Normal
                } else if piece_type == sentence_piece::Type::UNKNOWN {
                    PieceType::Unknown
                } else if piece_type == sentence_piece::Type::CONTROL {
                    PieceType::Control
                } else if piece_type == sentence_piece::Type::USER_DEFINED {
                    PieceType::UserDefined
                } else if piece_type == sentence_piece::Type::BYTE {
                    PieceType::Byte
                } else if piece_type == sentence_piece::Type::UNUSED {
                    PieceType::Unused
                } else {
                    return Err(TokenizerError::UnknownPieceType(piece_str));
                };
                pieces.insert(
                    piece_str,
                    Piece {
                        _tp: tp,
                        _score: score,
                        idx,
                    },
                );
            }

            return Ok(Tokenizer {
                pieces,
                #[cfg(feature = "with_tokenizers")]
                hf: None,
                bpe_merges: None,
                byte_decoder: None,
                byte_encoder: None,
            });
        }

        #[cfg(feature = "with_tokenizers")]
        {
            let hf_result = HFTokenizer::from_bytes(buffer.clone())
                .map_err(|e| (e, "from_bytes"))
                .or_else(|(_e, _)| {
                    HFTokenizer::from_file(&file_path)
                        .map_err(|e2| (e2, "from_file"))
                });
            match hf_result {
                Ok(hf_tok) => {
                    let vocab = hf_tok.get_vocab(true);
                    let mut pieces = BTreeMap::new();
                    for (token, id) in vocab.iter() {
                        pieces.insert(
                            token.clone(),
                            Piece {
                                _tp: PieceType::Normal,
                                _score: 0.0,
                                idx: *id as usize,
                            },
                        );
                    }
                    eprintln!(
                        "DEBUG tokenizer: loaded HF tokenizer with {} vocab entries",
                        pieces.len()
                    );
                    return Ok(Tokenizer {
                        pieces,
                        hf: Some(hf_tok),
                        bpe_merges: None,
                        byte_decoder: None,
                        byte_encoder: None,
                    });
                }
                Err((e, method)) => {
                    eprintln!(
                        "DEBUG tokenizer: HFTokenizer::{} failed: {}",
                        method, e
                    );
                    // Try direct JSON fallback with BPE support
                    match Self::from_tokenizer_json(&file_path) {
                        Ok(tok) => {
                            eprintln!(
                                "DEBUG tokenizer: fallback JSON tokenizer loaded with {} vocab entries",
                                tok.pieces.len()
                            );
                            return Ok(tok);
                        }
                        Err(e2) => {
                            return Err(TokenizerError::HFTokenizerError(format!(
                                "Failed to load tokenizer (HF: {}, JSON: {})",
                                e, e2
                            )));
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "with_tokenizers"))]
        {
            match Self::from_tokenizer_json(&file_path) {
                Ok(tok) => {
                    eprintln!(
                        "DEBUG tokenizer: JSON tokenizer loaded with {} vocab entries",
                        tok.pieces.len()
                    );
                    return Ok(tok);
                }
                Err(e) => {
                    use protobuf::Error as ProtobufError;
                    return Err(TokenizerError::ProtobufError(ProtobufError::Other(
                        format!("No valid tokenizer found: {}", e),
                    )));
                }
            }
        }
    }

    pub fn id_to_str(&self, id: i32) -> &str {
        let id = id as usize;
        for (piece_str, piece_info) in self.pieces.iter() {
            if piece_info.idx == id {
                return piece_str;
            }
        }
        panic!("id out of range");
    }

    pub fn str_to_id(&self, s: &str) -> Option<TokenId> {
        for (piece_str, piece_info) in self.pieces.iter() {
            if piece_str == s {
                return Some(piece_info.idx as i32);
            }
        }
        None
    }

    pub fn tokenize_to_pieces<S: AsRef<str>>(&self, s: S) -> Vec<&str> {
        if self.pieces.is_empty() {
            return vec![];
        }
        let mut s: &str = s.as_ref();
        let mut result: Vec<&str> = Vec::new();

        while !s.is_empty() {
            let mut best_candidate: &str = "";
            let mut best_candidate_len: usize = 0;
            let mut skip_s: &str = "";
            if s.starts_with('\n') {
                if self.str_to_id("<0x0A>").is_some() {
                    best_candidate = "<0x0A>";
                    best_candidate_len = best_candidate.len();
                    skip_s = &s[1..];
                } else {
                    best_candidate = "\\n";
                }
            } else {
                for (piece_str, _piece_info) in self.pieces.iter() {
                    if s.starts_with(piece_str) && best_candidate_len < piece_str.len() {
                        best_candidate = piece_str;
                        best_candidate_len = piece_str.len();
                        skip_s = &s[piece_str.len()..];
                    }
                }
            }
            if best_candidate_len == 0 {
                s = s.get(1..).unwrap_or("");
            } else {
                result.push(best_candidate);
                s = skip_s;
            }
        }
        result
    }

    pub fn tokenize_to_ids<S: AsRef<str>>(&self, s: S) -> Vec<TokenId> {
        #[cfg(feature = "with_tokenizers")]
        {
            if let Some(ref hf_tok) = self.hf {
                let encoding = hf_tok
                    .encode(s.as_ref(), false)
                    .map_err(|e| format!("Tokenization error: {}", e))
                    .unwrap();
                return encoding.get_ids().iter().map(|id| *id as i32).collect();
            }
        }

        if let Some(ref merges) = self.bpe_merges {
            if let Some(ref encoder) = self.byte_encoder {
                return self.bpe_encode(s.as_ref(), encoder, merges);
            }
        }

        let mut s: String = format!("▁{}", s.as_ref());
        s = s.replace(' ', "▁");
        let pieces = self.tokenize_to_pieces(s);
        let mut result = Vec::new();
        result.push(1);
        for piece in pieces {
            if let Some(piece_info) = self.pieces.get(piece) {
                result.push(piece_info.idx as i32);
            }
        }
        result
    }

    pub fn decode_token(&self, id: TokenId) -> String {
        let id = id as usize;
        let token_str = self
            .pieces
            .iter()
            .find(|(_, p)| p.idx == id)
            .map(|(s, _)| s.as_str())
            .unwrap_or("");

        if let Some(ref decoder) = self.byte_decoder {
            let mut bytes = Vec::new();
            for c in token_str.chars() {
                let cp = c as u32;
                if let Some(&b) = decoder.get(&cp) {
                    bytes.push(b);
                }
            }
            String::from_utf8(bytes).unwrap_or_default()
        } else {
            token_str.replace('▁', " ").replace("<0x0A>", "\n")
        }
    }

    fn build_byte_maps() -> (HashMap<u8, u32>, HashMap<u32, u8>) {
        let mut bs: Vec<u8> = Vec::new();
        bs.extend(b'!'..=b'~');
        bs.extend(161u8..=172u8);
        bs.extend(174u8..=255u8);

        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut n = 0u32;

        for b in 0u8..=255u8 {
            if bs.contains(&b) {
                encoder.insert(b, b as u32);
                decoder.insert(b as u32, b);
            } else {
                encoder.insert(b, 256 + n);
                decoder.insert(256 + n, b);
                n += 1;
            }
        }

        (encoder, decoder)
    }

    fn bpe_encode(
        &self,
        text: &str,
        encoder: &HashMap<u8, u32>,
        merges: &[(String, String)],
    ) -> Vec<TokenId> {
        let re = Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
        )
        .unwrap();

        let mut result = Vec::new();

        for m in re.find_iter(text) {
            let word = m.as_str();
            let bytes = word.as_bytes();
            let chars: Vec<String> = bytes
                .iter()
                .map(|&b| {
                    let default_cp = b as u32;
                    let cp = encoder.get(&b).unwrap_or(&default_cp);
                    char::from_u32(*cp).unwrap().to_string()
                })
                .collect();

            let merged = Self::apply_bpe_merges(&chars, merges);

            for token in &merged {
                if let Some(piece) = self.pieces.get(token) {
                    result.push(piece.idx as i32);
                }
            }
        }

        result
    }

    fn apply_bpe_merges(chars: &[String], merges: &[(String, String)]) -> Vec<String> {
        let mut seq: Vec<String> = chars.to_vec();

        for (a, b) in merges {
            let mut i = 0;
            while i + 1 < seq.len() {
                if seq[i] == *a && seq[i + 1] == *b {
                    let merged = format!("{}{}", seq[i], seq[i + 1]);
                    seq.splice(i..=i + 1, [merged].into_iter());
                } else {
                    i += 1;
                }
            }
            if seq.len() == 1 {
                break;
            }
        }

        seq
    }

    #[cfg(feature = "with_tokenizers")]
    fn from_tokenizer_json(path: &Path) -> Result<Tokenizer, String> {
        Self::from_tokenizer_json_inner(path)
    }

    #[cfg(not(feature = "with_tokenizers"))]
    fn from_tokenizer_json(path: &Path) -> Result<Tokenizer, String> {
        Self::from_tokenizer_json_inner(path)
    }

    fn from_tokenizer_json_inner(path: &Path) -> Result<Tokenizer, String> {
        let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open: {}", e))?;
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| format!("Cannot read: {}", e))?;
        let root: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("Invalid JSON: {}", e))?;

        let model = root
            .get("model")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "Missing 'model' section".to_string())?;

        let model_type = model
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "Missing 'model.vocab' section".to_string())?;

        let mut pieces = BTreeMap::new();
        let mut max_id: usize = 0;
        for (token, id_val) in vocab_obj {
            let id = id_val.as_i64().ok_or_else(|| {
                format!("Invalid vocab id for token '{}'", token)
            })? as usize;
            if id > max_id {
                max_id = id;
            }
            pieces.insert(
                token.clone(),
                Piece {
                    _tp: PieceType::Normal,
                    _score: 0.0,
                    idx: id,
                },
            );
        }

        let added_tokens = root.get("added_tokens").and_then(|v| v.as_array());
        if let Some(tokens) = added_tokens {
            for entry in tokens {
                if let Some(obj) = entry.as_object() {
                    if let (Some(content), Some(id_val)) =
                        (obj.get("content").and_then(|v| v.as_str()), obj.get("id"))
                    {
                        if let Some(id) = id_val.as_i64() {
                            let id = id as usize;
                            if id > max_id {
                                max_id = id;
                            }
                            pieces.entry(content.to_string()).or_insert(Piece {
                                _tp: PieceType::Normal,
                                _score: 0.0,
                                idx: id,
                            });
                        }
                    }
                }
            }
        }

        let (bpe_merges, byte_encoder, byte_decoder) = if model_type == "BPE" {
            let merges_arr = model
                .get("merges")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            let pair = v.as_array()?;
                            let a = pair.get(0)?.as_str()?;
                            let b = pair.get(1)?.as_str()?;
                            Some((a.to_string(), b.to_string()))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let (enc, dec) = Self::build_byte_maps();

            (Some(merges_arr), Some(enc), Some(dec))
        } else {
            (None, None, None)
        };

        Ok(Tokenizer {
            pieces,
            #[cfg(feature = "with_tokenizers")]
            hf: None,
            bpe_merges,
            byte_decoder,
            byte_encoder,
        })
    }
}
