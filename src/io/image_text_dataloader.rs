use crate::io::image::load_image_to_tensor;
use crate::tensor::Tensor;
use log::info;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn maybe_flip_horizontal(img: Tensor, enable: bool) -> Result<Tensor, String> {
    if !enable {
        return Ok(img);
    }
    if !rand::random::<bool>() {
        return Ok(img);
    }

    use ndarray::{s, Ix4};

    let arr = img.lock().storage.to_f32_array();
    let shape_dbg = arr.shape().to_vec();
    let arr4 = arr
        .into_dimensionality::<Ix4>()
        .map_err(|_| format!("Expected image tensor to have 4 dims [1,C,H,W], got shape {:?}", shape_dbg))?;
    if arr4.shape()[0] != 1 {
        return Err(format!(
            "Expected image tensor batch dim to be 1 (shape [1,C,H,W]), got shape {:?}",
            arr4.shape()
        ));
    }
    // Reverse the last axis (W) to flip horizontally.
    let flipped = arr4.slice(s![.., .., .., ..;-1]).to_owned().into_dyn();
    Ok(Tensor::new(flipped, false))
}

/// A simple image-text data loader that reads a manifest file where each line is:
/// <image_path>\t<caption>
/// It emits batches of (image_tensor, token_ids) pairs. Tokenization requires the
/// `with_tokenizers` feature; otherwise tokenization will return an error during construction.
pub struct ImageTextDataLoader {
    entries: Vec<(PathBuf, String)>,
    pub image_size: (u32, u32),
    pub batch_size: usize,
    pub shuffle: bool,
    pub augment: bool,
    pub parallel: bool,
}

impl ImageTextDataLoader {
    /// Create a new loader from a manifest file. Each line of `manifest_path` should contain
    /// an image path and a caption separated by a tab character.
    pub fn new_from_manifest<P: AsRef<Path>>(manifest_path: P, image_size: (u32, u32), batch_size: usize, shuffle: bool, augment: bool, parallel: bool) -> Result<Self, String> {
        let p = manifest_path.as_ref();
        if !p.exists() {
            return Err(format!("Manifest file not found: {}", p.display()));
        }
        let f = File::open(p).map_err(|e| format!("Failed to open manifest {}: {}", p.display(), e))?;
        let reader = BufReader::new(f);
        let mut entries = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let l = line.map_err(|e| format!("Failed to read manifest {} line {}: {}", p.display(), i + 1, e))?;
            let l = l.trim();
            if l.is_empty() { continue; }
            let parts: Vec<&str> = l.splitn(2, '\t').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid manifest line {}: '{}'. Expected '<image_path>\t<caption>'", i + 1, l));
            }
            let img_path = PathBuf::from(parts[0]);
            if !img_path.exists() {
                return Err(format!("Image file not found for manifest line {}: {}", i + 1, img_path.display()));
            }
            entries.push((img_path, parts[1].to_string()));
        }
        if entries.is_empty() {
            return Err(format!("No entries found in manifest {}", p.display()));
        }
        info!("ImageTextDataLoader created: manifest={} entries={} image_size={:?} batch_size={} shuffle={} augment={}", p.display(), entries.len(), image_size, batch_size, shuffle, augment);
        Ok(ImageTextDataLoader { entries, image_size, batch_size, shuffle, augment, parallel })
    }

    pub fn num_batches(&self) -> usize {
        self.entries.len().div_ceil(self.batch_size)
    }

    /// Load a batch by index. Returns a vector of image tensors and a vector of captions.
    /// Tokenization is left to the caller to allow flexibility.
    pub fn load_batch(&self, batch_idx: usize) -> Result<(Vec<Tensor>, Vec<String>), String> {
        let start = batch_idx * self.batch_size;
        let mut images = Vec::new();
        let mut captions = Vec::new();
        if start >= self.entries.len() {
            return Err(format!("Batch index out of range: {} >= {}", start, self.entries.len()));
        }
        let end = std::cmp::min(start + self.batch_size, self.entries.len());
        if self.parallel {
            // Use rayon's parallel iter if enabled
            #[cfg(feature = "parallel_io")]
            {
                use rayon::prelude::*;
                let slice = &self.entries[start..end];
                let results: Vec<Result<(Tensor, String), String>> = slice.par_iter().map(|(path, caption)| {
                    let img = load_image_to_tensor(path.to_str().ok_or_else(|| format!("Invalid path: {}", path.display()))?, Some(self.image_size))
                        .map_err(|e| format!("Failed to load image {}: {}", path.display(), e))?;
                    let img = maybe_flip_horizontal(img, self.augment)?;
                    Ok((img, caption.clone()))
                }).collect();
                for r in results.into_iter() {
                    match r {
                        Ok((img, cap)) => {
                            images.push(img);
                            captions.push(cap);
                        }
                        Err(e) => return Err(e)
                    }
                }
            }
            #[cfg(not(feature = "parallel_io"))]
            {
                for i in start..end {
                    let (ref path, ref caption) = self.entries[i];
                    let img = load_image_to_tensor(path.to_str().ok_or_else(|| format!("Invalid path: {}", path.display()))?, Some(self.image_size))
                        .map_err(|e| format!("Failed to load image {}: {}", path.display(), e))?;
                    let img = maybe_flip_horizontal(img, self.augment)?;
                    images.push(img);
                    captions.push(caption.clone());
                }
            }
        } else {
            for i in start..end {
                let (ref path, ref caption) = self.entries[i];
                let img = load_image_to_tensor(path.to_str().ok_or_else(|| format!("Invalid path: {}", path.display()))?, Some(self.image_size))
                    .map_err(|e| format!("Failed to load image {}: {}", path.display(), e))?;
                let img = maybe_flip_horizontal(img, self.augment)?;
                images.push(img);
                captions.push(caption.clone());
            }
        }
        Ok((images, captions))
    }

    #[cfg(feature = "with_tokenizers")]
    /// Load a batch and return both images and tokenized captions using provided HF Tokenizer.
    pub fn load_batch_tokenized(&self, batch_idx: usize, tokenizer: &tokenizers::Tokenizer) -> Result<(Vec<Tensor>, Vec<Vec<u32>>), String> {
        let (images, captions) = self.load_batch(batch_idx)?;
        let mut tokenized = Vec::with_capacity(captions.len());
        for c in captions.iter() {
            match crate::io::tokenizers::encode_text(tokenizer, c) {
                Ok(ids) => tokenized.push(ids),
                Err(e) => return Err(format!("Tokenization error: {}", e)),
            }
        }
        Ok((images, tokenized))
    }

    /// Shuffle the entries in place. Intended to be called at the start of an epoch.
    pub fn shuffle_in_place(&mut self) {
        let mut rng = rand::rng();
        self.entries.shuffle(&mut rng);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "vision")]
    fn test_image_text_dataloader_basic() {
        // Create temporary directory with one image and manifest
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("test.png");
        // Create a tiny 2x2 RGB png
        let mut imgbuf = image::RgbImage::new(2, 2);
        imgbuf.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        imgbuf.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        imgbuf.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        imgbuf.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        imgbuf.save(&img_path).expect("save image");
        let manifest_path = dir.path().join("manifest.txt");
        let mut f = std::fs::File::create(&manifest_path).expect("create manifest");
        writeln!(f, "{}\t{}", img_path.to_str().unwrap(), "a caption").expect("write manifest");

        let loader = ImageTextDataLoader::new_from_manifest(&manifest_path, (2, 2), 1, false, false, false).expect("create loader");
        assert_eq!(loader.num_batches(), 1);
        let (images, captions) = loader.load_batch(0).expect("load batch");
        assert_eq!(images.len(), 1);
        assert_eq!(captions.len(), 1);
        // Validate image load: shape [1,3,H,W]
        let t = &images[0];
        let shape = t.lock().storage.shape();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 2);
        assert_eq!(shape[3], 2);
    }
}
