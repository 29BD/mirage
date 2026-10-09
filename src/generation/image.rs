use std::io;
use std::path::{Path, PathBuf};
use google_ai_rs::Part;
use thiserror::Error;

#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum ImageError {
    #[error("Unsupported image extension: {0:?}")]
    UnsupportedImage(Option<String>),
    #[error("Failed to read image {}: {source}", path.display())]
    ReadImage {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

fn mime_for(path: &Path) -> Result<&'static str, ImageError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("png") => Ok("image/png"),
        Some("jpg") | Some("jpeg") => Ok("image/jpeg"),
        Some("webp") => Ok("image/webp"),
        Some("heic") => Ok("image/heic"),
        Some("heif") => Ok("image/heif"),
        _ => Err(ImageError::UnsupportedImage(ext)),
    }
}

/// Reads an image file into a `Part` ready to send to the model.
#[allow(dead_code)]
pub fn load_image_part(path: &Path) -> Result<Part, ImageError> {
    let mime = mime_for(path)?;
    let bytes = std::fs::read(path).map_err(|source| ImageError::ReadImage {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Part::blob(mime, bytes))
}
