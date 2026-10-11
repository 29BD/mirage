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
    #[error("Invalid image URL: {0:?}")]
    InvalidUrl(String),
}

/// True when `s` looks like a remote image reference (`http://` or `https://`).
pub fn is_image_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn mime_for_url(url: &str) -> Result<&'static str, ImageError> {
    if !is_image_url(url) {
        return Err(ImageError::InvalidUrl(url.to_string()));
    }
    // Strip query/fragment, then take the extension of the final path segment.
    let without_query = url.split(['?', '#']).next().unwrap_or(url);
    let last_segment = without_query.rsplit('/').next().unwrap_or(without_query);
    let ext = last_segment
        .rsplit('.')
        .next()
        .filter(|_| last_segment.contains('.'))
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

/// Builds an image [`Part`] from a publicly accessible HTTPS (or HTTP) URL.
///
/// The URL is passed through as `file_data`: Gemini fetches the bytes
/// server-side, so no download is needed client-side. MIME type is inferred
/// from the URL path extension, mirroring [`load_image_part`].
pub fn load_image_url_part(url: &str) -> Result<Part, ImageError> {
    let mime = mime_for_url(url)?;
    Ok(Part::file_data(mime, url))
}

/// Builds an image [`Part`] from a URL or file reference with an explicit
/// MIME type, for URIs without a file extension (e.g. Gemini Files API
/// references like `https://generativelanguage.googleapis.com/v1beta/files/abc123`).
pub fn image_url_part_with_mime(url: &str, mime: &str) -> Result<Part, ImageError> {
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("gs://") {
        Ok(Part::file_data(mime, url))
    } else {
        Err(ImageError::InvalidUrl(url.to_string()))
    }
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
