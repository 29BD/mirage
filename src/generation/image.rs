use std::path::Path;
use google_ai_rs::Part;
use super::errors::QuizError;

fn mime_for(path: &Path) -> Result<&'static str, QuizError> {
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
        _ => Err(QuizError::UnsupportedImage(ext)),
    }
}

/// Reads an image file into a `Part` ready to send to the model.
pub fn load_image_part(path: &Path) -> Result<Part, QuizError> {
    let mime = mime_for(path)?;
    let bytes = std::fs::read(path).map_err(|source| QuizError::ReadImage {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Part::blob(mime, bytes))
}