use std::io;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum QuizValidationError {
    #[error("Model returned no sections")]
    NoSections,
    #[error("Section {section} has no questions")]
    EmptySection { section: usize },
    #[error("Section {section} question {question}: empty question text")]
    EmptyQuestion { section: usize, question: usize },
    #[error(
        "Section {section} question {question}: multiple choice needs at least 2 options, got {got}"
    )]
    TooFewOptions {
        section: usize,
        question: usize,
        got: usize,
    },
}
 
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum QuizError {
    #[error("Unsupported image extension: {0:?}")]
    UnsupportedImage(Option<String>),
    #[error("Failed to read image {}: {source}", path.display())]
    ReadImage {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Gemini request failed: {0}")]
    Api(#[from] google_ai_rs::Error),
    #[error("Model returned an invalid quiz: {0}")]
    Invalid(#[from] QuizValidationError),
}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum QuizFromImageError {
    #[error(transparent)]
    Response(#[from] QuizError),
    #[error(transparent)]
    Validation(#[from] QuizValidationError),
}