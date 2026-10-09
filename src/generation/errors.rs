use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("generation request failed: {0}")]
    Generation(#[from] google_ai_rs::Error),
}