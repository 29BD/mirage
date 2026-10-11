pub mod generation;
pub mod output;

// Re-export external types appearing in the public API so downstream crates
// don't need to depend on `google-ai-rs` directly (and can't hit version
// mismatches on `Client` / `Part`).
pub use google_ai_rs::Client;
pub use google_ai_rs::Error as GeminiError;
pub use google_ai_rs::Part;

// Common entry points for library users.
pub use generation::errors::LlmError;
pub use generation::image::{
    ImageError, image_url_part_with_mime, is_image_url, load_image_part, load_image_url_part,
};
pub use generation::queue::Throttle;
pub use generation::quiz::errors::{QuizError, QuizFromImageError, QuizValidationError};
pub use generation::quiz::progress::{MpscSink, NullSink, ProgressSink, QuizProgress, WaitReason};
pub use generation::quiz::solutions::{Answer, Answerable, LlmContext};
pub use generation::quiz::{Quiz, QuizQuestion, QuizSection};
pub use output::render_quiz;
