#[allow(dead_code)]
pub const QUIZ_IMAGE_TO_ENCODED: &str = "\
The attached image is a quiz, worksheet, or page of study material. \
Extract every question from it into the requested structure, preserving the original order and wording. \
Group questions into sections the way the source does; if it has no visible sections, use a single section. \
Copy any instructions printed for a section into that section's `instructions`. \
Classify each question as multiple choice (only when answer options are printed), true/false, or other. \
Do not invent questions or answer options that are not in the image, and do not answer the questions.";