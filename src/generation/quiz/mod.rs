pub mod solutions;
pub mod errors;
mod prompts;

use crate::generation::image::{is_image_url, load_image_part, load_image_url_part};
use errors::{QuizError, QuizFromImageError, QuizValidationError};
use google_ai_rs::{AsSchema, Client};
use serde::Deserialize;
use std::path::Path;

/// A single question in a quiz.
///
/// Labels are stored separately: `question` and `options` hold bare text
/// without printed numbers or letters (the renderer adds `1)`, `a)`, ...).
/// Emphasized spans (bold/underline in the source) are kept inline as
/// `**...**` markdown.
#[allow(dead_code)]
#[derive(AsSchema, Deserialize, Debug)]
pub enum QuizQuestion {
    #[schema(description = "A question with pre-defined answers that you have to choose from")]
    MultipleChoice {
        #[schema(description = "The given question, without its printed number")]
        question: String,
        #[schema(description = "Options as bare text, without printed letters or bullets")]
        options: Vec<String>,
    },
    #[schema(description = "A plain true/false question with no emphasized span to correct")]
    TrueFalse {
        #[schema(description = "The given question, without its printed number")]
        question: String,
    },
    #[schema(description = "A true/false question where one span is emphasized (bold/underlined) and the instructions ask to correct that span when the statement is false; keep the span wrapped in **...**")]
    ModifiedTrueFalse {
        #[schema(description = "The statement, without its printed number, with the emphasized span wrapped in **...**")]
        question: String,
    },
    #[schema(description = "Any other kind of question, e.g. fill in the blank, problem solving, essay")]
    OtherQuestion {
        #[schema(description = "The given question, without its printed number; wrap emphasized spans in **...**")]
        question: String,
    },
}

/// A single _section_ in a quiz.
///
/// A section breaks up long quizzes into different parts. Typically, each
/// section holds questions of the same variant. Short quizzes may hold only
/// one section that spans the entire scope of the quiz.
#[allow(dead_code)]
#[derive(AsSchema, Deserialize, Debug)]
pub struct QuizSection {
    #[schema(description = "Short instructions shown above this section, e.g. how to answer")]
    pub instructions: Option<String>,
    #[schema(description = "Questions in this section; ideally all of the same kind")]
    pub questions: Vec<QuizQuestion>,
}

/// An entire quiz.
#[allow(dead_code)]
#[derive(Debug)]
pub struct Quiz(pub Vec<QuizSection>);

/// What the model returns.
#[allow(dead_code)]
#[derive(AsSchema, Deserialize, Debug)]
struct QuizResponse {
    #[schema(description = "Sections of the quiz, in the order they appear in the source")]
    pub sections: Vec<QuizSection>,
}

#[allow(dead_code)]
impl QuizResponse {
    pub async fn from_image(client: &Client, model_name: &str, image_path: impl AsRef<Path>) -> Result<Self, QuizError> {
        let image = load_image_part(image_path.as_ref())?;

        let model = client.typed_model::<QuizResponse>(model_name);
        let response = model.generate_content((prompts::QUIZ_IMAGE_TO_ENCODED, image)).await?;
        Ok(response)
    }

    pub async fn from_image_url(client: &Client, model_name: &str, image_url: &str) -> Result<Self, QuizError> {
        let image = load_image_url_part(image_url)?;

        let model = client.typed_model::<QuizResponse>(model_name);
        let response = model.generate_content((prompts::QUIZ_IMAGE_TO_ENCODED, image)).await?;
        Ok(response)
    }

    /// Accepts either a local path or an `http(s)://` image URL.
    pub async fn from_image_or_url(
        client: &Client,
        model_name: &str,
        image: &str,
    ) -> Result<Self, QuizError> {
        if is_image_url(image) {
            Self::from_image_url(client, model_name, image).await
        } else {
            Self::from_image(client, model_name, Path::new(image)).await
        }
    }
}

/// Transcription attempts: sampling is stochastic, so a reply that fails
/// validation is worth re-requesting before giving up.
const TRANSCRIBE_ATTEMPTS: u32 = 3;

#[allow(dead_code)]
impl Quiz {
    pub async fn from_image(
        client: &Client,
        model_name: &str,
        image_path: impl AsRef<Path>,
    ) -> Result<Self, QuizFromImageError> {
        let mut last_error = None;
        for attempt in 1..=TRANSCRIBE_ATTEMPTS {
            let quiz_response =
                QuizResponse::from_image(client, model_name, image_path.as_ref()).await?;
            match Self::try_from(quiz_response) {
                Ok(quiz) => return Ok(quiz),
                Err(err) => {
                    tracing::warn!(
                        "transcription failed validation (attempt {attempt}/{TRANSCRIBE_ATTEMPTS}): {err}"
                    );
                    last_error = Some(err);
                }
            }
        }
        Err(last_error
            .expect("attempts non-empty")
            .into())
    }

    pub async fn from_image_url(
        client: &Client,
        model_name: &str,
        image_url: &str,
    ) -> Result<Self, QuizFromImageError> {
        let mut last_error = None;
        for attempt in 1..=TRANSCRIBE_ATTEMPTS {
            let quiz_response =
                QuizResponse::from_image_url(client, model_name, image_url).await?;
            match Self::try_from(quiz_response) {
                Ok(quiz) => return Ok(quiz),
                Err(err) => {
                    tracing::warn!(
                        "transcription failed validation (attempt {attempt}/{TRANSCRIBE_ATTEMPTS}): {err}"
                    );
                    last_error = Some(err);
                }
            }
        }
        Err(last_error
            .expect("attempts non-empty")
            .into())
    }

    /// Accepts either a local path or an `http(s)://` image URL.
    pub async fn from_image_or_url(
        client: &Client,
        model_name: &str,
        image: &str,
    ) -> Result<Self, QuizFromImageError> {
        if is_image_url(image) {
            Self::from_image_url(client, model_name, image).await
        } else {
            Self::from_image(client, model_name, Path::new(image)).await
        }
    }
}

impl TryFrom<QuizResponse> for Quiz {
    type Error = QuizValidationError;

    fn try_from(mut r: QuizResponse) -> Result<Self, Self::Error> {
        if r.sections.is_empty() {
            return Err(QuizValidationError::NoSections);
        }
        for (section, s) in r.sections.iter_mut().enumerate() {
            if s.questions.is_empty() {
                return Err(QuizValidationError::EmptySection { section });
            }
            for (question, q) in s.questions.iter_mut().enumerate() {
                match q {
                    QuizQuestion::MultipleChoice {
                        question: text,
                        options,
                    } => {
                        *text = strip_question_label(text).to_string();
                        for option in options.iter_mut() {
                            *option = strip_option_label(option).to_string();
                        }
                        if text.trim().is_empty() {
                            return Err(QuizValidationError::EmptyQuestion { section, question });
                        }
                        if options.len() < 2 {
                            return Err(QuizValidationError::TooFewOptions {
                                section,
                                question,
                                got: options.len(),
                            });
                        }
                    }
                    QuizQuestion::TrueFalse { question: text }
                    | QuizQuestion::ModifiedTrueFalse { question: text }
                    | QuizQuestion::OtherQuestion { question: text } => {
                        *text = strip_question_label(text).to_string();
                        if text.trim().is_empty() {
                            return Err(QuizValidationError::EmptyQuestion { section, question });
                        }
                    }
                }
            }
        }
        Ok(Quiz(r.sections))
    }
}

/// Strip a leading question number (`1.`, `2)`) the model may have copied
/// despite being told not to. Only strips ASCII digits followed by `.` or
/// `)` plus whitespace, so content like `3x + 5 = 20` is untouched.
fn strip_question_label(s: &str) -> &str {
    let t = s.trim_start();
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits > 0
        && let Some(rest) = t.get(digits..)
    {
        let mut chars = rest.chars();
        match chars.next() {
            Some('.') | Some(')') => {
                let after = chars.as_str();
                if after.starts_with(char::is_whitespace) {
                    return after.trim_start();
                }
            }
            _ => {}
        }
    }
    t
}

/// Strip a leading option marker (`- A.`, `b)`, `C.`) the model may have
/// copied. Only strips a single letter plus `.`/`)` plus whitespace, so
/// numeric options like `48` are untouched.
fn strip_option_label(s: &str) -> &str {
    let mut t = s.trim_start();
    loop {
        let stripped = t
            .strip_prefix("- ")
            .or_else(|| t.strip_prefix("* "))
            .or_else(|| t.strip_prefix("• "));
        match stripped {
            Some(rest) => t = rest.trim_start(),
            None => break,
        }
    }
    let mut chars = t.chars();
    if let Some(c) = chars.next()
        && c.is_ascii_alphabetic()
    {
        let after = chars.as_str();
        let mut rest = after.chars();
        match rest.next() {
            Some('.') | Some(')') => {
                let tail = rest.as_str();
                if !tail.is_empty() && tail.starts_with(char::is_whitespace) {
                    return tail.trim_start();
                }
            }
            _ => {}
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::image::{
        ImageError, image_url_part_with_mime, is_image_url, load_image_url_part,
    };
    use std::path::PathBuf;

    /// Default model for the live test. Override with `GEMINI_MODEL`.
    /// (`gemini-2.5-flash-lite` was retired; 3.1-flash-lite is the cheap one
    /// that still works.)
    const DEFAULT_TEST_MODEL: &str = "gemini-3.1-flash-lite";

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("math_quiz.png")
    }

    #[test]
    fn fixture_image_exists() {
        let path = fixture_path();
        assert!(
            path.is_file(),
            "fixture image missing at {}",
            path.display()
        );
        let meta = std::fs::metadata(&path).expect("fixture metadata readable");
        assert!(meta.len() > 0, "fixture image is empty");
    }

    #[test]
    fn load_image_part_accepts_fixture() {
        let path = fixture_path();
        let part = load_image_part(&path).expect("fixture should load as image Part");
        // Part doesn't expose its payload; constructing without error is the assertion.
        let _ = part;
    }

    #[test]
    fn load_image_part_rejects_unsupported_extension() {
        let err = load_image_part(Path::new("quiz.txt")).unwrap_err();
        assert!(
            matches!(err, ImageError::UnsupportedImage(_)),
            "expected UnsupportedImage, got {err:?}"
        );
    }

    #[test]
    fn load_image_part_reports_missing_file() {
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("does-not-exist.png");
        let err = load_image_part(&missing).unwrap_err();
        assert!(
            matches!(err, ImageError::ReadImage { .. }),
            "expected ReadImage, got {err:?}"
        );
    }

    #[test]
    fn image_url_detection() {
        assert!(is_image_url("https://example.com/quiz.png"));
        assert!(is_image_url("http://example.com/quiz.jpg"));
        assert!(is_image_url("HTTPS://example.com/quiz.webp"));
        assert!(!is_image_url("tests/fixtures/math_quiz.png"));
        assert!(!is_image_url("quiz.png"));
        assert!(!is_image_url("ftp://example.com/quiz.png"));
    }

    #[test]
    fn load_image_url_part_accepts_http_urls() {
        for url in [
            "https://example.com/quiz.png",
            "https://example.com/photos/quiz.JPG?size=large",
            "https://example.com/a/b/quiz.webp#frag",
            "http://example.com/quiz.heic",
            "https://example.com/quiz.heif",
        ] {
            let part = load_image_url_part(url)
                .unwrap_or_else(|e| panic!("{url} should load as image Part: {e:?}"));
            let _ = part;
        }
    }

    #[test]
    fn load_image_url_part_rejects_non_urls() {
        let err = load_image_url_part("tests/fixtures/math_quiz.png").unwrap_err();
        assert!(
            matches!(err, ImageError::InvalidUrl(_)),
            "expected InvalidUrl, got {err:?}"
        );
    }

    #[test]
    fn load_image_url_part_rejects_unsupported_extension() {
        let err = load_image_url_part("https://example.com/quiz.txt").unwrap_err();
        assert!(
            matches!(err, ImageError::UnsupportedImage(_)),
            "expected UnsupportedImage, got {err:?}"
        );
        let err = load_image_url_part("https://example.com/quiz").unwrap_err();
        assert!(
            matches!(err, ImageError::UnsupportedImage(_)),
            "expected UnsupportedImage, got {err:?}"
        );
    }

    #[test]
    fn image_url_part_with_mime_accepts_extensionless_uris() {
        let part = image_url_part_with_mime(
            "https://generativelanguage.googleapis.com/v1beta/files/abc123",
            "image/png",
        )
        .expect("Files API URI should build a Part with explicit mime");
        let _ = part;
        let err = image_url_part_with_mime("notaurl", "image/png").unwrap_err();
        assert!(
            matches!(err, ImageError::InvalidUrl(_)),
            "expected InvalidUrl, got {err:?}"
        );
    }

    #[test]
    fn validation_rejects_empty_quiz() {
        let err = Quiz::try_from(QuizResponse { sections: vec![] }).unwrap_err();
        assert!(matches!(
            err,
            QuizValidationError::NoSections
        ));
    }

    #[test]
    fn strip_question_label_removes_numbers() {
        assert_eq!(strip_question_label("1. What is 2 + 2?"), "What is 2 + 2?");
        assert_eq!(strip_question_label("12) Solve for x"), "Solve for x");
        // Content starting with math is untouched.
        assert_eq!(strip_question_label("3x + 5 = 20"), "3x + 5 = 20");
        assert_eq!(strip_question_label("2³ = ___"), "2³ = ___");
        assert_eq!(strip_question_label("  6. Padded"), "Padded");
    }

    #[test]
    fn strip_option_label_removes_letters() {
        assert_eq!(strip_option_label("- A. 48"), "48");
        assert_eq!(strip_option_label("b) x = 5"), "x = 5");
        assert_eq!(strip_option_label("C. 36 cm²"), "36 cm²");
        // Numeric options are untouched.
        assert_eq!(strip_option_label("48"), "48");
        assert_eq!(strip_option_label("  - D. 80"), "80");
    }

    #[test]
    fn try_from_stores_bare_labels() {
        let quiz = Quiz::try_from(QuizResponse {
            sections: vec![QuizSection {
                instructions: None,
                questions: vec![QuizQuestion::MultipleChoice {
                    question: "1. What is 2 + 2?".to_string(),
                    options: vec!["- A. 3".to_string(), "b) 4".to_string()],
                }],
            }],
        })
        .expect("labeled transcript should normalize");
        match &quiz.0[0].questions[0] {
            QuizQuestion::MultipleChoice { question, options } => {
                assert_eq!(question, "What is 2 + 2?");
                assert_eq!(options, &["3".to_string(), "4".to_string()]);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[tokio::test]
    async fn from_image_reports_missing_file_without_network() {
        let client = Client::new("dummy-key-for-offline-test").await.unwrap();
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("does-not-exist.png");
        let err = Quiz::from_image(&client, DEFAULT_TEST_MODEL, &missing)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                QuizFromImageError::Response(QuizError::Image(ImageError::ReadImage { .. }))
            ),
            "expected ReadImage error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn from_image_url_rejects_bad_url_without_network() {
        let client = Client::new("dummy-key-for-offline-test").await.unwrap();
        let err = Quiz::from_image_url(&client, DEFAULT_TEST_MODEL, "https://example.com/quiz.txt")
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                QuizFromImageError::Response(QuizError::Image(ImageError::UnsupportedImage(_)))
            ),
            "expected UnsupportedImage error, got {err:?}"
        );
        let err = Quiz::from_image_or_url(&client, DEFAULT_TEST_MODEL, "https://example.com/quiz.txt")
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                QuizFromImageError::Response(QuizError::Image(ImageError::UnsupportedImage(_)))
            ),
            "expected UnsupportedImage error via from_image_or_url, got {err:?}"
        );
    }

    /// Live test: sends `tests/fixtures/math_quiz.png` (rendered from
    /// `tests/fixtures/quiz.md` via `render_quiz.py`) to Gemini and checks
    /// the returned `Quiz` matches the fixture's visible structure
    /// (Parts I-V: 5x multiple choice, 5x true/false-modified, 5x fill in the
    /// blank, 2x problem solving, 1x essay, plus the answer key).
    ///
    /// Requires `GEMINI_API_KEY`. Skips (passes) when unset so `cargo test`
    /// stays green offline. Override the model with `GEMINI_MODEL`.
    #[tokio::test]
    async fn quiz_from_fixture_image() {
        let api_key = match std::env::var("GEMINI_API_KEY") {
            Ok(k) if !k.trim().is_empty() => k,
            _ => {
                eprintln!("skipping live Gemini test: GEMINI_API_KEY not set");
                return;
            }
        };
        let path = fixture_path();
        let client = Client::new(api_key).await.unwrap();

        // Model candidates: an explicit GEMINI_MODEL wins; otherwise try the
        // flash-lite default (incl. the suggested 3.5 flash-lite) and fall back
        // to the API-recommended 3.8 flash.
        let candidates: Vec<String> = match std::env::var("GEMINI_MODEL") {
            Ok(m) if !m.trim().is_empty() => vec![m],
            _ => [
                DEFAULT_TEST_MODEL,
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        };

        let mut last_err = None;
        let mut quiz = None;
        for model in &candidates {
            match Quiz::from_image(&client, model, &path).await {
                Ok(q) => {
                    eprintln!("live test succeeded with model {model}");
                    quiz = Some(q);
                    break;
                }
                Err(e) => {
                    eprintln!("model {model} failed: {e:?}");
                    last_err = Some((model.clone(), e));
                }
            }
        }
        let quiz = quiz.unwrap_or_else(|| {
            let (model, err) = last_err.expect("candidates non-empty");
            panic!("Quiz::from_image failed on {model} (tried {candidates:?}): {err:?}")
        });

        // The fixture mirrors quiz.md: five quiz parts (5 + 5 + 5 + 2 + 1
        // questions) plus an answer key the model may fold in as extra
        // sections/questions, so assert lower bounds, not exact counts.
        let total: usize = quiz.0.iter().map(|s| s.questions.len()).sum();
        assert!(
            quiz.0.len() >= 5,
            "expected >= 5 sections, got {:#?}",
            quiz
        );
        assert!(
            total >= 18,
            "expected >= 18 questions, got {total}: {quiz:#?}"
        );
        assert_eq!(quiz.0[0].questions.len(), 5, "section 0: {:#?}", quiz);

        for (i, q) in quiz.0[0].questions.iter().enumerate() {
            match q {
                QuizQuestion::MultipleChoice { question, options } => {
                    assert!(!question.trim().is_empty(), "section 0 q{i} empty text");
                    assert!(
                        options.len() >= 4,
                        "section 0 q{i} should have >= 4 options, got {options:?}"
                    );
                }
                other => panic!("section 0 q{i} should be MultipleChoice, got {other:?}"),
            }
        }

        // Part II of the fixture is modified true/false: the corrected span
        // is emphasized, so each question must carry **...** markdown.
        assert_eq!(quiz.0[1].questions.len(), 5, "section 1: {:#?}", quiz);
        for (i, q) in quiz.0[1].questions.iter().enumerate() {
            match q {
                QuizQuestion::ModifiedTrueFalse { question } => {
                    assert!(
                        question.contains("**"),
                        "section 1 q{i} should keep the emphasized span in **...**: {question:?}"
                    );
                }
                other => panic!("section 1 q{i} should be ModifiedTrueFalse, got {other:?}"),
            }
        }

        // Spot-check content from across the doc, not invented text.
        let all_text: String = quiz
            .0
            .iter()
            .flat_map(|s| s.questions.iter())
            .map(|q| match q {
                QuizQuestion::MultipleChoice { question, .. }
                | QuizQuestion::TrueFalse { question }
                | QuizQuestion::ModifiedTrueFalse { question }
                | QuizQuestion::OtherQuestion { question } => question.as_str(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        for needle in ["3x", "six sides", "800", "dividing by zero", "square root"] {
            assert!(
                all_text.contains(needle),
                "expected {needle:?} from the fixture in questions, got:\n{all_text}"
            );
        }
    }
}
