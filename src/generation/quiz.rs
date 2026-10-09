use crate::generation::errors::QuizFromImageError;

use super::image::load_image_part;
use std::path::Path;
use super::errors::{QuizError, QuizValidationError};
use google_ai_rs::{AsSchema, Client};
use serde::Deserialize;

/// A single question in a quiz
#[allow(dead_code)]
#[derive(AsSchema, Deserialize, Debug)]
pub enum QuizQuestion {
    #[schema(description = "A question with pre-defined answers that you have to choose from")]
    MultipleChoice {
        #[schema(description = "The given question")]
        question: String,
        #[schema(description = "Options in a list")]
        options: Vec<String>,
    },
    #[schema(description = "A true or false question")]
    TrueFalse {
        #[schema(description = "The given question")]
        question: String,
    },
    #[schema(description = "Any other kind of question. Be it true or false modified, essay questions...")]
    OtherQuestion {
        #[schema(description = "The given question")]
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
        let response = model.generate_content((super::prompts::QUIZ_IMAGE_TO_ENCODED, image)).await?;
        Ok(response)
    }
}

#[allow(dead_code)]
impl Quiz {
    pub async fn from_image(client: &Client, model_name: &str, image_path: impl AsRef<Path>) -> Result<Self, QuizFromImageError> {
        let quiz_response = QuizResponse::from_image(client, model_name, image_path).await?;
        Ok(Self::try_from(quiz_response)?)
    }
}

impl TryFrom<QuizResponse> for Quiz {
    type Error = QuizValidationError;
 
    fn try_from(r: QuizResponse) -> Result<Self, Self::Error> {
        if r.sections.is_empty() {
            return Err(QuizValidationError::NoSections);
        }
        for (section, s) in r.sections.iter().enumerate() {
            if s.questions.is_empty() {
                return Err(QuizValidationError::EmptySection { section });
            }
            for (question, q) in s.questions.iter().enumerate() {
                match q {
                    QuizQuestion::MultipleChoice {
                        question: text,
                        options,
                    } => {
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
                    | QuizQuestion::OtherQuestion { question: text } => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Default model for the live test. Override with `GEMINI_MODEL`.
    /// (`gemini-2.0-flash` was retired; the API now suggests the 3.x family.)
    const DEFAULT_TEST_MODEL: &str = "gemini-2.5-flash-lite";

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
            matches!(err, QuizError::UnsupportedImage(_)),
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
            matches!(err, QuizError::ReadImage { .. }),
            "expected ReadImage, got {err:?}"
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
                QuizFromImageError::Response(QuizError::ReadImage { .. })
            ),
            "expected ReadImage error, got {err:?}"
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

        // Spot-check content from across the doc, not invented text.
        let all_text: String = quiz
            .0
            .iter()
            .flat_map(|s| s.questions.iter())
            .map(|q| match q {
                QuizQuestion::MultipleChoice { question, .. }
                | QuizQuestion::TrueFalse { question }
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
