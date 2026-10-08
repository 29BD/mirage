use std::{error::Error, path::Path};
use super::errors::{QuizError, QuizValidationError};
use google_ai_rs::{content::TryIntoContents, AsSchema, Client, Content, Part};
use serde::Deserialize;

/// A single question in a quiz
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
#[derive(AsSchema, Deserialize, Debug)]
pub struct QuizSection {
    #[schema(description = "Short instructions shown above this section, e.g. how to answer")]
    pub instructions: Option<String>,
    #[schema(description = "Questions in this section; ideally all of the same kind")]
    pub questions: Vec<QuizQuestion>,
}

/// An entire quiz.
#[derive(Debug)]
pub struct Quiz(pub Vec<QuizSection>);

/// What the model returns.
#[derive(AsSchema, Deserialize, Debug)]
struct QuizResponse {
    #[schema(description = "Sections of the quiz, in the order they appear in the source")]
    pub sections: Vec<QuizSection>,
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