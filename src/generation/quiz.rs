use google_ai_rs::{AsSchema, AsSchemaWithSerde};
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
pub struct Quiz(Vec<QuizSection>);