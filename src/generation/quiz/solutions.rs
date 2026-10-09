use super::{Quiz, QuizQuestion, QuizSection};
use super::super::errors::LlmError;
use super::prompts::ANSWER_QUESTION;
use google_ai_rs::{Client, GenerativeModel};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Answer(pub String);

impl Answer {
    pub fn text(&self) -> &str {
        &self.0
    }

    #[allow(dead_code)]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for Answer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.trim())
    }
}

pub struct LlmContext {
    pub client: Client,
    pub model: String,
}

pub trait Answerable {
    type Error;
    type Output;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error>;
}

async fn llm(
    client: &Client,
    model: &str,
    prompt: &str,
    system: &str,
) -> Result<String, LlmError> {
    let mut gen_model: GenerativeModel = client.generative_model(model);
    if !system.is_empty() {
        gen_model = gen_model.with_system_instruction(system);
    }

    let response = gen_model.generate_content(prompt).await?;
    Ok(response.text().to_string())
}

fn prompt_for(question: &QuizQuestion) -> String {
    match question {
        QuizQuestion::MultipleChoice { question, options } => {
            format!("{question}\nOptions: {}", options.join(", "))
        }
        QuizQuestion::TrueFalse { question } => {
            format!("{question}\nOptions: True, False")
        }
        QuizQuestion::OtherQuestion { question } => question.clone(),
    }
}

impl Answerable for QuizQuestion {
    type Error = LlmError;
    type Output = Answer;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        let prompt = prompt_for(self);
        let response = llm(
            &context.client,
            &context.model,
            &prompt,
            ANSWER_QUESTION,
        )
        .await?;

        Ok(Answer(response))
    }
}

impl Answerable for QuizSection {
    type Error = LlmError;
    type Output = Vec<Answer>;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        let mut answers = Vec::with_capacity(self.questions.len());
        for (i, question) in self.questions.iter().enumerate() {
            tracing::debug!("answering question {}...", i + 1);
            answers.push(question.answer(context).await?);
        }
        Ok(answers)
    }
}

impl Answerable for Quiz {
    type Error = LlmError;
    type Output = Vec<Vec<Answer>>;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        let mut all = Vec::with_capacity(self.0.len());
        for (i, section) in self.0.iter().enumerate() {
            let answers = section.answer(context).await?;
            tracing::info!("section {} finished...", i + 1);
            all.push(answers);
        }
        Ok(all)
    }
}
