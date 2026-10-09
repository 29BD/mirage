use super::{QuizQuestion, QuizSection, Quiz};
use google_ai_rs::{GenerativeModel, Client};
use super::super::errors::LlmError;
use super::prompts::ANSWER_QUESTION;

pub struct Answer(String);

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
    let text = response.text().to_string();
    Ok(text)
}


impl Answerable for QuizQuestion {
    type Error = LlmError;
    type Output = Answer;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        let (instructions, options) = match self {
            QuizQuestion::MultipleChoice { question, options } => (question, Some(options)),
            QuizQuestion::TrueFalse { question } => (question, Some(&vec!["True".to_owned(), "False".to_owned()])),
            QuizQuestion::OtherQuestion { question } => (question, None),
        };

        let response = llm(
            &context.client, 
            &context.model, 
            &format!(
                "{}\n{}",
                instructions,
                options.unwrap_or(&vec!["".to_owned()]).join(", "),
            ),
            ANSWER_QUESTION
        ).await?;

        Ok(Answer(response))
    }
}
