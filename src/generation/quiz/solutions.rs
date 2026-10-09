use super::{Quiz, QuizQuestion, QuizSection};
use super::super::errors::{backoff_delay, LlmError};
use super::super::queue::Throttle;
use super::prompts::ANSWER_QUESTION;
use google_ai_rs::{Client, GenerativeModel};
use std::fmt;
use tokio::time::sleep;

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

#[derive(Debug, Clone)]
pub struct LlmContext {
    pub client: Client,
    pub model: String,
    pub throttle: Throttle,
    pub max_retries: u32,
}

impl LlmContext {
    pub fn new(client: Client, model: String, rpm: u32, max_retries: u32) -> Self {
        Self {
            client,
            model,
            throttle: Throttle::from_rpm(rpm),
            max_retries,
        }
    }
}

pub trait Answerable {
    type Error;
    type Output;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error>;
}

async fn llm(
    context: &LlmContext,
    prompt: &str,
    system: &str,
) -> Result<String, LlmError> {
    let mut attempt: u32 = 0;
    loop {
        context.throttle.wait().await;

        let mut gen_model: GenerativeModel = context.client.generative_model(&context.model);
        if !system.is_empty() {
            gen_model = gen_model.with_system_instruction(system);
        }

        match gen_model.generate_content(prompt).await {
            Ok(response) => return Ok(response.text().to_string()),
            Err(err) => {
                attempt += 1;
                let err = LlmError::from(err);
                if !err.is_rate_limited() || attempt > context.max_retries {
                    return Err(err);
                }
                let delay = err.retry_after().unwrap_or_else(|| backoff_delay(attempt));
                tracing::warn!(
                    "rate limited (attempt {attempt}/{}), retrying in {:.0}s...",
                    context.max_retries,
                    delay.as_secs_f64(),
                );
                sleep(delay).await;
            }
        }
    }
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
        let response = llm(context, &prompt, ANSWER_QUESTION).await?;

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
