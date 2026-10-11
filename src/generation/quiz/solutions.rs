use super::{Quiz, QuizQuestion, QuizSection};
use super::super::errors::{backoff_delay, LlmError};
use super::super::queue::Throttle;
use super::prompts::ANSWER_QUESTION;
use google_ai_rs::{Client, GenerativeModel};
use std::fmt;
use tokio::time::sleep;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Simple(String),
    ModifiedTrueFalse {
        verdict: bool,
        correction: Option<String>,
    },
}

impl Answer {
    pub fn text(&self) -> String {
        match self {
            Answer::Simple(s) => s.trim().to_string(),
            Answer::ModifiedTrueFalse { verdict: true, .. } => "TRUE".to_string(),
            Answer::ModifiedTrueFalse {
                verdict: false,
                correction,
            } => match correction.as_ref().map(|c| c.trim()).filter(|c| !c.is_empty()) {
                Some(c) => format!("FALSE — {c}"),
                None => "FALSE".to_string(),
            },
        }
    }

    /// Whether the answer is usable: non-empty, and for modified
    /// true/false a FALSE verdict must carry its correction.
    pub fn is_complete(&self) -> bool {
        match self {
            Answer::Simple(s) => !s.trim().is_empty(),
            Answer::ModifiedTrueFalse { verdict: true, .. } => true,
            Answer::ModifiedTrueFalse {
                verdict: false,
                correction,
            } => correction.as_ref().map_or(false, |c| !c.trim().is_empty()),
        }
    }

    #[allow(dead_code)]
    pub fn into_inner(self) -> String {
        self.text()
    }
}

impl fmt::Display for Answer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// Parse a modified true/false reply (`TRUE`, or `FALSE — <replacement for
/// the **bold** span>`). The verdict is the first word; anything after a
/// `—`/`-`/`:`/`,` separator is the correction.
pub fn parse_modified_answer(raw: &str) -> Answer {
    let t = raw.trim();
    let token = t.split_whitespace().next().unwrap_or("");
    let verdict = token
        .trim_end_matches(['.', ':', ',', '!', '?'])
        .to_ascii_uppercase();
    let rest = t.get(token.len()..).unwrap_or("").trim_start();
    let correction = rest
        .trim_start_matches(['—', '–', '-', ':', ',', '.'])
        .trim();
    match verdict.as_str() {
        "TRUE" => Answer::ModifiedTrueFalse {
            verdict: true,
            correction: None,
        },
        "FALSE" => Answer::ModifiedTrueFalse {
            verdict: false,
            correction: (!correction.is_empty()).then(|| correction.to_string()),
        },
        _ => Answer::Simple(t.to_string()),
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
        QuizQuestion::ModifiedTrueFalse { question } => {
            format!(
                "{question}\nReply TRUE if the statement is correct, else FALSE — <replacement for the **bold** span>"
            )
        }
        QuizQuestion::OtherQuestion { question } => question.clone(),
    }
}

fn answer_for(question: &QuizQuestion, raw: &str) -> Answer {
    match question {
        QuizQuestion::ModifiedTrueFalse { .. } => parse_modified_answer(raw),
        _ => Answer::Simple(raw.trim().to_string()),
    }
}

impl QuizQuestion {
    async fn answer_with(&self, context: &LlmContext, system: &str) -> Result<Answer, LlmError> {
        let prompt = prompt_for(self);
        let response = llm(context, &prompt, system).await?;
        Ok(answer_for(self, &response))
    }
}

impl Answerable for QuizQuestion {
    type Error = LlmError;
    type Output = Answer;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        self.answer_with(context, ANSWER_QUESTION).await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modified_verdicts() {
        assert_eq!(
            parse_modified_answer("TRUE"),
            Answer::ModifiedTrueFalse {
                verdict: true,
                correction: None,
            }
        );
        assert_eq!(
            parse_modified_answer("FALSE — 9"),
            Answer::ModifiedTrueFalse {
                verdict: false,
                correction: Some("9".to_string()),
            }
        );
        assert_eq!(
            parse_modified_answer("False - less than"),
            Answer::ModifiedTrueFalse {
                verdict: false,
                correction: Some("less than".to_string()),
            }
        );
        assert_eq!(
            parse_modified_answer("FALSE: right triangle"),
            Answer::ModifiedTrueFalse {
                verdict: false,
                correction: Some("right triangle".to_string()),
            }
        );
        // FALSE without a correction parses but is incomplete.
        let bare = parse_modified_answer("FALSE");
        assert!(!bare.is_complete());
        assert_eq!(bare.text(), "FALSE");
    }

    #[test]
    fn modified_text_renders_verdict_and_correction() {
        let t = Answer::ModifiedTrueFalse {
            verdict: true,
            correction: None,
        };
        assert!(t.is_complete());
        assert_eq!(t.text(), "TRUE");
        let f = Answer::ModifiedTrueFalse {
            verdict: false,
            correction: Some("180°".to_string()),
        };
        assert!(f.is_complete());
        assert_eq!(f.text(), "FALSE — 180°");
        assert!(!Answer::Simple("  ".to_string()).is_complete());
    }
}
