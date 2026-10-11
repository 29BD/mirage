use super::super::errors::{LlmError, backoff_delay};
use super::super::queue::Throttle;
use super::progress::{ProgressSink, QuizProgress, WaitReason};
use super::prompts::ANSWER_QUESTION;
use super::{Quiz, QuizQuestion, QuizSection};
use crate::output::option_label;
use google_ai_rs::Client;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
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
            } => match correction
                .as_ref()
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
            {
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
            } => correction.as_ref().is_some_and(|c| !c.trim().is_empty()),
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

/// Plain-text check for answers: reject LaTeX and markdown markers.
/// The robot writes with a pen, so `$`, `\`, backticks, and `**` are out.
pub fn is_plain_text(s: &str) -> bool {
    !(s.contains('$') || s.contains('\\') || s.contains('`') || s.contains("**"))
}

/// Count sentences by splitting on `.`, `!`, `?` runs.
pub fn count_sentences(s: &str) -> usize {
    s.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .count()
}

/// Parse a generic `N-M sentences` requirement out of section instructions.
pub fn parse_sentence_range(instructions: &str) -> Option<(usize, usize)> {
    let chars: Vec<char> = instructions.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let lo: usize = chars[i..j].iter().collect::<String>().parse().ok()?;
            let mut k = j;
            while k < chars.len() && (chars[k] == ' ' || chars[k] == '\t') {
                k += 1;
            }
            if k < chars.len() && (chars[k] == '-' || chars[k] == '–' || chars[k] == '—') {
                k += 1;
                while k < chars.len() && (chars[k] == ' ' || chars[k] == '\t') {
                    k += 1;
                }
                let m = k;
                while k < chars.len() && chars[k].is_ascii_digit() {
                    k += 1;
                }
                if k > m {
                    let hi: usize = chars[m..k].iter().collect::<String>().parse().ok()?;
                    let tail: String = chars[k..].iter().collect::<String>();
                    if tail
                        .trim_start()
                        .to_ascii_lowercase()
                        .starts_with("sentence")
                    {
                        return Some((lo, hi));
                    }
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    None
}

/// Whether section instructions ask for shown work/a solution.
pub fn requires_show_work(instructions: &str) -> bool {
    let lower = instructions.to_ascii_lowercase();
    lower.contains("show") && (lower.contains("work") || lower.contains("solution"))
}

/// System prompt for a section: base rules plus that section's instructions
/// as hard constraints, with generic reinforcements the model tends to drop
/// (show-work, sentence counts).
pub fn system_for_section(instructions: Option<&str>) -> String {
    let mut system = String::from(ANSWER_QUESTION);
    if let Some(raw) = instructions.map(str::trim).filter(|s| !s.is_empty()) {
        system.push_str("\nSection instructions (hard constraints, must obey exactly): ");
        system.push_str(raw);
        if requires_show_work(raw) {
            system.push_str(" Show brief work before each final answer in this section.");
        }
        if let Some((lo, hi)) = parse_sentence_range(raw) {
            system.push_str(&format!(
                " Free-response answers in this section must be {lo}-{hi} sentences long."
            ));
        }
    }
    system
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

/// 0-based identity of one question for progress events: `section` counts
/// sections in the quiz, `question` counts questions in that section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuestionScope {
    pub section: usize,
    pub question: usize,
}

pub(crate) type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// One LLM request. Owned (and `'static`) so backends can be shared across
/// retries and scripted in tests.
#[derive(Debug, Clone)]
pub(crate) struct LlmRequest {
    pub prompt: String,
    pub system: String,
    pub scope: Vec<QuestionScope>,
}

/// Backend that issues an LLM request. Production code talks to Gemini;
/// unit tests swap in a scripted backend via [`LlmContext::with_transport`].
pub(crate) trait Transport: Send + Sync {
    fn send(&self, request: LlmRequest) -> BoxFuture<Result<String, LlmError>>;
}

struct GeminiTransport {
    client: Client,
    model: String,
}

impl Transport for GeminiTransport {
    fn send(&self, request: LlmRequest) -> BoxFuture<Result<String, LlmError>> {
        let client = self.client.clone();
        let model = self.model.clone();
        Box::pin(async move {
            let mut gen_model = client.generative_model(&model);
            if !request.system.is_empty() {
                gen_model = gen_model.with_system_instruction(request.system.as_str());
            }
            gen_model
                .generate_content(request.prompt)
                .await
                .map(|response| response.text().to_string())
                .map_err(LlmError::from)
        })
    }
}

/// Type-erased [`ProgressSink`] carried by [`LlmContext`]. Empty is the
/// silent default; [`LlmContext::with_progress`] installs a real sink.
#[derive(Clone, Default)]
pub(crate) struct ProgressDispatcher {
    inner: Option<Arc<dyn Fn(QuizProgress) + Send + Sync>>,
}

impl ProgressDispatcher {
    pub(crate) fn null() -> Self {
        Self { inner: None }
    }

    pub(crate) fn of<S: ProgressSink + 'static>(sink: &S) -> Self {
        let sink = sink.clone();
        Self {
            inner: Some(Arc::new(move |event| sink.emit(event))),
        }
    }

    pub(crate) fn emit(&self, event: QuizProgress) {
        if let Some(inner) = &self.inner {
            inner(event);
        }
    }
}

#[derive(Clone)]
pub struct LlmContext {
    pub client: Client,
    pub model: String,
    pub throttle: Throttle,
    pub max_retries: u32,
    calls: Arc<AtomicU64>,
    progress: ProgressDispatcher,
    transport: Arc<dyn Transport>,
}

impl fmt::Debug for LlmContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmContext")
            .field("model", &self.model)
            .field("throttle", &self.throttle)
            .field("max_retries", &self.max_retries)
            .field("calls", &self.calls())
            .finish_non_exhaustive()
    }
}

impl LlmContext {
    pub fn new(client: Client, model: String, rpm: u32, max_retries: u32) -> Self {
        let transport: Arc<dyn Transport> = Arc::new(GeminiTransport {
            client: client.clone(),
            model: model.clone(),
        });
        Self {
            client,
            model,
            throttle: Throttle::from_rpm(rpm),
            max_retries,
            calls: Arc::new(AtomicU64::new(0)),
            progress: ProgressDispatcher::null(),
            transport,
        }
    }

    /// Same as [`Self::new`], but shares an existing [`Throttle`] (e.g. a
    /// global limiter) instead of minting a fresh per-quiz one, so concurrent
    /// quizzes pace against one request slot.
    pub fn with_throttle(
        client: Client,
        model: String,
        throttle: Throttle,
        max_retries: u32,
    ) -> Self {
        let transport: Arc<dyn Transport> = Arc::new(GeminiTransport {
            client: client.clone(),
            model: model.clone(),
        });
        Self {
            client,
            model,
            throttle,
            max_retries,
            calls: Arc::new(AtomicU64::new(0)),
            progress: ProgressDispatcher::null(),
            transport,
        }
    }

    /// Clone this context delivering [`QuizProgress`] events to `sink`.
    /// [`Answerable::answer`] uses whatever sink the context carries, so this
    /// plus `answer()` is the silent-by-default compat path, while
    /// `answer_with_progress` wraps both in one call.
    pub fn with_progress<S: ProgressSink + 'static>(&self, sink: S) -> Self {
        Self {
            progress: ProgressDispatcher::of(&sink),
            ..self.clone()
        }
    }

    /// Swap the request backend (unit tests script responses here).
    #[cfg(test)]
    pub(crate) fn with_transport<T: Transport + 'static>(&self, transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            ..self.clone()
        }
    }

    /// Emit one event to the installed sink (no-op by default).
    pub(crate) fn emit(&self, event: QuizProgress) {
        self.progress.emit(event);
    }

    /// Emit one event per question in `scope`.
    pub(crate) fn emit_for(
        &self,
        scope: &[QuestionScope],
        mut event: impl FnMut(QuestionScope) -> QuizProgress,
    ) {
        for id in scope {
            self.emit(event(*id));
        }
    }

    /// Number of LLM requests issued through this context (incl. retries).
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

pub trait Answerable {
    type Error;
    type Output;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error>;
}

/// One throttled, retried request through the context backend, emitting
/// progress for every question in `scope`: `Waiting` around the throttle
/// sleep and rate-limit backoffs, `Generating` immediately before each
/// attempt. Terminal states are the caller's job (`Done` once the reply is
/// accepted, `Failed` on terminal error), so a batch reply that fails
/// validation never reads as done.
#[tracing::instrument(skip(context, request), fields(model = %context.model, prompt_chars = request.prompt.len(), system_chars = request.system.len()))]
async fn llm_loop(context: &LlmContext, request: &LlmRequest) -> Result<String, LlmError> {
    let mut attempt: u32 = 0;
    loop {
        if let Some(waited) = context.throttle.wait_and_report().await {
            context.emit_for(&request.scope, |id| QuizProgress::QuestionWaiting {
                section: id.section,
                question: id.question,
                reason: WaitReason::Throttle,
                wait: waited,
            });
        }
        let call_no = context.calls.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::debug!(call_no, attempt = attempt + 1, "issuing LLM request");

        context.emit_for(&request.scope, |id| QuizProgress::QuestionGenerating {
            section: id.section,
            question: id.question,
        });
        let started = std::time::Instant::now();
        match context.transport.send(request.clone()).await {
            Ok(text) => {
                tracing::debug!(
                    call_no,
                    elapsed_secs = started.elapsed().as_secs_f64(),
                    response_chars = text.len(),
                    "LLM request succeeded"
                );
                return Ok(text);
            }
            Err(err) => {
                attempt += 1;
                if !err.is_rate_limited() || attempt > context.max_retries {
                    tracing::error!(
                        call_no,
                        attempt,
                        error = %err.friendly(),
                        "LLM request failed"
                    );
                    return Err(err);
                }
                let delay = err.retry_after().unwrap_or_else(|| backoff_delay(attempt));
                tracing::warn!(
                    call_no,
                    attempt,
                    max_retries = context.max_retries,
                    retry_in_secs = delay.as_secs_f64(),
                    "rate limited, backing off"
                );
                context.emit_for(&request.scope, |id| QuizProgress::QuestionWaiting {
                    section: id.section,
                    question: id.question,
                    reason: WaitReason::RateLimit,
                    wait: delay,
                });
                sleep(delay).await;
            }
        }
    }
}

/// One request for `scope`, built from `prompt`/`system`.
async fn llm(
    context: &LlmContext,
    prompt: &str,
    system: &str,
    scope: &[QuestionScope],
) -> Result<String, LlmError> {
    llm_loop(
        context,
        &LlmRequest {
            prompt: prompt.to_string(),
            system: system.to_string(),
            scope: scope.to_vec(),
        },
    )
    .await
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
    #[tracing::instrument(skip(self, context, system), fields(system_chars = system.len()))]
    async fn answer_with(
        &self,
        context: &LlmContext,
        system: &str,
        scope: QuestionScope,
    ) -> Result<Answer, LlmError> {
        context.emit(QuizProgress::QuestionStalled {
            section: scope.section,
            question: scope.question,
        });
        let prompt = prompt_for(self);
        tracing::debug!(prompt_chars = prompt.len(), "answering single question");
        let response = match llm(context, &prompt, system, &[scope]).await {
            Ok(response) => response,
            Err(err) => {
                context.emit(QuizProgress::QuestionFailed {
                    section: scope.section,
                    question: scope.question,
                    error: err.friendly(),
                });
                return Err(err);
            }
        };
        context.emit(QuizProgress::QuestionDone {
            section: scope.section,
            question: scope.question,
        });
        let answer = answer_for(self, &response);
        if !answer.is_complete() {
            tracing::warn!("single-question answer incomplete");
        }
        Ok(answer)
    }

    /// Answer this one question, delivering [`QuizProgress`] events to `sink`.
    /// A standalone question reports as section 0, question 0.
    pub async fn answer_with_progress<S: ProgressSink + 'static>(
        &self,
        context: &LlmContext,
        sink: S,
    ) -> Result<Answer, LlmError> {
        self.answer(&context.with_progress(sink)).await
    }
}

impl Answerable for QuizQuestion {
    type Error = LlmError;
    type Output = Answer;

    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        self.answer_with(
            context,
            ANSWER_QUESTION,
            QuestionScope {
                section: 0,
                question: 0,
            },
        )
        .await
    }
}

/// One prompt for a whole section: numbered questions with type tags.
/// Option letters are generated here at solve time; they are not stored.
fn prompt_for_section(section: &QuizSection) -> String {
    let mut out = format!(
        "Answer these {} questions in order. Reply with exactly {} answers: \
each answer starts on its own line as `1) ...`, `2) ...`, and any extra \
lines belong to the answer they follow.\n",
        section.questions.len(),
        section.questions.len(),
    );
    for (i, question) in section.questions.iter().enumerate() {
        let rendered = match question {
            QuizQuestion::MultipleChoice { question, options } => {
                let labeled: Vec<String> = options
                    .iter()
                    .enumerate()
                    .map(|(oi, o)| format!("{}) {}", option_label(oi), o.trim()))
                    .collect();
                format!(
                    "[multiple choice] {question} Options: {}",
                    labeled.join("; ")
                )
            }
            QuizQuestion::TrueFalse { question } => {
                format!("[true/false] {question} (reply TRUE or FALSE)")
            }
            QuizQuestion::ModifiedTrueFalse { question } => {
                format!(
                    "[modified true/false] {question} (reply TRUE, or FALSE — <replacement for the **bold** span>; correction required when FALSE)"
                )
            }
            QuizQuestion::OtherQuestion { question } => {
                format!("[free response] {question}")
            }
        };
        out.push_str(&format!("\n{}. {}", i + 1, rendered.trim()));
    }
    out
}

/// Split a section reply into one block per question. A line starting with
/// `N)` or `N.` opens block N; other non-empty lines continue the current
/// block (shown work, essay sentences). Returns `None` when the reply does
/// not hold exactly one complete answer per question.
fn parse_section_answers(section: &QuizSection, raw: &str) -> Option<Vec<Answer>> {
    let mut blocks: Vec<String> = Vec::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if strip_leading_marker(t).is_some() {
            blocks.push(t.to_string());
        } else {
            let last = blocks.last_mut()?;
            last.push('\n');
            last.push_str(t);
        }
    }
    if blocks.len() != section.questions.len() {
        return None;
    }
    let answers: Vec<Answer> = section
        .questions
        .iter()
        .zip(blocks.iter())
        .map(|(question, block)| {
            let body = strip_leading_marker(block).unwrap_or(block);
            answer_for(question, body)
        })
        .collect();
    if answers.iter().all(Answer::is_complete) {
        Some(answers)
    } else {
        None
    }
}

/// Strip a leading `N)` / `N.` answer marker, returning the body.
fn strip_leading_marker(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    let rest = t.get(digits..)?;
    let mut chars = rest.chars();
    match chars.next() {
        Some('.') | Some(')') => Some(chars.as_str().trim_start()),
        _ => None,
    }
}

impl Answerable for QuizSection {
    type Error = LlmError;
    type Output = Vec<Answer>;

    #[tracing::instrument(skip(self, context), fields(questions = self.questions.len()))]
    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        self.answer_inner(0, context, false).await
    }
}

impl QuizSection {
    fn scope(&self, section: usize) -> Vec<QuestionScope> {
        (0..self.questions.len())
            .map(|question| QuestionScope { section, question })
            .collect()
    }

    /// Shared section path: one batch request for the whole section, falling
    /// back to per-question requests when the batch fails validation.
    /// `section` is this section's 0-based index in the quiz (0 standalone).
    /// A batch covers every question at once, so each question shares the
    /// batch's `Generating`/`Done` states around that single request.
    #[tracing::instrument(skip(self, context), fields(section, questions = self.questions.len()))]
    async fn answer_inner(
        &self,
        section: usize,
        context: &LlmContext,
        per_question: bool,
    ) -> Result<Vec<Answer>, LlmError> {
        let system = system_for_section(self.instructions.as_deref());
        if !per_question {
            let scope = self.scope(section);
            context.emit_for(&scope, |id| QuizProgress::QuestionStalled {
                section: id.section,
                question: id.question,
            });
            let prompt = prompt_for_section(self);
            tracing::debug!(
                prompt_chars = prompt.len(),
                system_chars = system.len(),
                "answering section in one batch request"
            );
            match llm(context, &prompt, &system, &scope).await {
                Ok(raw) => {
                    if let Some(answers) = parse_section_answers(self, &raw) {
                        tracing::info!(
                            questions = answers.len(),
                            "section answered in one request"
                        );
                        context.emit_for(&scope, |id| QuizProgress::QuestionDone {
                            section: id.section,
                            question: id.question,
                        });
                        return Ok(answers);
                    }
                    tracing::warn!(
                        response_chars = raw.len(),
                        "section batch failed validation; falling back to per-question"
                    );
                }
                Err(err) => {
                    let message = err.friendly();
                    tracing::error!(error = %message, "section batch request failed");
                    context.emit_for(&scope, |id| QuizProgress::QuestionFailed {
                        section: id.section,
                        question: id.question,
                        error: message.clone(),
                    });
                    return Err(err);
                }
            }
        }
        tracing::info!(
            questions = self.questions.len(),
            "answering section question-by-question"
        );
        let mut answers = Vec::with_capacity(self.questions.len());
        for (i, question) in self.questions.iter().enumerate() {
            tracing::debug!(
                question = i + 1,
                total = self.questions.len(),
                "answering question"
            );
            answers.push(
                question
                    .answer_with(
                        context,
                        &system,
                        QuestionScope {
                            section,
                            question: i,
                        },
                    )
                    .await?,
            );
        }
        tracing::info!(questions = answers.len(), "section answered per-question");
        Ok(answers)
    }

    /// Answer each question with its own LLM request, so progress states
    /// track exactly one request per question (no shared batch states).
    /// Reports as section 0; prefer [`Quiz::answer_per_question`] for real
    /// indices.
    pub async fn answer_per_question(&self, context: &LlmContext) -> Result<Vec<Answer>, LlmError> {
        self.answer_inner(0, context, true).await
    }

    /// Answer this section (batch first, per-question fallback), delivering
    /// [`QuizProgress`] events to `sink`. Reports as section 0.
    pub async fn answer_with_progress<S: ProgressSink + 'static>(
        &self,
        context: &LlmContext,
        sink: S,
    ) -> Result<Vec<Answer>, LlmError> {
        self.answer(&context.with_progress(sink)).await
    }
}

impl Quiz {
    /// Shared quiz path; `per_question` skips the per-section batch.
    async fn answer_inner(
        &self,
        context: &LlmContext,
        per_question: bool,
    ) -> Result<Vec<Vec<Answer>>, LlmError> {
        let total: usize = self.0.iter().map(|s| s.questions.len()).sum();
        tracing::info!(sections = self.0.len(), questions = total, "answering quiz");
        let mut all = Vec::with_capacity(self.0.len());
        for (i, section) in self.0.iter().enumerate() {
            let section_no = i + 1;
            tracing::debug!(
                section = section_no,
                questions = section.questions.len(),
                "answering section"
            );
            let answers = section.answer_inner(i, context, per_question).await?;
            tracing::info!(
                section = section_no,
                questions = answers.len(),
                "section finished"
            );
            all.push(answers);
        }
        tracing::info!(
            sections = all.len(),
            llm_calls = context.calls(),
            "quiz answered"
        );
        Ok(all)
    }

    /// Answer every section question-by-question (no batch requests), so
    /// progress states track exactly one request per question.
    pub async fn answer_per_question(
        &self,
        context: &LlmContext,
    ) -> Result<Vec<Vec<Answer>>, LlmError> {
        self.answer_inner(context, true).await
    }

    /// Answer the quiz (batch first, per-question fallback), delivering
    /// [`QuizProgress`] events to `sink`.
    pub async fn answer_with_progress<S: ProgressSink + 'static>(
        &self,
        context: &LlmContext,
        sink: S,
    ) -> Result<Vec<Vec<Answer>>, LlmError> {
        self.answer(&context.with_progress(sink)).await
    }
}

impl Answerable for Quiz {
    type Error = LlmError;
    type Output = Vec<Vec<Answer>>;

    #[tracing::instrument(skip(self, context), fields(sections = self.0.len()))]
    async fn answer(&self, context: &LlmContext) -> Result<Self::Output, Self::Error> {
        self.answer_inner(context, false).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::quiz::progress::MpscSink;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::Duration;

    /// Recording sink for asserting event order.
    #[derive(Debug, Clone, Default)]
    struct RecSink {
        events: Arc<Mutex<Vec<QuizProgress>>>,
    }

    impl ProgressSink for RecSink {
        fn emit(&self, event: QuizProgress) {
            self.events.lock().unwrap().push(event);
        }
    }

    impl RecSink {
        fn take(&self) -> Vec<QuizProgress> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }
    }

    /// Scripted backend: pops one canned reply per request.
    struct ScriptTransport {
        script: Mutex<VecDeque<Result<String, LlmError>>>,
    }

    impl ScriptTransport {
        fn replies(replies: Vec<Result<String, LlmError>>) -> Self {
            Self {
                script: Mutex::new(replies.into()),
            }
        }
    }

    impl Transport for ScriptTransport {
        fn send(&self, _request: LlmRequest) -> BoxFuture<Result<String, LlmError>> {
            let next = self
                .script
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted transport exhausted");
            Box::pin(async move { next })
        }
    }

    fn rate_limited(msg: &str) -> LlmError {
        LlmError::from(google_ai_rs::Error::from(msg))
    }

    async fn scripted_context(
        sink: RecSink,
        rpm: u32,
        max_retries: u32,
        script: Vec<Result<String, LlmError>>,
    ) -> LlmContext {
        let client = Client::new("dummy-key-for-offline-test")
            .await
            .expect("dummy client builds offline");
        LlmContext::new(client, "test-model".to_string(), rpm, max_retries)
            .with_transport(ScriptTransport::replies(script))
            .with_progress(sink)
    }

    fn two_question_section() -> QuizSection {
        QuizSection {
            instructions: None,
            questions: vec![
                QuizQuestion::TrueFalse {
                    question: "The sky is blue.".to_string(),
                },
                QuizQuestion::OtherQuestion {
                    question: "Why?".to_string(),
                },
            ],
        }
    }

    fn stalled(section: usize, question: usize) -> QuizProgress {
        QuizProgress::QuestionStalled { section, question }
    }

    fn generating(section: usize, question: usize) -> QuizProgress {
        QuizProgress::QuestionGenerating { section, question }
    }

    fn done(section: usize, question: usize) -> QuizProgress {
        QuizProgress::QuestionDone { section, question }
    }

    fn waiting(
        section: usize,
        question: usize,
        reason: WaitReason,
        wait: Duration,
    ) -> QuizProgress {
        QuizProgress::QuestionWaiting {
            section,
            question,
            reason,
            wait,
        }
    }

    /// Throttle waits are real sleeps (~1s at 60rpm), so exact durations
    /// vary: assert the range, then canonicalize for exact order comparison.
    fn normalize_waits(events: &mut [QuizProgress]) {
        for e in events.iter_mut() {
            if let QuizProgress::QuestionWaiting { wait, .. } = e {
                assert!(
                    *wait >= Duration::from_millis(900),
                    "throttle wait too short: {wait:?}"
                );
                assert!(
                    *wait <= Duration::from_secs(2),
                    "throttle wait too long: {wait:?}"
                );
                *wait = Duration::from_secs(1);
            }
        }
    }

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

    #[test]
    fn plain_text_rejects_latex_and_markdown() {
        assert!(is_plain_text("x = 7, y = 3"));
        assert!(is_plain_text("The sale price is P680."));
        assert!(!is_plain_text("x = \\frac{3}{4}"));
        assert!(!is_plain_text("$x = 5$"));
        assert!(!is_plain_text("**bold** answer"));
        assert!(!is_plain_text("`code`"));
    }

    #[test]
    fn sentence_range_parses_generic_counts() {
        assert_eq!(
            parse_sentence_range("In 3-5 sentences, explain why."),
            Some((3, 5))
        );
        assert_eq!(
            parse_sentence_range("Answer in 2 – 4 sentences."),
            Some((2, 4))
        );
        assert_eq!(parse_sentence_range("Show your work."), None);
        assert_eq!(parse_sentence_range("15 points, 3 points each"), None);
    }

    #[test]
    fn instructions_detect_show_work() {
        assert!(requires_show_work("Show your complete solution."));
        assert!(requires_show_work("Show your work where indicated."));
        assert!(!requires_show_work("Circle the letter."));
    }

    #[test]
    fn section_system_embeds_instructions() {
        let system = system_for_section(Some("In 3-5 sentences, explain why."));
        assert!(system.contains("hard constraints"));
        assert!(system.contains("In 3-5 sentences"));
        assert!(system.contains("3-5 sentences long"));
        assert!(system.contains("plain text only"));
        let bare = system_for_section(None);
        assert!(!bare.contains("must obey exactly"));
    }

    #[test]
    fn counts_sentences() {
        assert_eq!(count_sentences("One. Two! Three?"), 3);
        assert_eq!(count_sentences("  "), 0);
    }

    fn sample_section() -> QuizSection {
        QuizSection {
            instructions: Some("Show your work.".to_string()),
            questions: vec![
                QuizQuestion::MultipleChoice {
                    question: "What is 2 + 2?".to_string(),
                    options: vec!["3".to_string(), "4".to_string()],
                },
                QuizQuestion::ModifiedTrueFalse {
                    question: "The sky is **green**.".to_string(),
                },
                QuizQuestion::OtherQuestion {
                    question: "Explain.".to_string(),
                },
            ],
        }
    }

    #[test]
    fn section_prompt_numbers_and_labels_options() {
        let prompt = prompt_for_section(&sample_section());
        assert!(prompt.contains("1. [multiple choice] What is 2 + 2?"));
        assert!(prompt.contains("a) 3; b) 4"));
        assert!(prompt.contains("2. [modified true/false]"));
        assert!(prompt.contains("3. [free response] Explain."));
    }

    #[test]
    fn section_reply_parses_one_block_per_question() {
        let raw = "1) b\n2) FALSE — blue\n3) Mass attracts mass.\n   It pulls things down.";
        let answers = parse_section_answers(&sample_section(), raw).expect("should parse");
        assert_eq!(answers.len(), 3);
        assert_eq!(answers[0], Answer::Simple("b".to_string()));
        assert_eq!(
            answers[1],
            Answer::ModifiedTrueFalse {
                verdict: false,
                correction: Some("blue".to_string()),
            }
        );
        assert_eq!(
            answers[2],
            Answer::Simple("Mass attracts mass.\nIt pulls things down.".to_string())
        );
    }

    #[test]
    fn section_reply_rejects_bad_counts_and_bare_false() {
        // Too few blocks.
        assert!(parse_section_answers(&sample_section(), "1) b\n2) TRUE").is_none());
        // Modified FALSE without a correction is incomplete.
        assert!(parse_section_answers(&sample_section(), "1) b\n2) FALSE\n3) x").is_none());
        // Leading prose with no marker.
        assert!(
            parse_section_answers(&sample_section(), "Here you go\n1) b\n2) TRUE\n3) x").is_none()
        );
    }

    #[tokio::test]
    async fn progress_batch_success_emits_ordered_events() {
        let sink = RecSink::default();
        let ctx = scripted_context(
            sink.clone(),
            60,
            6,
            vec![Ok(
                "1) TRUE\n2) Because air scatters blue light.".to_string()
            )],
        )
        .await;
        let section = two_question_section();

        let answers = section.answer(&ctx).await.expect("batch should parse");
        assert_eq!(
            answers,
            vec![
                Answer::Simple("TRUE".to_string()),
                Answer::Simple("Because air scatters blue light.".to_string()),
            ]
        );
        // One shared request: both questions stall, then generate and finish
        // together; a fresh throttle claims its first slot immediately.
        assert_eq!(
            sink.take(),
            vec![
                stalled(0, 0),
                stalled(0, 1),
                generating(0, 0),
                generating(0, 1),
                done(0, 0),
                done(0, 1),
            ]
        );
        assert_eq!(ctx.calls(), 1);
    }

    #[tokio::test]
    async fn progress_fallback_emits_per_question_events() {
        let sink = RecSink::default();
        let ctx = scripted_context(
            sink.clone(),
            60,
            6,
            vec![
                // Batch reply with no markers: fails validation.
                Ok("Here are your answers, enjoy!".to_string()),
                Ok("TRUE".to_string()),
                Ok("Because air scatters blue light.".to_string()),
            ],
        )
        .await;
        let section = two_question_section();

        let answers = section.answer(&ctx).await.expect("fallback should answer");
        assert_eq!(
            answers,
            vec![
                Answer::Simple("TRUE".to_string()),
                Answer::Simple("Because air scatters blue light.".to_string()),
            ]
        );
        // Batch attempt stalls/generates without resolving (its reply fails
        // validation); each fallback question then runs its own cycle,
        // pacing against the throttle between requests.
        let mut events = sink.take();
        normalize_waits(&mut events);
        assert_eq!(
            events,
            vec![
                stalled(0, 0),
                stalled(0, 1),
                generating(0, 0),
                generating(0, 1),
                stalled(0, 0),
                waiting(0, 0, WaitReason::Throttle, Duration::from_secs(1)),
                generating(0, 0),
                done(0, 0),
                stalled(0, 1),
                waiting(0, 1, WaitReason::Throttle, Duration::from_secs(1)),
                generating(0, 1),
                done(0, 1),
            ]
        );
        assert_eq!(ctx.calls(), 3);
    }

    #[tokio::test]
    async fn progress_per_question_mode_skips_batch() {
        let sink = RecSink::default();
        let ctx = scripted_context(
            sink.clone(),
            60,
            6,
            vec![
                Ok("TRUE".to_string()),
                Ok("Because air scatters blue light.".to_string()),
            ],
        )
        .await;

        let answers = two_question_section()
            .answer_per_question(&ctx)
            .await
            .expect("per-question should answer");
        assert_eq!(answers.len(), 2);
        let mut events = sink.take();
        normalize_waits(&mut events);
        assert_eq!(
            events,
            vec![
                stalled(0, 0),
                generating(0, 0),
                done(0, 0),
                stalled(0, 1),
                waiting(0, 1, WaitReason::Throttle, Duration::from_secs(1)),
                generating(0, 1),
                done(0, 1),
            ]
        );
        assert_eq!(ctx.calls(), 2);
    }

    #[tokio::test]
    async fn progress_throttle_wait_emits_waiting_with_duration() {
        let sink = RecSink::default();
        // 1s interval keeps the test fast; the first slot is free.
        let ctx = scripted_context(
            sink.clone(),
            60,
            6,
            vec![Ok("a".to_string()), Ok("b".to_string())],
        )
        .await;
        let scope = [QuestionScope {
            section: 0,
            question: 0,
        }];
        let request = |prompt: &str| LlmRequest {
            prompt: prompt.to_string(),
            system: String::new(),
            scope: scope.to_vec(),
        };

        llm_loop(&ctx, &request("first"))
            .await
            .expect("first request succeeds");
        assert_eq!(
            sink.take(),
            vec![generating(0, 0)],
            "first slot is immediate: no waiting event"
        );

        llm_loop(&ctx, &request("second"))
            .await
            .expect("second request succeeds");
        let events = sink.take();
        assert_eq!(events.len(), 2);
        match &events[0] {
            QuizProgress::QuestionWaiting {
                section: 0,
                question: 0,
                reason: WaitReason::Throttle,
                wait,
            } => {
                assert!(*wait >= Duration::from_millis(900), "wait was {wait:?}");
                assert!(*wait <= Duration::from_secs(2), "wait was {wait:?}");
            }
            other => panic!("expected throttle waiting, got {other:?}"),
        }
        assert_eq!(events[1], generating(0, 0));
    }

    #[tokio::test]
    async fn progress_retry_emits_backoff_then_regenerates() {
        let sink = RecSink::default();
        let ctx = scripted_context(
            sink.clone(),
            60,
            6,
            vec![
                // Server hint clamps to a 1s backoff: fast, yet exercises the
                // retry_after path instead of the blind exponential one.
                Err(rate_limited(
                    "429 ResourceExhausted: Please retry in 0.01s.",
                )),
                Ok("TRUE".to_string()),
            ],
        )
        .await;

        let answer = QuizQuestion::TrueFalse {
            question: "The sky is blue.".to_string(),
        }
        .answer(&ctx)
        .await
        .expect("retry should recover");
        assert_eq!(answer, Answer::Simple("TRUE".to_string()));

        let events = sink.take();
        assert_eq!(events.len(), 5);
        assert_eq!(events[0], stalled(0, 0));
        assert_eq!(events[1], generating(0, 0));
        match &events[2] {
            QuizProgress::QuestionWaiting {
                section: 0,
                question: 0,
                reason: WaitReason::RateLimit,
                wait,
            } => assert_eq!(*wait, Duration::from_secs(1)),
            other => panic!("expected rate-limit waiting, got {other:?}"),
        }
        // The retried attempt regenerates, then resolves.
        assert_eq!(events[3], generating(0, 0));
        assert_eq!(events[4], done(0, 0));
        assert_eq!(ctx.calls(), 2);
    }

    #[tokio::test]
    async fn progress_terminal_failure_emits_failed() {
        let sink = RecSink::default();
        let ctx = scripted_context(
            sink.clone(),
            60,
            0,
            vec![Err(rate_limited("boom: malformed request"))],
        )
        .await;

        let err = QuizQuestion::TrueFalse {
            question: "The sky is blue.".to_string(),
        }
        .answer(&ctx)
        .await
        .expect_err("non-retryable error should surface");
        assert!(err.friendly().contains("boom"));

        let events = sink.take();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], stalled(0, 0));
        assert_eq!(events[1], generating(0, 0));
        match &events[2] {
            QuizProgress::QuestionFailed {
                section: 0,
                question: 0,
                error,
            } => assert!(error.contains("boom"), "error was {error:?}"),
            other => panic!("expected failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn answer_with_progress_delivers_events_over_mpsc() {
        let (mpsc, mut rx) = MpscSink::channel();
        let client = Client::new("dummy-key-for-offline-test")
            .await
            .expect("dummy client builds offline");
        let ctx = LlmContext::new(client, "test-model".to_string(), 60, 6).with_transport(
            ScriptTransport::replies(vec![Ok("1) TRUE\n2) Because.".to_string())]),
        );

        let quiz = Quiz(vec![two_question_section()]);
        let answers = quiz
            .answer_with_progress(&ctx, mpsc)
            .await
            .expect("batch should parse");
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].len(), 2);

        let mut got = Vec::new();
        while let Some(e) = rx.recv().await {
            got.push(e);
            if got.len() == 6 {
                break;
            }
        }
        assert_eq!(
            got,
            vec![
                stalled(0, 0),
                stalled(0, 1),
                generating(0, 0),
                generating(0, 1),
                done(0, 0),
                done(0, 1),
            ]
        );
    }
}
