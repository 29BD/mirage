use anyhow::{Context, Result};
use clap::Parser;
use google_ai_rs::Client;
use mirage::generation::quiz::Quiz;
use mirage::generation::quiz::solutions::{Answerable, LlmContext};
use mirage::output;

/// Transcribe a quiz image into pretty markdown, with answers.
#[derive(Debug, Parser)]
#[command(
    name = "mirage",
    version,
    about = "Transcribe quiz images to markdown + answers"
)]
struct Args {
    /// Path to the quiz image (png, jpg/jpeg, webp, heic/heif), or a public
    /// http(s):// image URL.
    image: String,

    /// Gemini model to use for transcription and answering.
    #[arg(long, env = "GEMINI_MODEL", default_value = "gemini-3.1-flash-lite")]
    model: String,

    /// Only transcribe the quiz; skip generating answers.
    #[arg(long)]
    no_answers: bool,

    /// Max LLM requests per minute while answering (free tier is ~15).
    #[arg(long, env = "MIRAGE_RPM", default_value_t = 10)]
    rpm: u32,

    /// Retries per request when the API reports rate limiting.
    #[arg(long, env = "MIRAGE_MAX_RETRIES", default_value_t = 6)]
    max_retries: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    tracing::debug!(
        image = %args.image,
        model = %args.model,
        no_answers = args.no_answers,
        rpm = args.rpm,
        max_retries = args.max_retries,
        "starting mirage"
    );

    let api_key = std::env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY must be set (export GEMINI_API_KEY=...)")?;
    if api_key.trim().is_empty() {
        tracing::error!("GEMINI_API_KEY is empty");
        anyhow::bail!("GEMINI_API_KEY is empty");
    }
    let client = Client::new(api_key)
        .await
        .context("failed to create client")?;
    tracing::debug!("gemini client created");

    let started = std::time::Instant::now();
    tracing::info!(image = %args.image, model = %args.model, "transcribing quiz");
    let quiz = Quiz::from_image_or_url(&client, &args.model, &args.image).await?;
    let total: usize = quiz.0.iter().map(|s| s.questions.len()).sum();
    tracing::info!(
        sections = quiz.0.len(),
        questions = total,
        elapsed_secs = started.elapsed().as_secs_f64(),
        "transcription complete"
    );

    if args.no_answers {
        tracing::info!("skipping answers (--no-answers)");
        print!("{}", output::render_quiz(&quiz, None));
        return Ok(());
    }

    let rpm = args.rpm.clamp(1, 60);
    tracing::info!(
        questions = total,
        rpm,
        max_retries = args.max_retries,
        "answering quiz"
    );
    let context = LlmContext::new(client, args.model, rpm, args.max_retries);
    match quiz.answer(&context).await {
        Ok(answers) => {
            tracing::info!(
                llm_calls = context.calls(),
                elapsed_secs = started.elapsed().as_secs_f64(),
                "quiz solved"
            );
            print!("{}", output::render_quiz(&quiz, Some(&answers)));
            Ok(())
        }
        Err(err) => {
            tracing::error!(
                error = %err.friendly(),
                llm_calls = context.calls(),
                "answering failed; printing transcript without answers"
            );
            // Don't lose the transcription after minutes of paced requests:
            // print it answer-free before failing with a one-line message
            // (the raw error embeds a huge proto details blob).
            print!("{}", output::render_quiz(&quiz, None));
            Err(anyhow::anyhow!(err.friendly())
                .context("answering failed (transcript printed above)"))
        }
    }
}
