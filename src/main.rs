use anyhow::{Context, Result};
use clap::Parser;
use mirage::generation::quiz::solutions::{Answerable, LlmContext};
use mirage::generation::quiz::Quiz;
use mirage::output;
use google_ai_rs::Client;
use std::path::PathBuf;

/// Transcribe a quiz image into pretty markdown, with answers.
#[derive(Debug, Parser)]
#[command(name = "mirage", version, about = "Transcribe quiz images to markdown + answers")]
struct Args {
    /// Path to the quiz image (png, jpg/jpeg, webp, heic/heif).
    image: PathBuf,

    /// Gemini model to use for transcription and answering.
    #[arg(long, env = "GEMINI_MODEL", default_value = "gemini-2.5-flash-lite")]
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

    let api_key = std::env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY must be set (export GEMINI_API_KEY=...)")?;
    if api_key.trim().is_empty() {
        anyhow::bail!("GEMINI_API_KEY is empty");
    }
    let client = Client::new(api_key).await.context("failed to create client")?;

    tracing::info!("transcribing {}...", args.image.display());
    let quiz = Quiz::from_image(&client, &args.model, &args.image).await?;
    let total: usize = quiz.0.iter().map(|s| s.questions.len()).sum();
    tracing::info!(
        "transcribed {} section(s), {} question(s)",
        quiz.0.len(),
        total
    );

    if args.no_answers {
        print!("{}", output::render_quiz(&quiz, None));
        return Ok(());
    }

    let rpm = args.rpm.clamp(1, 60);
    tracing::info!("answering {total} question(s) at ~{rpm} req/min...");
    let context = LlmContext::new(client, args.model, rpm, args.max_retries);
    match quiz.answer(&context).await {
        Ok(answers) => {
            tracing::info!("solved with {} LLM call(s)", context.calls());
            print!("{}", output::render_quiz(&quiz, Some(&answers)));
            Ok(())
        }
        Err(err) => {
            // Don't lose the transcription after minutes of paced requests:
            // print it answer-free before failing with a one-line message
            // (the raw error embeds a huge proto details blob).
            print!("{}", output::render_quiz(&quiz, None));
            Err(anyhow::anyhow!(err.friendly())
                .context("answering failed (transcript printed above)"))
        }
    }
}
