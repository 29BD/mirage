mod generation;
mod output;

use anyhow::{Context, Result};
use clap::Parser;
use generation::quiz::solutions::{Answerable, LlmContext};
use generation::quiz::Quiz;
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

    let answers = if args.no_answers {
        None
    } else {
        let context = LlmContext {
            client,
            model: args.model,
        };
        tracing::info!("answering...");
        let answers = quiz.answer(&context).await?;
        Some(answers)
    };

    print!("{}", output::render_quiz(&quiz, answers.as_deref()));

    Ok(())
}
