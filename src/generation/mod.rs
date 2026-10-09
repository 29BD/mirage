mod prompts;
pub mod quiz;
pub mod errors;
mod image;
use google_ai_rs::{Client, GenerativeModel};

pub(crate) async fn llm(
    client: &Client,
    model: &str,
    prompt: &str,
    system: &str,
) -> anyhow::Result<String> {
    let mut gen_model: GenerativeModel = client.generative_model(model);
    if !system.is_empty() {
        gen_model = gen_model.with_system_instruction(system);
    }

    let response = gen_model.generate_content(prompt).await?;
    let text = response.text().to_string();
    Ok(text)
}