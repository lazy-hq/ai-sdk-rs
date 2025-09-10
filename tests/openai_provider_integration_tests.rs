//! Integration tests for the OpenAI provider.

use aisdk::{
    core::{GenerateTextCallOptions, generate_stream, generate_text},
    providers::openai::{OpenAI, OpenAIProviderSettings},
};
use dotenv::dotenv;
use futures::StreamExt;

#[tokio::test]
async fn test_generate_text_with_openai() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let settings = OpenAIProviderSettings::builder()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .model_name("gpt-4o".to_string())
        .build()
        .expect("Failed to build OpenAIProviderSettings");

    let openai = OpenAI::new(settings);

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase.\n 
                Do not include any punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(openai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_stream_with_openai() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let settings = OpenAIProviderSettings::builder()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .model_name("gpt-4o".to_string())
        .build()
        .expect("Failed to build OpenAIProviderSettings");

    let openai = OpenAI::new(settings);

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase\n 
            10 times each on new lines. Do not include any punctuation,\n 
            prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let mut stream = generate_stream(openai, options).await.unwrap().stream;
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(lang_resp) => {
                if !lang_resp.text.is_empty() {
                    buf.push_str(&lang_resp.text);
                }
            }
            _ => {}
        }
    }
    assert!(buf.contains("hello"));
}
