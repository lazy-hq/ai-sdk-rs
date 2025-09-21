//! Integration tests for the OpenAI provider.

use aisdk::{
    core::{LanguageModelRequest, Message},
    providers::openai::OpenAI,
};
use dotenv::dotenv;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[tokio::test]
async fn test_generate_text_with_openai() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Respond with exactly the word 'hello' in all lowercase.Do not include any punctuation, prefixes, or suffixes.")
        .build()
        .generate_text()
        .await;
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

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Respond with exactly the word 'hello' in all lowercase.Do not include any punctuation, prefixes, or suffixes.")
        .build()
        .stream_text()
        .await
        .unwrap();

    let mut stream = response.stream;

    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        if let Ok(lang_resp) = chunk
            && !lang_resp.text.is_empty()
        {
            buf.push_str(&lang_resp.text);
        }
    }

    if let Some(model) = response.model {
        assert!(model.starts_with("gpt-4o"));
    }

    assert!(buf.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_system_prompt() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    // with custom openai provider settings
    let openai = OpenAI::builder()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .model_name("gpt-4o")
        .build()
        .expect("Failed to build OpenAIProviderSettings");

    let result = LanguageModelRequest::builder()
        .model(openai)
        .system("Only say hello whatever the user says. all lowercase no punctuation, prefixes, or suffixes.")
        .prompt("Hello how are you doing?")
        .build()
        .generate_text()
        .await;

    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_messages() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    // with custom openai provider settings
    let openai = OpenAI::builder()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .model_name("gpt-4o")
        .build()
        .expect("Failed to build OpenAIProviderSettings");

    let messages = Message::builder()
        .system("You are a helpful assistant.")
        .user("Whatsup?, Surafel is here")
        .assistant("How could I help you?")
        .user("Could you tell my name?")
        .build();

    let mut language_model = LanguageModelRequest::builder()
        .model(openai)
        .messages(messages)
        .build();

    let result = language_model.generate_text().await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("Surafel"));
}

#[tokio::test]
async fn test_generate_text_with_messages_and_system_prompt() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let messages = Message::builder()
        .system("Only say hello whatever the user says. \n all lowercase no punctuation, prefixes, or suffixes.")
        .user("Whatsup?, Surafel is here")
        .assistant("How could I help you?")
        .user("Could you tell my name?")
        .build();

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Only say hello whatever the user says. all lowercase no punctuation, prefixes, or suffixes.")
        .messages(messages)
        .build()
        .generate_text()
        .await;

    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_output_schema() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct User {
        name: String,
        age: u32,
        email: String,
        phone: String,
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("generate user with dummy data, and and name of 'John Doe'")
        .schema::<User>()
        .build()
        .generate_text()
        .await
        .unwrap();

    let result: User = result.into_schema().unwrap();

    assert_eq!(result.name, "John Doe");
}
