//! Integration tests for the Anthropic provider.

use aisdk::{
    core::{GenerateTextCallOptions, Message, generate_stream, generate_text},
    providers::anthropic::Anthropic,
};
use dotenv::dotenv;
use futures::StreamExt;

#[tokio::test]
async fn test_generate_text_with_anthropic() {
    dotenv().ok();

    // This test requires a valid Anthropic API key to be set in the environment.
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase.\n 
                Do not include any punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(Anthropic::new("claude-3-5-haiku-20241022"), options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_stream_with_anthropic() {
    dotenv().ok();

    // This test requires a valid Anthropic API key to be set in the environment.
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase\n 
            10 times each on new lines. Do not include any punctuation,\n 
            prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let response = generate_stream(Anthropic::new("claude-3-5-haiku-20241022"), options)
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
        assert!(model.starts_with("claude-3-5-haiku"));
    }

    assert!(buf.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_system_prompt() {
    dotenv().ok();

    // This test requires a valid Anthropic API key to be set in the environment.
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    // with custom anthropic provider settings
    let anthropic = Anthropic::builder()
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap())
        .model_name("claude-3-5-haiku-20241022")
        .build()
        .expect("Failed to build AnthropicProviderSettings");

    let options = GenerateTextCallOptions::builder()
        .system(Some(
            "Only say hello whatever the user says. \n 
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .prompt(Some("Hello how are you doing?".to_string()))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(anthropic, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_messages() {
    dotenv().ok();

    // This test requires a valid Anthropic API key to be set in the environment.
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    // with custom anthropic provider settings
    let anthropic = Anthropic::builder()
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap())
        .model_name("claude-3-5-haiku-20241022")
        .build()
        .expect("Failed to build AnthropicProviderSettings");

    let messages = Message::builder()
        .system("You are a helpful assistant.")
        .user("Whatsup?, Rohan is here")
        .assistant("How could I help you?")
        .user("Could you tell my name?")
        .build();

    let options = GenerateTextCallOptions::builder()
        .messages(Some(messages))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(anthropic, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("Rohan"));
}

#[tokio::test]
async fn test_generate_text_with_messages_and_system_prompt() {
    dotenv().ok();

    // This test requires a valid Anthropic API key to be set in the environment.
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    let messages = Message::builder()
        .system("Only say hello whatever the user says. \n all lowercase no punctuation, prefixes, or suffixes.")
        .user("Whatsup?, Rohan is here")
        .assistant("How could I help you?")
        .user("Could you tell my name?")
        .build();

    let options = GenerateTextCallOptions::builder()
        .system(Some(
            "Only say hello whatever the user says. \n
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .messages(Some(messages))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(Anthropic::new("claude-3-5-haiku-20241022"), options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_messages_and_inmessage_system_prompt() {
    dotenv().ok();

    // This test requires a valid Anthropic API key to be set in the environment.
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    let messages = Message::builder()
        .system("Only say hello whatever the user says. \n all lowercase no punctuation, prefixes, or suffixes.")
        .user("Whatsup?, Rohan is here")
        .assistant("How could I help you?")
        .user("Could you tell my name?")
        .build();

    let options = GenerateTextCallOptions::builder()
        .messages(Some(messages))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(Anthropic::new("claude-3-5-haiku-20241022"), options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}
