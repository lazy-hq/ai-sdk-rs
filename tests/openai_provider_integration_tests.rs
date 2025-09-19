//! Integration tests for the OpenAI provider.

use aisdk::{
    Error,
    core::{GenerateTextCallOptions, Message, generate_stream, generate_text},
    providers::openai::OpenAI,
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

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase.\n 
                Do not include any punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(OpenAI::new("gpt-4o"), options).await;
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

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase\n 
            10 times each on new lines. Do not include any punctuation,\n 
            prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let response = generate_stream(OpenAI::new("gpt-4o"), options)
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

    let options = GenerateTextCallOptions::builder()
        .system(Some(
            "Only say hello whatever the user says. \n 
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .prompt(Some("Hello how are you doing?".to_string()))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(openai, options).await;
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

    let options = GenerateTextCallOptions::builder()
        .messages(Some(messages))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(openai, options).await;
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

    let options = GenerateTextCallOptions::builder()
        .system(Some(
            "Only say hello whatever the user says. \n
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .messages(Some(messages))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(OpenAI::new("gpt-4o"), options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_messages_and_inmessage_system_prompt() {
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

    let options = GenerateTextCallOptions::builder()
        .messages(Some(messages))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(OpenAI::new("gpt-4o"), options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_builder_with_both_prompt_and_messages() {
    dotenv().ok();

    // the builder should fail
    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Only say hello whatever the user says. \n 
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .messages(Some(vec![Message::User(
            "Whatsup?, Surafel is here".into(),
        )]))
        .build();

    assert!(options.is_err());
    match options.unwrap_err() {
        Error::InvalidInput(msg) => {
            assert!(msg.contains("Cannot set both prompt and messages"));
        }
        _ => {
            panic!("Expected InvalidInput error");
        }
    }
}

#[tokio::test]
async fn test_generate_text_builder_with_no_prompt_and_messages() {
    dotenv().ok();

    // the builder should fail
    let options = GenerateTextCallOptions::builder()
        .system(Some("You are a helpful assistant.".to_string()))
        .build();

    assert!(options.is_err());

    match options.unwrap_err() {
        Error::InvalidInput(msg) => {
            assert!(msg.contains("Messages or prompt must be set"));
        }
        _ => {
            panic!("Expected InvalidInput error");
        }
    }
}
