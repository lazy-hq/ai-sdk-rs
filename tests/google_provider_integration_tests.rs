//! Integration tests for Google providers (GoogleGenerativeAI and VertexAI).

use aisdk::{
    core::{GenerateTextCallOptions, Message, generate_stream, generate_text},
    providers::google::{GoogleGenerativeAI, VertexAI},
};
use dotenv::dotenv;
use futures::StreamExt;

// ===== GoogleGenerativeAI Tests =====

#[tokio::test]
async fn test_generate_text_with_google_generative_ai() {
    dotenv().ok();

    // This test requires a valid Google AI API key to be set in the environment.
    if std::env::var("GOOGLE_API_KEY").is_err() {
        println!("Skipping test: GOOGLE_API_KEY not set");
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

    let result = generate_text(GoogleGenerativeAI::new("gemini-1.5-flash"), options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_stream_with_google_generative_ai() {
    dotenv().ok();

    // This test requires a valid Google AI API key to be set in the environment.
    if std::env::var("GOOGLE_API_KEY").is_err() {
        println!("Skipping test: GOOGLE_API_KEY not set");
        return;
    }

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase\n 
            5 times each on new lines. Do not include any punctuation,\n 
            prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let response = generate_stream(GoogleGenerativeAI::new("gemini-1.5-flash"), options)
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
        assert!(model.starts_with("gemini-1.5-flash"));
    }

    assert!(buf.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_google_ai_system_prompt() {
    dotenv().ok();

    // This test requires a valid Google AI API key to be set in the environment.
    if std::env::var("GOOGLE_API_KEY").is_err() {
        println!("Skipping test: GOOGLE_API_KEY not set");
        return;
    }

    // with custom google generative ai provider settings
    let google_ai = GoogleGenerativeAI::builder()
        .api_key(std::env::var("GOOGLE_API_KEY").unwrap())
        .model_name("gemini-1.5-flash")
        .build()
        .expect("Failed to build GoogleGenerativeAI");

    let options = GenerateTextCallOptions::builder()
        .system(Some(
            "Only say hello whatever the user says. \n 
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .prompt(Some("Hello how are you doing?".to_string()))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(google_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_google_ai_messages() {
    dotenv().ok();

    // This test requires a valid Google AI API key to be set in the environment.
    if std::env::var("GOOGLE_API_KEY").is_err() {
        println!("Skipping test: GOOGLE_API_KEY not set");
        return;
    }

    // with custom google generative ai provider settings
    let google_ai = GoogleGenerativeAI::builder()
        .api_key(std::env::var("GOOGLE_API_KEY").unwrap())
        .model_name("gemini-1.5-flash")
        .build()
        .expect("Failed to build GoogleGenerativeAI");

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

    let result = generate_text(google_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("Rohan"));
}

// ===== VertexAI Tests =====

#[tokio::test]
async fn test_generate_text_with_vertex_ai() {
    dotenv().ok();

    // This test requires valid Google Cloud credentials to be set in the environment.
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
        println!("Skipping test: GOOGLE_CLOUD_PROJECT not set");
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

    let vertex_ai = VertexAI::new("test-project", "us-central1", "gemini-1.5-flash");
    let result = generate_text(vertex_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_stream_with_vertex_ai() {
    dotenv().ok();

    // This test requires valid Google Cloud credentials to be set in the environment.
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
        println!("Skipping test: GOOGLE_CLOUD_PROJECT not set");
        return;
    }

    let options = GenerateTextCallOptions::builder()
        .prompt(Some(
            "Respond with exactly the word 'hello' in all lowercase\n 
            5 times each on new lines. Do not include any punctuation,\n 
            prefixes, or suffixes."
                .to_string(),
        ))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let vertex_ai = VertexAI::new("test-project", "us-central1", "gemini-1.5-flash");
    let response = generate_stream(vertex_ai, options).await.unwrap();

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
        assert!(model.starts_with("gemini-1.5-flash"));
    }

    assert!(buf.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_vertex_ai_system_prompt() {
    dotenv().ok();

    // This test requires valid Google Cloud credentials to be set in the environment.
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
        println!("Skipping test: GOOGLE_CLOUD_PROJECT not set");
        return;
    }

    // with custom vertex ai provider settings
    let vertex_ai = VertexAI::builder()
        .project_id("test-project")
        .location("us-central1")
        .model_name("gemini-1.5-flash")
        .build()
        .expect("Failed to build VertexAI");

    let options = GenerateTextCallOptions::builder()
        .system(Some(
            "Only say hello whatever the user says. \n 
            all lowercase no punctuation, prefixes, or suffixes."
                .to_string(),
        ))
        .prompt(Some("Hello how are you doing?".to_string()))
        .build()
        .expect("Failed to build GenerateTextCallOptions");

    let result = generate_text(vertex_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_vertex_ai_messages() {
    dotenv().ok();

    // This test requires valid Google Cloud credentials to be set in the environment.
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
        println!("Skipping test: GOOGLE_CLOUD_PROJECT not set");
        return;
    }

    // with custom vertex ai provider settings
    let vertex_ai = VertexAI::builder()
        .project_id("test-project")
        .location("us-central1")
        .model_name("gemini-1.5-flash")
        .build()
        .expect("Failed to build VertexAI");

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

    let result = generate_text(vertex_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("Rohan"));
}

#[tokio::test]
async fn test_generate_text_with_vertex_ai_messages_and_system_prompt() {
    dotenv().ok();

    // This test requires valid Google Cloud credentials to be set in the environment.
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
        println!("Skipping test: GOOGLE_CLOUD_PROJECT not set");
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

    let vertex_ai = VertexAI::new("test-project", "us-central1", "gemini-1.5-flash");
    let result = generate_text(vertex_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_generate_text_with_vertex_ai_messages_and_inmessage_system_prompt() {
    dotenv().ok();

    // This test requires valid Google Cloud credentials to be set in the environment.
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_err() {
        println!("Skipping test: GOOGLE_CLOUD_PROJECT not set");
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

    let vertex_ai = VertexAI::new("test-project", "us-central1", "gemini-1.5-flash");
    let result = generate_text(vertex_ai, options).await;
    assert!(result.is_ok());

    let text = result.as_ref().expect("Failed to get result").text.trim();
    assert!(text.contains("hello"));
}
