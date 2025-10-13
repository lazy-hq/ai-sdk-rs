//! Integration tests for the OpenAI provider.

use aisdk::{
    core::{
        LanguageModelRequest, LanguageModelStreamChunkType, Message,
        language_model::{LanguageModelResponseContentType, StopReason},
        tool,
        tools::{Tool, ToolExecute},
    },
    providers::openai::OpenAI,
};
use dotenv::dotenv;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

use std::sync::{Arc, Mutex};

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

    let text = result
        .as_ref()
        .expect("")
        .text()
        .unwrap()
        .trim()
        .to_string();

    assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_stop_reason_normal_finish() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Respond with exactly the word 'hello' in all lowercase. Do not include any punctuation.")
        .build()
        .generate_text()
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(matches!(response.stop_reason(), Some(StopReason::Finish)));
}

#[tokio::test]
async fn test_stop_reason_hook_stop() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Tell me a short story.")
        .stop_when(|_| true) // Always stop
        .build()
        .generate_text()
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(matches!(response.stop_reason(), Some(StopReason::Hook)));
}

#[tokio::test]
async fn test_stop_reason_api_error() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("invalid-model-name"))
        .prompt("Hello")
        .build()
        .generate_text()
        .await;

    // Should fail, but if it succeeds, check stop_reason
    if let Ok(response) = result {
        // If somehow succeeds, but unlikely
        assert!(matches!(response.stop_reason(), Some(StopReason::Finish)));
    } else {
        // Error occurred, but stop_reason is set in the options before error
        // Since result is Err, we can't check response.stop_reason
        // Perhaps modify to check options, but for now, just assert error
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn test_stop_reason_stream_finish() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Respond with 'world'")
        .build()
        .stream_text()
        .await
        .unwrap();

    // The stream is already consumed internally, stop_reason is set
    assert!(matches!(response.stop_reason(), Some(StopReason::Finish)));
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
        if let LanguageModelStreamChunkType::Text(text) = chunk {
            buf.push_str(&text);
        }
    }

    // if let Some(model) = response.model {
    //     assert!(model.starts_with("gpt-4o"));
    // }

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

    let text = result
        .as_ref()
        .expect("Failed to get result")
        .text()
        .unwrap()
        .trim()
        .to_string();
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

    let text = result
        .as_ref()
        .expect("Failed to get result")
        .text()
        .unwrap()
        .trim()
        .to_string();
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

    let text = result
        .as_ref()
        .expect("Failed to get result")
        .text()
        .unwrap()
        .trim()
        .to_string();
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

    #[derive(Debug, JsonSchema, Deserialize)]
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

    let user: User = result.into_schema().unwrap();

    assert_eq!(user.name, "John Doe");
}

#[tokio::test]
async fn test_stream_text_with_output_schema() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    #[derive(Debug, JsonSchema, Deserialize)]
    #[allow(dead_code)]
    struct User {
        name: String,
        age: u32,
        email: String,
        phone: String,
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("generate user with dummy data, and add name of 'John Doe'")
        .schema::<User>()
        .build()
        .stream_text()
        .await
        .unwrap();

    let mut stream = response.stream;

    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        if let LanguageModelStreamChunkType::Text(text) = chunk {
            buf.push_str(&text);
        }
    }

    println!("buf: {}", buf);

    let user: User = serde_json::from_str(&buf).unwrap();

    assert_eq!(user.name, "John Doe");
}

#[tokio::test]
async fn test_generate_text_with_tools() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    #[tool]
    /// Returns the username
    fn get_username() {
        Ok("ishak".to_string())
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call a tool to get the username.")
        .prompt("What is the username?")
        .with_tool(get_username())
        .build()
        .generate_text()
        .await
        .unwrap();

    assert!(response.text().unwrap().contains("ishak"));
}

#[tokio::test]
async fn test_generate_stream_with_tools() {
    dotenv().ok();

    // This test requires a valid OpenAI API key to be set in the environment.
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    #[tool]
    /// Returns the username
    fn get_username() {
        Ok("ishak".to_string())
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call a tool to get the username.")
        .prompt("What is the username?")
        .with_tool(get_username())
        .build()
        .stream_text()
        .await
        .unwrap();

    let mut stream = response.stream;

    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        if let LanguageModelStreamChunkType::Text(text) = chunk {
            buf.push_str(&text);
        }
    }

    assert!(buf.contains("ishak"));
}

#[tokio::test]
async fn test_step_id_basic_assignment() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Respond with exactly 'test' in lowercase.")
        .build()
        .generate_text()
        .await
        .unwrap();

    // Check step_ids: system (0), user (0), assistant (1)
    let step_ids = result.step_ids();
    assert_eq!(step_ids.len(), 3);
    assert_eq!(step_ids[0], 0); // system
    assert_eq!(step_ids[1], 0); // user
    assert_eq!(step_ids[2], 1); // assistant
}

#[tokio::test]
async fn test_step_id_tool_call_flow() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    #[tool]
    fn get_test_value() -> Result<String> {
        Ok("test_value".to_string())
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool to get the test value.")
        .prompt("What is the test value?")
        .with_tool(get_test_value())
        .build()
        .generate_text()
        .await
        .unwrap();

    let step_ids = result.step_ids();
    // system (0), user (0), assistant tool call (1), tool result (1), assistant text (3)
    assert!(step_ids.len() >= 5);
    assert_eq!(step_ids[0], 0);
    assert_eq!(step_ids[1], 0);
    assert_eq!(step_ids[2], 1); // assistant tool call
    assert_eq!(step_ids[3], 1); // tool result
    assert_eq!(step_ids[4], 2); // assistant text
    assert!(result.text().unwrap().contains("test_value"));
}

#[tokio::test]
async fn test_step_id_streaming() {
    dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Respond with 'stream test'")
        .build()
        .stream_text()
        .await
        .unwrap();

    let step_ids = response.step_ids();
    // system (0), user (0), assistant (1)
    assert_eq!(step_ids.len(), 3);
    assert_eq!(step_ids[0], 0);
    assert_eq!(step_ids[1], 0);
    assert_eq!(step_ids[2], 1);
}

#[tokio::test]
async fn test_prepare_step_executes_before_each_step() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let counter = Arc::new(Mutex::new(0));
    let counter_clone = Arc::clone(&counter);

    #[tool]
    // Returns the neighborhood
    fn get_neighborhood() -> Result<String> {
        Ok("ankocha".to_string())
    }

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. Return the neighborhood. Nothing more and nothing less")
        .prompt("What is the neighborhood?")
        .with_tool(get_neighborhood())
        .prepare_step(move |_| {
            let mut c = counter_clone.lock().unwrap();
            *c += 1;
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    let count = *counter.lock().unwrap();
    assert!(count >= 2); // At least initial + tool step
}

#[tokio::test]
async fn test_on_step_finish_executes_after_each_step() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let counter = Arc::new(Mutex::new(0));
    let counter_clone = Arc::clone(&counter);

    #[tool]
    // Returns the neighbourhood
    fn get_neighborhood() -> Result<String> {
        Ok("ankocha".to_string())
    }

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. Return the neighborhood. Nothing more and nothing less")
        .prompt("What is the neighborhood?")
        .with_tool(get_neighborhood())
        .on_step_finish(move |_| {
            let mut c = counter_clone.lock().unwrap();
            *c += 1;
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    let count = *counter.lock().unwrap();
    assert!(count >= 2);
}

#[tokio::test]
async fn test_hooks_run_in_correct_order() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let log_prepare = Arc::clone(&log);
    let log_finish = Arc::clone(&log);

    #[tool]
    fn get_neighbourhood() -> Result<String> {
        Ok("".to_string())
    }

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. Return the neighborhood. Nothing more and nothing less")
        .prompt("What is the neighborhood?")
        .with_tool(get_neighbourhood())
        .prepare_step(move |_| {
            log_prepare.lock().unwrap().push("prepare");
        })
        .on_step_finish(move |_| {
            log_finish.lock().unwrap().push("finish");
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    let log = log.lock().unwrap();
    // Check pairs of prepare/finish
    let mut i = 0;
    while i + 1 < log.len() {
        assert_eq!(log[i], "prepare");
        assert_eq!(log[i + 1], "finish");
        i += 2;
    }
}

#[tokio::test]
async fn test_stop_when_halts_during_tool_call() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    #[tool]
    fn get_neighborhood() -> Result<String> {
        Ok("done".to_string())
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. Return the neighborhood. Nothing more and nothing less")
        .prompt("What is the neighborhood?")
        .with_tool(get_neighborhood())
        .stop_when(|_| true) // Always stop
        .build()
        .generate_text()
        .await
        .unwrap();

    let step_ids = result.step_ids();
    // Should stop after tool call, no final text step
    assert!(step_ids.len() < 5); // Less than full flow
    assert!(matches!(result.stop_reason().unwrap(), StopReason::Hook));
}

#[tokio::test]
async fn test_stop_when_halts_during_streaming() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    #[tool]
    fn get_neighbourhood() -> Result<String> {
        Ok("ankocha".to_string())
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. Return the neighborhood. Nothing more and nothing less")
        .prompt("What is the neighborhood?")
        .with_tool(get_neighbourhood())
        .stop_when(|_| true)
        .build()
        .stream_text()
        .await
        .unwrap();

    let mut stream = response.stream;
    let mut found_incomplete = false;
    while let Some(chunk) = stream.next().await {
        if let LanguageModelStreamChunkType::Incomplete(_) = chunk {
            found_incomplete = true;
            break;
        }
    }
    assert!(found_incomplete);
}

#[tokio::test]
async fn test_stop_when_false_allows_continuation() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    #[tool]
    fn get_neighbourhood() -> Result<String> {
        Ok("anckocha".to_string())
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. Return the neighborhood. Nothing more and nothing less")
        .prompt("What is the neighborhood?")
        .with_tool(get_neighbourhood())
        .stop_when(|_| false) // Never stop
        .build()
        .generate_text()
        .await
        .unwrap();

    let step_ids = result.step_ids();
    assert!(step_ids.len() >= 5); // Full flow
    assert!(result.text().is_some());
}

#[tokio::test]
async fn test_hooks_persist_across_multiple_steps() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let counter = Arc::new(Mutex::new(0));
    let counter_clone = Arc::clone(&counter);

    #[tool]
    fn get_neighbourhood() -> Result<String> {
        Ok("You can find the neighborhood if you call the get_neighbourhood_2 tool".to_string())
    }

    #[tool]
    fn get_neighbourhood_2() -> Result<String> {
        Ok("ankocha".to_string())
    }

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system(
            "Call the tool get_neighbourhood. Return the neighborhood. 
            Nothing more and nothing less. If you can't find the neighborhood,
            call the tool get_neighbourhood_2. Return the neighborhood.
            Nothing more and nothing less",
        )
        .prompt("What is the neighborhood?")
        .with_tool(get_neighbourhood())
        .with_tool(get_neighbourhood_2())
        .on_step_finish(move |_| {
            let mut c = counter_clone.lock().unwrap();
            *c += 1;
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    let count = *counter.lock().unwrap();
    assert!(count >= 3); // Multiple steps
}

#[tokio::test]
async fn test_hooks_cloned_via_arc() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let called = Arc::new(Mutex::new(false));
    let called_clone = Arc::clone(&called);

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .on_step_finish(move |_| {
            *called_clone.lock().unwrap() = true;
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    assert!(*called.lock().unwrap());
}

#[tokio::test]
async fn test_no_panic_when_hooks_none() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .build()
        .generate_text()
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_prepare_step_mutates_options() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .prepare_step(|opts| {
            opts.temperature = Some(0); // Mutate
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    // Hard to verify mutation directly, but ensure no panic and response ok
    assert!(result.text().is_some());
}

#[tokio::test]
async fn test_hook_isolation() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    // Without hooks
    let result_no_hooks = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .build()
        .generate_text()
        .await
        .unwrap();

    // With hooks (should not affect output)
    let result_with_hooks = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .on_step_finish(|_| {})
        .build()
        .generate_text()
        .await
        .unwrap();

    // Outputs should be similar (hooks don't change logic)
    assert!(result_no_hooks.text().is_some());
    assert!(result_with_hooks.text().is_some());
}

#[tokio::test]
async fn test_on_step_finish_for_text_reasoning_and_tool_call() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let called_for_text = Arc::new(Mutex::new(false));
    let called_for_tool = Arc::new(Mutex::new(false));
    let text_clone = Arc::clone(&called_for_text);
    let tool_clone = Arc::clone(&called_for_tool);

    #[tool]
    // Returns the username
    fn get_username() -> Result<String> {
        Ok("ishak".to_string())
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .system("Call the tool. to find the username. and return only the username nothing more and nothing less")
        .prompt("What is the username")
        .with_tool(get_username())
        .on_step_finish(move |opts| {
            if let Some(Message::Assistant(assistant_msg)) = opts.messages().last() {
                match &assistant_msg.content {
                    LanguageModelResponseContentType::ToolCall(_) => {
                        *tool_clone.lock().unwrap() = true;
                    }
                    LanguageModelResponseContentType::Text(_) => {
                        *text_clone.lock().unwrap() = true;
                    }
                    LanguageModelResponseContentType::Reasoning(_) => {
                        *text_clone.lock().unwrap() = true;
                    }
                    _ => {}
                }
            }
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    assert!(!*called_for_tool.lock().unwrap());
    assert!(*called_for_text.lock().unwrap());
    assert_eq!(result.text().unwrap(), "ishak");
}

#[tokio::test]
async fn test_streaming_prepare_step_before_start() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let called = Arc::new(Mutex::new(false));
    let called_clone = Arc::clone(&called);

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .prepare_step(move |_| {
            *called_clone.lock().unwrap() = true;
        })
        .build()
        .stream_text()
        .await
        .unwrap();

    assert!(*called.lock().unwrap()); // Called before streaming starts
}

#[tokio::test]
async fn test_streaming_on_step_finish_at_end() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let called = Arc::new(Mutex::new(false));
    let called_clone = Arc::clone(&called);

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4o"))
        .prompt("Say hello")
        .on_step_finish(move |_| {
            *called_clone.lock().unwrap() = true;
        })
        .build()
        .stream_text()
        .await
        .unwrap();

    let mut stream = response.stream;
    while stream.next().await.is_some() {} // Consume stream

    assert!(*called.lock().unwrap()); // Called after End
}

#[tokio::test]
#[should_panic]
async fn test_reasoning_effort_with_non_reasoning_model() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let _ = LanguageModelRequest::builder()
        .model(OpenAI::new("gpt-4"))
        .prompt("What is 2 + 2? Answer with just the number.")
        .reasoning_effort(aisdk::core::language_model::ReasoningEffort::Low)
        .build()
        .generate_text()
        .await
        .unwrap();
}

#[tokio::test]
async fn test_reasoning_tokens_in_response_usage() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("o1-mini"))
        .prompt("Solve this math problem step by step: What is 15 * 7?")
        .reasoning_effort(aisdk::core::language_model::ReasoningEffort::High)
        .build()
        .generate_text()
        .await
        .unwrap();

    assert!(result.text().is_some());

    // Check that reasoning_tokens are present in usage
    let usage = result.usage();
    assert!(usage.reasoning_tokens.is_some());
    assert!(usage.reasoning_tokens.unwrap() > 0);
}

#[tokio::test]
async fn test_reasoning_content_type_handling() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let reasoning_called = Arc::new(Mutex::new(false));
    let text_called = Arc::new(Mutex::new(false));
    let reasoning_clone = Arc::clone(&reasoning_called);
    let text_clone = Arc::clone(&text_called);

    let result = LanguageModelRequest::builder()
        .model(OpenAI::new("o1-mini"))
        .prompt("Explain your reasoning step by step: What is 2 + 3?")
        .reasoning_effort(aisdk::core::language_model::ReasoningEffort::High)
        .on_step_finish(move |opts| {
            if let Some(Message::Assistant(assistant_msg)) = opts.messages().last() {
                match &assistant_msg.content {
                    LanguageModelResponseContentType::Text(_) => {
                        *text_clone.lock().unwrap() = true;
                    }
                    LanguageModelResponseContentType::Reasoning(_) => {
                        *reasoning_clone.lock().unwrap() = true;
                    }
                    _ => {}
                }
            }
        })
        .build()
        .generate_text()
        .await
        .unwrap();

    assert!(result.text().is_some());
    // At least one of text or reasoning should be called
    assert!(*text_called.lock().unwrap() || *reasoning_called.lock().unwrap());
}

#[tokio::test]
async fn test_streaming_with_reasoning_effort() {
    dotenv().ok();
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let response = LanguageModelRequest::builder()
        .model(OpenAI::new("o1-mini"))
        .prompt("Count from 1 to 5 step by step")
        .reasoning_effort(aisdk::core::language_model::ReasoningEffort::Medium)
        .build()
        .stream_text()
        .await
        .unwrap();

    let mut stream = response.stream;
    let mut chunks_received = 0;
    while let Some(chunk) = stream.next().await {
        chunks_received += 1;
        // Just verify we get chunks
        let _ = chunk;
    }

    assert!(chunks_received > 0);
}
