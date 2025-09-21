//! Integration tests for the OpenAI provider.

use aisdk::{
    core::{
        LanguageModelRequest, Message,
        tools::{Tool, ToolExecute},
    },
    providers::openai::OpenAI,
};
use aisdk_macros::tool;
use dotenv::dotenv;
use futures::StreamExt;

#[tool]
/// Hello
fn example_tool(a: String) -> Tool {
    Ok("".to_string())
}

#[tool]
// Fetch The User Name
fn get_user_name() {
    Ok("john_doe".to_string())
}

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

//#[tokio::test]
//async fn test_generate_text_with_messages_and_inmessage_system_prompt() {
//    dotenv().ok();
//
//    // This test requires a valid OpenAI API key to be set in the environment.
//    if std::env::var("OPENAI_API_KEY").is_err() {
//        println!("Skipping test: OPENAI_API_KEY not set");
//        return;
//    }
//
//    let messages = Message::builder()
//        .system("Only say hello whatever the user says. \n all lowercase no punctuation, prefixes, or suffixes.")
//        .user("Whatsup?, Surafel is here")
//        .assistant("How could I help you?")
//        .user("Could you tell my name?")
//        .build();
//
//    let options = GenerateTextCallOptions::builder()
//        .messages(Some(messages))
//        .build()
//        .expect("Failed to build GenerateTextCallOptions");
//
//    let result = generate_text(OpenAI::new("gpt-4o"), options).await;
//    assert!(result.is_ok());
//
//    let text = result.as_ref().expect("Failed to get result").text.trim();
//    assert!(text.contains("hello"));
//}
//
//#[tokio::test]
//async fn test_generate_text_builder_with_both_prompt_and_messages() {
//    dotenv().ok();
//
//    // the builder should fail
//    let options = GenerateTextCallOptions::builder()
//        .prompt(Some(
//            "Only say hello whatever the user says. \n
//            all lowercase no punctuation, prefixes, or suffixes."
//                .to_string(),
//        ))
//        .messages(Some(vec![Message::User(
//            "Whatsup?, Surafel is here".into(),
//        )]))
//        .build();
//
//    //println!("opt reslut 1: {:?}", options);
//    assert!(options.is_err());
//    match options.unwrap_err() {
//        Error::InvalidInput(msg) => {
//            assert!(msg.contains("Cannot set both prompt and messages"));
//        }
//        _ => {
//            panic!("Expected InvalidInput error");
//        }
//    }
//}
//
//#[tokio::test]
//async fn test_generate_text_builder_with_no_prompt_and_messages() {
//    dotenv().ok();
//
//    // the builder should fail
//    let options = GenerateTextCallOptions::builder()
//        .system(Some("You are a helpful assistant.".to_string()))
//        .build();
//
//    //println!("opt reslut 1: {:?}", options);
//    assert!(options.is_err());
//
//    match options.unwrap_err() {
//        Error::InvalidInput(msg) => {
//            assert!(msg.contains("Messages or prompt must be set"));
//        }
//        _ => {
//            panic!("Expected InvalidInput error");
//        }
//    }
//}

#[tokio::test]
async fn test_tool_call_for_generate_text() {
    //dotenv().ok();
    //
    //// This test requires a valid OpenAI API key to be set in the environment.
    //if std::env::var("OPENAI_API_KEY").is_err() {
    //    println!("Skipping test: OPENAI_API_KEY not set");
    //    return;
    //}
    //
    //let options = GenerateTextCallOptions::builder()
    //    .prompt(Some(
    //        "Respond with exactly username of the user in all lowercase.\n
    //            Do not include any punctuation, prefixes, or suffixes. if you\n
    //            can't find the user return 'unknown'"
    //            .to_string(),
    //    ))
    //    .with_tool(get_user_name())
    //    .build()
    //    .expect("Failed to build GenerateTextCallOptions");
    //
    //let result = generate_text(OpenAI::new("gpt-4o"), options).await;
    //println!("result: {:#?}", result);
    //assert!(result.is_ok());
    //
    ////let text = result.as_ref().expect("Failed to get result").text.trim();
    ////assert!(text.contains("hello"));
}

#[tokio::test]
async fn test_tool_call_for_generate_stream() {
    //todo!();
    //dotenv().ok();
    //
    //// This test requires a valid OpenAI API key to be set in the environment.
    //if std::env::var("OPENAI_API_KEY").is_err() {
    //    println!("Skipping test: OPENAI_API_KEY not set");
    //    return;
    //}
    //
    //let options = GenerateTextCallOptions::builder()
    //    .prompt(Some(
    //        "Respond with exactly the name of the user in all lowercase\n
    //        100 times each on new lines. Do not include any punctuation,\n
    //        prefixes, or suffixes. remove any underscore and capitalize each\n
    //        each part of the name"
    //            .to_string(),
    //    ))
    //    .with_tool(get_user_name())
    //    .build()
    //    .expect("Failed to build GenerateTextCallOptions");
    //
    //let response = generate_stream(OpenAI::new("gpt-4o"), options)
    //    .await
    //    .unwrap();
    //
    //let mut stream = response.stream;
    //
    //let mut buf = String::new();
    //while let Some(chunk) = stream.next().await {
    //    if let Ok(lang_resp) = chunk
    //        && !lang_resp.text.is_empty()
    //    {
    //        buf.push_str(&lang_resp.text);
    //    }
    //}
    //
    //println!("response: {}", buf);
}
