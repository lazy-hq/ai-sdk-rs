use crate::core::{ModelMessage, SystemMessage, UserMessage};

/// Resolves the message to be used for text generation.
///
/// This function takes a prompt and a list of messages and returns a vector of
/// messages that can be used for LanguageModelCallOptions.
/// if no messages are provided, a default message is created with the prompt and system prompt.
pub fn resolve_message(
    system_prompt: Option<String>,
    prompt: Option<String>,
    messages: Option<Vec<ModelMessage>>,
) -> (String, Vec<ModelMessage>) {
    let messages = messages.unwrap_or_else(|| {
        vec![
            ModelMessage::System(SystemMessage::new(
                system_prompt.to_owned().unwrap_or_default(),
            )),
            ModelMessage::User(UserMessage::new(prompt.unwrap_or_default())),
        ]
    });

    let system = system_prompt.unwrap_or_else(|| {
        messages
            .iter()
            .find_map(|m| match m {
                ModelMessage::System(s) => Some(s.content.to_string()),
                _ => None,
            })
            .unwrap_or_default()
    });

    (system, messages)
}
