use crate::core::Message;

/// Resolves the message to be used for text generation.
///
/// This function takes a prompt and a list of messages and returns a vector of
/// messages that can be used for LanguageModelCallOptions.
/// if no messages are provided, a default message is created with the prompt and system prompt.
pub fn resolve_message(
    system_prompt: &Option<String>,
    prompt: &Option<String>,
    messages: &[Message],
) -> (String, Vec<Message>) {
    let messages = if messages.is_empty() {
        vec![
            Message::System(system_prompt.to_owned().unwrap_or_default().into()),
            Message::User(prompt.to_owned().unwrap_or_default().into()),
        ]
    } else {
        messages.to_vec()
    };

    let system = system_prompt.to_owned().unwrap_or_else(|| {
        messages
            .iter()
            .find_map(|m| match m {
                Message::System(s) => Some(s.content.to_string()),
                _ => None,
            })
            .unwrap_or_default()
    });

    (system, messages)
}
