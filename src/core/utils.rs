use crate::core::{Message, language_model::LanguageModelOptions, messages::TaggedMessage};

/// Resolves the message to be used for text generation.
///
/// This function takes a prompt and a list of messages and returns a vector of
/// messages that can be used for LanguageModelCallOptions.
/// if no messages are provided, a default message is created with the prompt and system prompt.
pub(crate) fn resolve_message(
    options: &LanguageModelOptions,
    prompt: &Option<String>,
) -> (String, Vec<TaggedMessage>) {
    let messages = if options.messages.is_empty() {
        vec![
            TaggedMessage::initial_step_msg(Message::System(
                options.system.to_owned().unwrap_or_default().into(),
            )),
            TaggedMessage::initial_step_msg(Message::User(
                prompt.to_owned().unwrap_or_default().into(),
            )),
        ]
    } else {
        options.messages.to_vec()
    };

    let system = options.system.to_owned().unwrap_or_else(|| {
        messages
            .iter()
            .find_map(|m| match m.message {
                Message::System(ref s) => Some(s.content.to_string()),
                _ => None,
            })
            .unwrap_or_default()
    });

    (system, messages)
}

pub fn sum_options(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        _ => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_options_both_some() {
        assert_eq!(sum_options(Some(1), Some(2)), Some(3));
        assert_eq!(sum_options(Some(0), Some(0)), Some(0));
        assert_eq!(sum_options(Some(10), Some(20)), Some(30));
    }

    #[test]
    fn test_sum_options_first_some_second_none() {
        assert_eq!(sum_options(Some(5), None), Some(5));
        assert_eq!(sum_options(Some(0), None), Some(0));
    }

    #[test]
    fn test_sum_options_first_none_second_some() {
        assert_eq!(sum_options(None, Some(7)), Some(7));
        assert_eq!(sum_options(None, Some(0)), Some(0));
    }

    #[test]
    fn test_sum_options_both_none() {
        assert_eq!(sum_options(None, None), None);
    }
}
