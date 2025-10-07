use crate::core::{
    Message, ToolCallInfo, ToolOutputInfo,
    language_model::{DEFAULT_TOOL_STEP_COUNT, LanguageModelOptions},
    messages::TaggedMessage,
};

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

/// Calls the requested tools, adds tool ouput message to messages,
/// and decrements the step count. uses the previous step id for tagging
/// the created messages.
pub(crate) async fn handle_tool_call(
    options: &mut LanguageModelOptions,
    inputs: Vec<ToolCallInfo>,
    outputs: &mut Vec<ToolOutputInfo>,
) {
    if let Some(tools) = &options.tools {
        let tool_results = tools.execute(inputs.clone()).await;
        let mut tool_output_infos = Vec::new();
        tool_results
            .into_iter()
            .zip(inputs)
            .for_each(|(tool_result, tool_info)| {
                let mut tool_output_info = ToolOutputInfo::new(&tool_info.tool.name);
                let output = match tool_result {
                    Ok(result) => serde_json::Value::String(result),
                    Err(err) => serde_json::Value::String(format!("Error: {}", err)),
                };
                tool_output_info.output(output);
                tool_output_info.id(&tool_info.tool.id);
                tool_output_infos.push(tool_output_info.clone());

                // update messages
                let _ = &options.messages.push(TaggedMessage::new(
                    options.current_step_id,
                    Message::Tool(tool_output_info),
                ));
            });
        *outputs = tool_output_infos;
    }

    if let Some(step_count) = &options.step_count {
        if *step_count == 1 {
            options.tools = None; // remove the tools
            let _ = &options.messages.push(TaggedMessage::new(
                options.current_step_id,
                Message::Developer("Error: Maximum tool calls cycle reached".to_string()),
            ));
        } else {
            options.step_count = Some(step_count - 1);
        }
    } else {
        let step_count = DEFAULT_TOOL_STEP_COUNT - 1;
        options.step_count = Some(step_count);
    }
}

pub fn sum_options(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
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
