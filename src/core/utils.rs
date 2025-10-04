use crate::core::{
    AssistantMessage, Message, ToolCallInfo, ToolOutputInfo,
    language_model::{DEFAULT_TOOL_STEP_COUNT, LanguageModelOptions},
};

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

pub fn handle_tool_call(
    options: &mut LanguageModelOptions,
    tool_infos: Vec<ToolCallInfo>,
    steps: &mut Vec<ToolOutputInfo>,
) {
    let tool_results = &options
        .tools
        .as_ref()
        .map(|tools| tools.execute(tool_infos.clone()));

    if let Some(tool_results) = tool_results {
        let mut tool_output_infos = Vec::new();
        tool_results
            .iter()
            .zip(tool_infos)
            .for_each(|(tool_result, tool_info)| {
                let mut tool_output_info = ToolOutputInfo::new(&tool_info.tool.name);
                tool_output_info.output(serde_json::Value::String(
                    tool_result.clone().unwrap_or("".to_string()),
                ));

                tool_output_info.id(&tool_info.tool.id);
                tool_output_infos.push(tool_output_info.clone());

                // update messages
                let _ = &options
                    .messages
                    .push(Message::Assistant(AssistantMessage::ToolCall(tool_info)));
                let _ = &options
                    .messages
                    .push(Message::Tool(tool_output_info.clone()));

                steps.push(tool_output_info);
            });
    }

    if let Some(step_count) = &options.step_count {
        println!("step({step_count})");
        if *step_count == 1 {
            options.tools = None; // remove the tools
            let _ = &options.messages.push(Message::Developer(
                "Error: Maximum tool calls cycle reached".to_string(),
            ));
        } else {
            options.step_count = Some(step_count - 1);
        }
    } else {
        println!("step({DEFAULT_TOOL_STEP_COUNT})");
        let step_count = DEFAULT_TOOL_STEP_COUNT - 1;
        options.step_count = Some(step_count);
    }
}
