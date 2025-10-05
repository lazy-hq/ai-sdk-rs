use crate::core::{
    Message, ToolCallInfo, ToolOutputInfo,
    language_model::{
        DEFAULT_TOOL_STEP_COUNT, LanguageModelOptions, LanguageModelResponseContentType,
    },
    messages::AssistantMessage,
};

/// Resolves the message to be used for text generation.
///
/// This function takes a prompt and a list of messages and returns a vector of
/// messages that can be used for LanguageModelCallOptions.
/// if no messages are provided, a default message is created with the prompt and system prompt.
pub fn resolve_message(
    options: &LanguageModelOptions,
    prompt: &Option<String>,
) -> (String, Vec<Message>) {
    let messages = if options.messages.is_empty() {
        vec![
            Message::System(options.system.to_owned().unwrap_or_default().into()),
            Message::User(prompt.to_owned().unwrap_or_default().into()),
        ]
    } else {
        options.messages.to_vec()
    };

    let system = options.system.to_owned().unwrap_or_else(|| {
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

/// Calls the requested tools, updates the messages, and decrements the step count.
pub async fn handle_tool_call(
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
                let _ = &options
                    .messages
                    .push(Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::ToolCall(tool_info),
                    )));
                let _ = &options.messages.push(Message::Tool(tool_output_info));
            });
        *outputs = tool_output_infos;
    }

    if let Some(step_count) = &options.step_count {
        if *step_count == 1 {
            options.tools = None; // remove the tools
            let _ = &options.messages.push(Message::Developer(
                "Error: Maximum tool calls cycle reached".to_string(),
            ));
        } else {
            options.step_count = Some(step_count - 1);
        }
    } else {
        let step_count = DEFAULT_TOOL_STEP_COUNT - 1;
        options.step_count = Some(step_count);
    }
}
