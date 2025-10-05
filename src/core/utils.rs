use crate::core::{
    Message, ToolCallInfo, ToolOutputInfo,
    language_model::{
        DEFAULT_TOOL_STEP_COUNT, LanguageModelOptions, LanguageModelResponseContentType,
    },
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

pub async fn handle_tool_call(
    options: &mut LanguageModelOptions,
    tool_infos: Vec<ToolCallInfo>,
    steps: &mut Vec<ToolOutputInfo>,
) {
    if let Some(tools) = &options.tools {
        let tool_results = tools.execute(tool_infos.clone()).await;
        let mut tool_output_infos = Vec::new();
        tool_results
            .into_iter()
            .zip(tool_infos)
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
                let _ = &options.messages.push(Message::Assistant(
                    LanguageModelResponseContentType::ToolCall(tool_info),
                ));
                let _ = &options.messages.push(Message::Tool(tool_output_info));
            });
        *steps = tool_output_infos;
        println!("steps: {:?}", steps);
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
