//! Helper functions and conversions for the OpenAI provider.

use async_openai::types::responses::{
    CreateResponse, Function, Input, InputContent, InputItem, InputMessage, InputMessageType, Role,
    ToolDefinition,
};

use crate::core::{
    tools::Tool,
    types::{LanguageModelCallOptions, Message},
};

impl From<Tool> for ToolDefinition {
    fn from(value: Tool) -> Self {
        ToolDefinition::Function(Function {
            name: value.name,
            description: Some(value.description),
            strict: true,
            parameters: value.input_schema.to_value(),
        })
    }
}

impl From<LanguageModelCallOptions> for CreateResponse {
    fn from(options: LanguageModelCallOptions) -> Self {
        let mut items: Vec<InputItem> = options
            .messages
            .unwrap_or_default()
            .into_iter()
            .map(|m| InputItem::Message(m.into()))
            .collect();

        // system prompt first since openai likes it at the top
        if let Some(system) = options.system {
            items.insert(
                0,
                InputItem::Message(InputMessage {
                    role: Role::System,
                    kind: InputMessageType::default(),
                    content: InputContent::TextInput(system),
                }),
            );
        }

        let tools: Option<Vec<ToolDefinition>> = options
            .tools
            .map(|t| t.iter().map(|t| ToolDefinition::from(t.clone())).collect());

        CreateResponse {
            input: Input::Items(items),
            temperature: options.temperature.map(|t| t as f32 / 100.0),
            max_output_tokens: options.max_tokens,
            stream: Some(false),
            top_p: options.top_p.map(|t| t as f32 / 100.0),
            tools,
            tool_choice: None,
            ..Default::default() // TODO: add support for other options
        }
    }
}

impl From<Message> for InputMessage {
    fn from(m: Message) -> Self {
        let (role, text) = match m {
            Message::System(s) => (Role::System, s.content),
            Message::User(u) => (Role::User, u.content),
            Message::Assistant(a) => (Role::Assistant, a.content),
        };
        InputMessage {
            role,
            kind: InputMessageType::default(),
            content: InputContent::TextInput(text),
        }
    }
}
