//! Helper functions and conversions for the OpenAI provider.

use crate::core::language_model::LanguageModelOptions;
use crate::core::messages::Message;
use crate::core::tools::Tool;
use async_openai::types::responses::{
    CreateResponse, Function, Input, InputContent, InputItem, InputMessage, InputMessageType, Role,
    ToolDefinition,
};
use serde_json::Value;

impl From<Tool> for ToolDefinition {
    fn from(value: Tool) -> Self {
        let mut params = value.input_schema.to_value();

        // open ai requires 'additionalProperties' to be false
        params["additionalProperties"] = Value::Bool(false);

        // open ai requires 'properties' to be an object
        let properties = params.get("properties");
        if let Some(Value::Object(_)) = properties {
        } else {
            params["properties"] = Value::Object(serde_json::Map::new());
        }

        ToolDefinition::Function(Function {
            name: value.name,
            description: Some(value.description),
            strict: true,
            parameters: params,
        })
    }
}

impl From<LanguageModelOptions> for CreateResponse {
    fn from(options: LanguageModelOptions) -> Self {
        let mut items: Vec<InputItem> = options.messages.into_iter().map(|m| m.into()).collect();

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
            max_output_tokens: options.max_output_tokens,
            stream: Some(false),
            top_p: options.top_p.map(|t| t as f32 / 100.0),
            tools,
            tool_choice: None,
            ..Default::default() // TODO: add support for other options
        }
    }
}

impl From<Message> for InputItem {
    fn from(m: Message) -> Self {
        let mut _common_input_msg = InputMessage {
            role: Role::System,
            kind: InputMessageType::default(),
            content: InputContent::TextInput(Default::default()),
        };
        match m {
            Message::Tool(ref tool_info) => {
                // manually adding the types because async_openai didn't implement it.
                let mut custom_msg = Value::Object(serde_json::Map::new());
                custom_msg["type"] = Value::String("function_call_output".to_string());
                custom_msg["call_id"] = Value::String(tool_info.tool.id.clone());
                custom_msg["output"] = tool_info.output.clone();
                InputItem::Custom(custom_msg)
            }
            Message::Assistant(_assistant_msg) => {
                todo!()
            }
            _ => unimplemented!(),
        };
        //panic!();
        //
        if let Message::Tool(tool_info) = m {
            // manually adding the types because async_openai didn't implement it.
            let mut custom_msg = Value::Object(serde_json::Map::new());
            custom_msg["type"] = Value::String("function_call_output".to_string());
            custom_msg["call_id"] = Value::String(tool_info.tool.id);
            custom_msg["output"] = tool_info.output;
            InputItem::Custom(custom_msg)
        } else {
            let (role, text) = match m {
                Message::System(s) => (Role::System, s.content),
                Message::User(u) => (Role::User, u.content),
                Message::Assistant(_a) => todo!(),
                _ => unreachable!(
                    "Tool is handling separately in an if let clause. this is the else."
                ),
            };

            let input_msg = InputMessage {
                role,
                kind: InputMessageType::default(),
                content: InputContent::TextInput(text),
            };

            InputItem::Message(input_msg)
        }
    }
}
