//! Helper functions and conversions for the OpenAI provider.

use crate::core::language_model::{LanguageModelOptions, LanguageModelResponseContentType, Usage};
use crate::core::messages::Message;
use crate::core::tools::Tool;
use async_openai::types::ResponseFormatJsonSchema;
use async_openai::types::responses::{
    CreateResponse, Function, Input, InputContent, InputItem, InputMessage, InputMessageType, Role,
    TextConfig, TextResponseFormat, ToolDefinition, Usage as OpenAIUsage,
};
use schemars::Schema;
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
        let mut items: Vec<InputItem> = options
            .messages
            .into_iter()
            .filter_map(|m| m.message.into())
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

        let tools: Option<Vec<ToolDefinition>> = options.tools.map(|t| {
            t.tools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .map(|t| ToolDefinition::from(t.clone()))
                .collect()
        });

        CreateResponse {
            input: Input::Items(items),
            text: Some(TextConfig {
                format: options
                    .schema
                    .map(from_schema_to_response_format)
                    .map(TextResponseFormat::JsonSchema)
                    .unwrap_or(TextResponseFormat::Text),
            }),
            temperature: options.temperature.map(|t| t as f32 / 100.0),
            max_output_tokens: options.max_output_tokens,
            stream: Some(false),
            top_p: options.top_p.map(|t| t as f32 / 100.0),
            tools,
            ..Default::default()
        }
    }
}

impl From<Message> for Option<InputItem> {
    fn from(m: Message) -> Self {
        let mut text_inp = InputMessage {
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
                custom_msg["output"] = tool_info
                    .output
                    .clone()
                    .unwrap_or_else(|e| Value::String(e.to_string()));
                Some(InputItem::Custom(custom_msg))
            }
            Message::Assistant(ref assistant_msg) => match assistant_msg.content {
                LanguageModelResponseContentType::Text(ref msg) => {
                    text_inp.role = Role::Assistant;
                    text_inp.content = InputContent::TextInput(msg.to_owned());
                    Some(InputItem::Message(text_inp))
                }
                LanguageModelResponseContentType::ToolCall(ref tool_info) => {
                    let mut custom_msg = Value::Object(serde_json::Map::new());
                    custom_msg["arguments"] = Value::String(tool_info.input.to_string().clone());
                    custom_msg["call_id"] = Value::String(tool_info.tool.id.clone());
                    custom_msg["name"] = Value::String(tool_info.tool.name.clone());
                    custom_msg["type"] = Value::String("function_call".to_string());
                    Some(InputItem::Custom(custom_msg))
                }
                _ => None,
            },
            Message::User(u) => {
                text_inp.role = Role::User;
                text_inp.content = InputContent::TextInput(u.content);
                Some(InputItem::Message(text_inp))
            }
            Message::System(s) => {
                text_inp.role = Role::System;
                text_inp.content = InputContent::TextInput(s.content);
                Some(InputItem::Message(text_inp))
            }
            Message::Developer(d) => {
                text_inp.role = Role::Developer;
                text_inp.content = InputContent::TextInput(d);
                Some(InputItem::Message(text_inp))
            }
        }
    }
}

impl From<OpenAIUsage> for Usage {
    fn from(value: OpenAIUsage) -> Self {
        Self {
            input_tokens: Some(value.input_tokens as usize),
            output_tokens: Some(value.output_tokens as usize),
            total_tokens: Some(value.total_tokens as usize),
            cached_tokens: Some(value.input_tokens_details.cached_tokens.unwrap_or(0) as usize),
            reasoning_tokens: Some(
                value.output_tokens_details.reasoning_tokens.unwrap_or(0) as usize
            ),
        }
    }
}

fn from_schema_to_response_format(schema: Schema) -> ResponseFormatJsonSchema {
    let json = serde_json::to_value(schema).expect("Failed to serialize schema");
    ResponseFormatJsonSchema {
        name: json
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Response Schema")
            .to_owned(),
        description: json
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        schema: Some(json),
        strict: Some(false),
    }
}
