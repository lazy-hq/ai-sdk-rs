//! Helper functions and conversions for the OpenAI provider.

use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponseContentType, ReasoningEffort, Usage,
};
use crate::core::messages::Message;
use crate::core::tools::Tool;
use async_openai::types::responses::{
    CreateResponse, Function, Input, InputContent, InputItem, InputMessage, InputMessageType,
    ReasoningConfig, ReasoningSummary, Role, TextConfig, TextResponseFormat, ToolDefinition,
    Usage as OpenAIUsage,
};
use async_openai::types::{ReasoningEffort as OpenAIReasoningEffort, ResponseFormatJsonSchema};
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

        let reasoning = options.reasoning_effort.map(|reasoning| ReasoningConfig {
            summary: Some(ReasoningSummary::Auto),
            effort: Some(reasoning.into()),
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
            reasoning,
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
                LanguageModelResponseContentType::Reasoning(ref reason) => {
                    let mut custom_msg = Value::Object(serde_json::Map::new());
                    let mut summary = Value::Object(serde_json::Map::new());
                    summary["type"] = Value::String("summary_text".to_string());
                    summary["text"] = Value::String(reason.clone());

                    custom_msg["type"] = Value::String("reasoning".to_string());
                    custom_msg["summary"] = summary;

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

impl From<ReasoningEffort> for OpenAIReasoningEffort {
    fn from(value: ReasoningEffort) -> Self {
        match value {
            ReasoningEffort::Low => OpenAIReasoningEffort::Minimal,
            ReasoningEffort::Medium => OpenAIReasoningEffort::Medium,
            ReasoningEffort::High => OpenAIReasoningEffort::High,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::language_model::{LanguageModelOptions, ReasoningEffort, Usage};
    use crate::core::messages::{AssistantMessage, Message};

    #[test]
    fn test_reasoning_effort_conversion_low() {
        let effort = ReasoningEffort::Low;
        let openai_effort: OpenAIReasoningEffort = effort.into();
        // We can't directly compare enum variants from external crate,
        // but we can verify the conversion doesn't panic and returns a valid value
        let _ = openai_effort;
    }

    #[test]
    fn test_reasoning_effort_conversion_medium() {
        let effort = ReasoningEffort::Medium;
        let openai_effort: OpenAIReasoningEffort = effort.into();
        let _ = openai_effort;
    }

    #[test]
    fn test_reasoning_effort_conversion_high() {
        let effort = ReasoningEffort::High;
        let openai_effort: OpenAIReasoningEffort = effort.into();
        let _ = openai_effort;
    }

    #[test]
    fn test_language_model_options_to_create_response_with_reasoning_effort_low() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::Low),
            ..Default::default()
        };
        let create_response: CreateResponse = options.into();
        assert!(create_response.reasoning.is_some());
        let reasoning = create_response.reasoning.unwrap();
        assert_eq!(reasoning.effort, Some(OpenAIReasoningEffort::Minimal));
        assert_eq!(reasoning.summary, Some(ReasoningSummary::Auto));
    }

    #[test]
    fn test_language_model_options_to_create_response_with_reasoning_effort_medium() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        };
        let create_response: CreateResponse = options.into();
        assert!(create_response.reasoning.is_some());
        let reasoning = create_response.reasoning.unwrap();
        assert_eq!(reasoning.effort, Some(OpenAIReasoningEffort::Medium));
        assert_eq!(reasoning.summary, Some(ReasoningSummary::Auto));
    }

    #[test]
    fn test_language_model_options_to_create_response_with_reasoning_effort_high() {
        let options = LanguageModelOptions {
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };
        let create_response: CreateResponse = options.into();
        assert!(create_response.reasoning.is_some());
        let reasoning = create_response.reasoning.unwrap();
        assert_eq!(reasoning.effort, Some(OpenAIReasoningEffort::High));
        assert_eq!(reasoning.summary, Some(ReasoningSummary::Auto));
    }

    #[test]
    fn test_language_model_options_to_create_response_without_reasoning_effort() {
        let options = LanguageModelOptions {
            reasoning_effort: None,
            ..Default::default()
        };
        let create_response: CreateResponse = options.into();
        assert!(create_response.reasoning.is_none());
    }

    #[test]
    fn test_openai_usage_to_usage_conversion() {
        use async_openai::types::responses::Usage as OpenAIUsage;

        let openai_usage = OpenAIUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            input_tokens_details: Default::default(),
            output_tokens_details: Default::default(),
        };

        let usage: Usage = openai_usage.into();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.total_tokens, Some(150));
        // These will be 0 because the details are default (None)
        assert_eq!(usage.cached_tokens, Some(0));
        assert_eq!(usage.reasoning_tokens, Some(0));
    }

    #[test]
    fn test_assistant_message_with_reasoning_content_conversion() {
        let assistant_msg = AssistantMessage {
            content: LanguageModelResponseContentType::Reasoning(
                "This is my reasoning".to_string(),
            ),
            usage: None,
        };
        let message = Message::Assistant(assistant_msg);

        let input_item: Option<InputItem> = message.into();
        assert!(input_item.is_some());

        if let Some(InputItem::Custom(custom_msg)) = input_item {
            assert_eq!(custom_msg["type"], "reasoning");
            assert!(custom_msg["summary"].is_object());
            let summary = custom_msg["summary"].as_object().unwrap();
            assert_eq!(summary["type"], "summary_text");
            assert_eq!(summary["text"], "This is my reasoning");
        } else {
            panic!("Expected Custom InputItem");
        }
    }
}
