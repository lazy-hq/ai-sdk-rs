//! Helper functions and conversions for the OpenAI provider.

use async_openai::types::ResponseFormatJsonSchema;
use async_openai::types::responses::{
    CreateResponse, Input, InputContent, InputItem, InputMessage, InputMessageType, Role,
    TextConfig, TextResponseFormat,
};
use schemars::Schema;

use crate::core::language_model::LanguageModelOptions;
use crate::core::messages::Message;

impl From<LanguageModelOptions> for CreateResponse {
    fn from(options: LanguageModelOptions) -> Self {
        let mut items: Vec<InputItem> = options
            .messages
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
