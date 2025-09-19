//! Helper functions and conversions for the OpenAI provider.

use async_openai::types::responses::{
    CreateResponse, Input, InputContent, InputItem, InputMessage, InputMessageType, Role,
};

use crate::core::types::{LanguageModelCallOptions, Message};

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

        CreateResponse {
            input: Input::Items(items),
            temperature: options.temperature,
            max_output_tokens: options.max_tokens,
            stream: Some(false),
            top_p: options.top_p,
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
