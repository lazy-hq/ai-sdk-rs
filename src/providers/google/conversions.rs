//! Helper functions and conversions for Google providers.

use crate::core::types::{LanguageModelCallOptions, Message};
use crate::providers::google::{
    GoogleGenerationConfig, GoogleMessage, GooglePart, GoogleRequest, GoogleSystemInstruction,
};

impl From<LanguageModelCallOptions> for GoogleRequest {
    fn from(options: LanguageModelCallOptions) -> Self {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        // Handle system prompt
        if let Some(system) = options.system {
            system_instruction = Some(GoogleSystemInstruction {
                parts: vec![GooglePart { text: system }],
            });
        }

        if let Some(msgs) = options.messages {
            for msg in msgs {
                match msg {
                    Message::System(s) => {
                        // If we already have a system prompt from options, prioritize it
                        if system_instruction.is_none() {
                            system_instruction = Some(GoogleSystemInstruction {
                                parts: vec![GooglePart { text: s.content }],
                            });
                        }
                    }
                    Message::User(u) => {
                        contents.push(GoogleMessage {
                            role: "user".into(),
                            parts: vec![GooglePart { text: u.content }],
                        });
                    }
                    Message::Assistant(a) => {
                        contents.push(GoogleMessage {
                            role: "model".into(), // Google uses "model" instead of "assistant"
                            parts: vec![GooglePart { text: a.content }],
                        });
                    }
                }
            }
        }

        // If we still have no messages, this shouldn't happen with proper validation,
        // but we'll handle it gracefully
        if contents.is_empty() {
            contents.push(GoogleMessage {
                role: "user".into(),
                parts: vec![GooglePart {
                    text: "Hello".into(),
                }],
            });
        }

        GoogleRequest {
            contents,
            system_instruction,
            generation_config: options.max_tokens.map(|max_tokens| GoogleGenerationConfig {
                max_output_tokens: Some(max_tokens),
            }),
        }
    }
}
