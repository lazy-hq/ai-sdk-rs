//! Helper functions and conversions for the Anthropic provider.

use crate::core::types::{LanguageModelCallOptions, Message};
use crate::providers::anthropic::{AnthropicMessage, AnthropicRequest};

impl From<LanguageModelCallOptions> for AnthropicRequest {
    fn from(options: LanguageModelCallOptions) -> Self {
        let mut messages = Vec::new();
        let mut system = options.system;

        if let Some(msgs) = options.messages {
            for msg in msgs {
                match msg {
                    Message::System(s) => {
                        // If we already have a system prompt from options, prioritize it
                        if system.is_none() {
                            system = Some(s.content);
                        }
                    }
                    Message::User(u) => {
                        messages.push(AnthropicMessage {
                            role: "user".into(),
                            content: u.content,
                        });
                    }
                    Message::Assistant(a) => {
                        messages.push(AnthropicMessage {
                            role: "assistant".into(),
                            content: a.content,
                        });
                    }
                }
            }
        }

        // If we still have no messages, this shouldn't happen with proper validation,
        // but we'll handle it gracefully
        if messages.is_empty() {
            messages.push(AnthropicMessage {
                role: "user".into(),
                content: "Hello".into(),
            });
        }

        AnthropicRequest {
            model: "claude-4-sonnet".into(), // Default, will be overridden
            max_tokens: options.max_tokens.unwrap_or(1024),
            messages,
            system,
            stream: None,
            temperature: options.temperature,
            top_p: options.top_p,
            top_k: options.top_k,
            stop_sequences: options.stop,
        }
    }
}
