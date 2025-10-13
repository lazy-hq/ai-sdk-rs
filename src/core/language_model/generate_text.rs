use crate::error::Result;
use crate::{
    Error,
    core::{
        AssistantMessage, Message,
        language_model::{
            LanguageModel, LanguageModelOptions, LanguageModelResponse,
            LanguageModelResponseContentType, StopReason, request::LanguageModelRequest,
        },
        messages::TaggedMessage,
        utils::resolve_message,
    },
};
use serde::de::DeserializeOwned;
use serde::ser::Error as SerdeError;
use std::ops::Deref;

impl<M: LanguageModel> LanguageModelRequest<M> {
    /// Generates text using a specified language model.
    ///
    /// Generate a text and call tools for a given prompt using a language model.
    /// This function does not stream the output. If you want to stream the output, use `StreamText` instead.
    ///
    /// Returns an `Error` if the underlying model fails to generate a response.
    pub async fn generate_text(&mut self) -> Result<GenerateTextResponse> {
        let (system_prompt, messages) = resolve_message(&self.options, &self.prompt);

        let mut options = LanguageModelOptions {
            system: Some(system_prompt),
            messages,
            schema: self.options.schema.to_owned(),
            stop_sequences: self.options.stop_sequences.to_owned(),
            tools: self.options.tools.to_owned(),
            stop_when: self.options.stop_when.clone(),
            prepare_step: self.options.prepare_step.clone(),
            on_step_finish: self.options.on_step_finish.clone(),
            stop_reason: None,
            ..self.options
        };

        loop {
            // Update the current step
            options.current_step_id += 1;

            // Prepare the next step
            if let Some(hook) = options.prepare_step.clone() {
                hook(&mut options);
            }

            let response: LanguageModelResponse = self
                .model
                .generate_text(options.clone())
                .await
                .inspect_err(|e| {
                options.stop_reason = Some(StopReason::Error(e.clone()));
            })?;

            for output in response.contents.iter() {
                match output {
                    LanguageModelResponseContentType::Text(text) => {
                        let assistant_msg = Message::Assistant(AssistantMessage {
                            content: text.clone().into(),
                            usage: response.usage.clone(),
                        });
                        options
                            .messages
                            .push(TaggedMessage::new(options.current_step_id, assistant_msg));
                    }
                    LanguageModelResponseContentType::Reasoning(reason) => {
                        let assistant_msg = Message::Assistant(AssistantMessage {
                            content: reason.clone().into(),
                            usage: response.usage.clone(),
                        });
                        options
                            .messages
                            .push(TaggedMessage::new(options.current_step_id, assistant_msg));
                    }
                    LanguageModelResponseContentType::ToolCall(tool_info) => {
                        // add tool message
                        let usage = response.usage.clone();
                        let _ = &options.messages.push(TaggedMessage::new(
                            options.current_step_id.to_owned(),
                            Message::Assistant(AssistantMessage::new(
                                LanguageModelResponseContentType::ToolCall(tool_info.clone()),
                                usage,
                            )),
                        ));
                        options.handle_tool_call(tool_info).await;
                    }
                    _ => (),
                }
            }

            // Finish the step
            if let Some(ref hook) = options.on_step_finish {
                hook(&options);
            };

            if response.contents.is_empty() {
                options.stop_reason = Some(StopReason::Error(Error::Other(
                    "Language model returned empty response".to_string(),
                )));
                break;
            }

            // Stop If
            if let Some(hook) = &options.stop_when.clone()
                && hook(&options)
            {
                options.stop_reason = Some(StopReason::Hook);
                break;
            }

            match response.contents.last() {
                Some(LanguageModelResponseContentType::ToolCall(_)) => (),
                _ => {
                    options.stop_reason = Some(StopReason::Finish);
                    break;
                }
            };
        }

        Ok(GenerateTextResponse { options })
    }
}

// ============================================================================
// Section: response types
// ============================================================================

/// Response from a generate call on `GenerateText`.
#[derive(Debug, Clone)]
pub struct GenerateTextResponse {
    /// The options that generated this response
    options: LanguageModelOptions,
}

impl GenerateTextResponse {
    pub fn into_schema<T: DeserializeOwned>(&self) -> std::result::Result<T, serde_json::Error> {
        if let Some(text) = &self.text() {
            serde_json::from_str(text)
        } else {
            Err(serde_json::Error::custom("No text response found"))
        }
    }

    #[cfg(any(test, feature = "test-access"))]
    pub fn step_ids(&self) -> Vec<usize> {
        self.options.messages.iter().map(|t| t.step_id).collect()
    }
}

impl Deref for GenerateTextResponse {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        AssistantMessage, ToolCallInfo, ToolResultInfo,
        language_model::{LanguageModelResponseContentType, Usage},
        messages::TaggedMessage,
    };

    #[test]
    fn test_generate_text_response_step() {
        let options = LanguageModelOptions {
            messages: vec![
                TaggedMessage::new(0, Message::System("System".to_string().into())),
                TaggedMessage::new(0, Message::User("User".to_string().into())),
                TaggedMessage::new(
                    1,
                    Message::Assistant(AssistantMessage {
                        content: LanguageModelResponseContentType::Text("Assistant".to_string()),
                        usage: None,
                    }),
                ),
            ],
            ..Default::default()
        };
        let response = GenerateTextResponse { options };

        let step0 = response.step(0).unwrap();
        assert_eq!(step0.step_id, 0);
        assert_eq!(step0.messages.len(), 2);

        let step1 = response.step(1).unwrap();
        assert_eq!(step1.step_id, 1);
        assert_eq!(step1.messages.len(), 1);

        assert!(response.step(2).is_none());
    }

    #[test]
    fn test_generate_text_response_final_step() {
        let options = LanguageModelOptions {
            messages: vec![
                TaggedMessage::new(0, Message::System("System".to_string().into())),
                TaggedMessage::new(1, Message::User("User".to_string().into())),
                TaggedMessage::new(
                    2,
                    Message::Assistant(AssistantMessage {
                        content: LanguageModelResponseContentType::Text("Assistant".to_string()),
                        usage: None,
                    }),
                ),
            ],
            ..Default::default()
        };
        let response = GenerateTextResponse { options };

        let final_step = response.last_step().unwrap();
        assert_eq!(final_step.step_id, 2);
        assert_eq!(final_step.messages.len(), 1);
    }

    #[test]
    fn test_generate_text_response_steps() {
        let options = LanguageModelOptions {
            messages: vec![
                TaggedMessage::new(0, Message::System("System".to_string().into())),
                TaggedMessage::new(0, Message::User("User".to_string().into())),
                TaggedMessage::new(
                    1,
                    Message::Assistant(AssistantMessage {
                        content: LanguageModelResponseContentType::Text("Assistant1".to_string()),
                        usage: None,
                    }),
                ),
                TaggedMessage::new(
                    2,
                    Message::Assistant(AssistantMessage {
                        content: LanguageModelResponseContentType::Text("Assistant2".to_string()),
                        usage: None,
                    }),
                ),
            ],
            ..Default::default()
        };
        let response = GenerateTextResponse { options };

        let steps = response.steps();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].step_id, 0);
        assert_eq!(steps[0].messages.len(), 2);
        assert_eq!(steps[1].step_id, 1);
        assert_eq!(steps[1].messages.len(), 1);
        assert_eq!(steps[2].step_id, 2);
        assert_eq!(steps[2].messages.len(), 1);
    }

    #[test]
    fn test_generate_text_response_usage() {
        let options = LanguageModelOptions {
            messages: vec![
                TaggedMessage::new(0, Message::System("System".to_string().into())),
                TaggedMessage::new(
                    1,
                    Message::Assistant(AssistantMessage {
                        content: LanguageModelResponseContentType::Text("Assistant1".to_string()),
                        usage: Some(Usage {
                            input_tokens: Some(10),
                            output_tokens: Some(5),
                            total_tokens: Some(15),
                            reasoning_tokens: Some(2),
                            cached_tokens: Some(1),
                        }),
                    }),
                ),
                TaggedMessage::new(
                    2,
                    Message::Assistant(AssistantMessage {
                        content: LanguageModelResponseContentType::Text("Assistant2".to_string()),
                        usage: Some(Usage {
                            input_tokens: Some(5),
                            output_tokens: Some(3),
                            total_tokens: Some(8),
                            reasoning_tokens: Some(1),
                            cached_tokens: Some(0),
                        }),
                    }),
                ),
            ],
            ..Default::default()
        };
        let response = GenerateTextResponse { options };

        let total_usage = response.usage();
        assert_eq!(total_usage.input_tokens, Some(15));
        assert_eq!(total_usage.output_tokens, Some(8));
        assert_eq!(total_usage.total_tokens, Some(23));
        assert_eq!(total_usage.reasoning_tokens, Some(3));
        assert_eq!(total_usage.cached_tokens, Some(1));
    }

    fn create_tool_call_message(step_id: usize, tool_name: &str) -> TaggedMessage {
        TaggedMessage::new(
            step_id,
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::ToolCall(ToolCallInfo::new(tool_name)),
                usage: None,
            }),
        )
    }

    fn create_tool_result_message(step_id: usize, tool_name: &str) -> TaggedMessage {
        TaggedMessage::new(step_id, Message::Tool(ToolResultInfo::new(tool_name)))
    }

    fn create_text_assistant_message(step_id: usize, text: &str) -> TaggedMessage {
        TaggedMessage::new(
            step_id,
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text(text.to_string()),
                usage: None,
            }),
        )
    }

    fn create_response_with_messages(messages: Vec<TaggedMessage>) -> GenerateTextResponse {
        let options = LanguageModelOptions {
            messages,
            ..Default::default()
        };
        GenerateTextResponse { options }
    }

    // Tests for GenerateTextResponse tool_calls()
    #[test]
    fn test_generate_text_response_tool_calls_empty_messages() {
        let response = create_response_with_messages(vec![]);
        assert_eq!(response.tool_calls(), None);
    }

    #[test]
    fn test_generate_text_response_tool_calls_only_non_assistant_messages() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            TaggedMessage::new(0, Message::User("User".to_string().into())),
            create_tool_result_message(0, "tool1"),
        ];
        let response = create_response_with_messages(messages);
        assert_eq!(response.tool_calls(), None);
    }

    #[test]
    fn test_generate_text_response_tool_calls_single_assistant_with_tool_call() {
        let messages = vec![create_tool_call_message(0, "test_tool")];
        let response = create_response_with_messages(messages);
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool.name, "test_tool");
    }

    #[test]
    fn test_generate_text_response_tool_calls_multiple_assistant_with_tool_calls_different_steps() {
        let messages = vec![
            create_tool_call_message(0, "tool1"),
            create_tool_call_message(1, "tool2"),
            create_tool_call_message(2, "tool3"),
        ];
        let response = create_response_with_messages(messages);
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].tool.name, "tool1");
        assert_eq!(calls[1].tool.name, "tool2");
        assert_eq!(calls[2].tool.name, "tool3");
    }

    #[test]
    fn test_generate_text_response_tool_calls_assistant_without_tool_call() {
        let messages = vec![create_text_assistant_message(0, "Hello")];
        let response = create_response_with_messages(messages);
        assert_eq!(response.tool_calls(), None);
    }

    #[test]
    fn test_generate_text_response_tool_calls_mixed_message_types_multiple_steps() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            TaggedMessage::new(0, Message::User("User".to_string().into())),
            create_tool_call_message(1, "test_tool"),
            create_tool_result_message(1, "other_tool"),
            create_tool_call_message(2, "another_tool"),
        ];
        let response = create_response_with_messages(messages);
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool.name, "test_tool");
        assert_eq!(calls[1].tool.name, "another_tool");
    }

    #[test]
    fn test_generate_text_response_tool_calls_duplicate_tool_calls() {
        let messages = vec![
            create_tool_call_message(0, "tool1"),
            create_tool_call_message(1, "tool1"), // Same name
            create_tool_call_message(2, "tool1"), // Same name again
        ];
        let response = create_response_with_messages(messages);
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].tool.name, "tool1");
        assert_eq!(calls[1].tool.name, "tool1");
        assert_eq!(calls[2].tool.name, "tool1");
    }

    #[test]
    fn test_generate_text_response_tool_calls_from_specific_steps_only() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            create_tool_call_message(1, "tool_from_step1"),
            TaggedMessage::new(1, Message::User("User".to_string().into())),
            create_tool_call_message(2, "tool_from_step2"),
            create_tool_result_message(2, "result_from_step2"),
            create_tool_call_message(3, "tool_from_step3"),
        ];
        let response = create_response_with_messages(messages);
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].tool.name, "tool_from_step1");
        assert_eq!(calls[1].tool.name, "tool_from_step2");
        assert_eq!(calls[2].tool.name, "tool_from_step3");
    }

    // Tests for GenerateTextResponse tool_results()
    #[test]
    fn test_generate_text_response_tool_results_empty_messages() {
        let response = create_response_with_messages(vec![]);
        assert!(response.tool_results().is_none());
    }

    #[test]
    fn test_generate_text_response_tool_results_only_non_tool_messages() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            TaggedMessage::new(0, Message::User("User".to_string().into())),
            create_text_assistant_message(0, "Assistant"),
        ];
        let response = create_response_with_messages(messages);
        assert!(response.tool_results().is_none());
    }

    #[test]
    fn test_generate_text_response_tool_results_single_tool_message() {
        let messages = vec![create_tool_result_message(0, "test_tool")];
        let response = create_response_with_messages(messages);
        let results = response.tool_results().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool.name, "test_tool");
    }

    #[test]
    fn test_generate_text_response_tool_results_multiple_tool_messages_different_steps() {
        let messages = vec![
            create_tool_result_message(0, "tool1"),
            create_tool_result_message(1, "tool2"),
            create_tool_result_message(2, "tool3"),
        ];
        let response = create_response_with_messages(messages);
        let results = response.tool_results().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].tool.name, "tool1");
        assert_eq!(results[1].tool.name, "tool2");
        assert_eq!(results[2].tool.name, "tool3");
    }

    #[test]
    fn test_generate_text_response_tool_results_mixed_message_types() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            TaggedMessage::new(0, Message::User("User".to_string().into())),
            create_tool_result_message(1, "test_tool"),
            create_text_assistant_message(1, "Assistant"),
            create_tool_result_message(2, "another_tool"),
        ];
        let response = create_response_with_messages(messages);
        let results = response.tool_results().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool.name, "test_tool");
        assert_eq!(results[1].tool.name, "another_tool");
    }

    #[test]
    fn test_generate_text_response_tool_results_no_tool_messages_but_others_present() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            TaggedMessage::new(0, Message::User("User".to_string().into())),
            create_text_assistant_message(0, "Assistant"),
        ];
        let response = create_response_with_messages(messages);
        assert!(response.tool_results().is_none());
    }

    #[test]
    fn test_generate_text_response_tool_results_duplicate_tool_entries() {
        let messages = vec![
            create_tool_result_message(0, "tool1"),
            create_tool_result_message(1, "tool1"), // Same name
            create_tool_result_message(2, "tool1"), // Same name again
        ];
        let response = create_response_with_messages(messages);
        let results = response.tool_results().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].tool.name, "tool1");
        assert_eq!(results[1].tool.name, "tool1");
        assert_eq!(results[2].tool.name, "tool1");
    }

    #[test]
    fn test_generate_text_response_tool_results_preserving_original_message_order() {
        let messages = vec![
            TaggedMessage::new(0, Message::System("System".to_string().into())),
            create_tool_result_message(1, "tool1"),
            TaggedMessage::new(1, Message::User("User".to_string().into())),
            create_tool_result_message(2, "tool2"),
            create_text_assistant_message(2, "Assistant"),
            create_tool_result_message(3, "tool3"),
        ];
        let response = create_response_with_messages(messages);
        let results = response.tool_results().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].tool.name, "tool1");
        assert_eq!(results[1].tool.name, "tool2");
        assert_eq!(results[2].tool.name, "tool3");
    }

    #[test]
    fn test_generate_text_response_tool_results_large_number_of_messages() {
        let mut messages = Vec::new();
        // Add 1000 messages with tool results interspersed
        for i in 0..1000 {
            messages.push(create_tool_result_message(0, &format!("tool{}", i)));
            if i % 100 == 0 {
                messages.push(TaggedMessage::new(
                    0,
                    Message::User(format!("User message {}", i).into()),
                ));
            }
        }
        let response = create_response_with_messages(messages);
        let results = response.tool_results().unwrap();
        assert_eq!(results.len(), 1000);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.tool.name, format!("tool{}", i));
        }
    }
}
