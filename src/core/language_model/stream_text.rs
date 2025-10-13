use crate::core::{
    AssistantMessage, LanguageModelStreamChunkType, Message,
    language_model::{
        LanguageModel, LanguageModelOptions, LanguageModelResponseContentType, LanguageModelStream,
        LanguageModelStreamChunk, StopReason, request::LanguageModelRequest,
    },
    messages::TaggedMessage,
    utils::resolve_message,
};
use crate::error::Result;
use futures::StreamExt;
use std::ops::Deref;

impl<M: LanguageModel> LanguageModelRequest<M> {
    /// Generates Streaming text using a specified language model.
    ///
    /// Generate a text and call tools for a given prompt using a language model.
    /// This function streams the output. If you do not want to stream the output, use `GenerateText` instead.
    ///
    /// Returns an `Error` if the underlying model fails to generate a response.
    pub async fn stream_text(&mut self) -> Result<StreamTextResponse> {
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

        let (tx, stream) = LanguageModelStream::new();
        let _ = tx.send(LanguageModelStreamChunkType::Start);

        loop {
            // Update the current step
            options.current_step_id += 1;

            // Prepare the next step
            if let Some(hook) = options.prepare_step.clone() {
                hook(&mut options);
            }

            let mut response = self
                .model
                .stream_text(options.clone())
                .await
                .inspect_err(|e| {
                    options.stop_reason = Some(StopReason::Error(e.clone()));
                })?;

            while let Some(ref chunk) = response.next().await {
                match chunk {
                    Ok(chunk) => {
                        for output in chunk {
                            match output {
                                LanguageModelStreamChunk::Done(final_msg) => {
                                    match final_msg.content {
                                        LanguageModelResponseContentType::Text(_) => {
                                            let assistant_msg =
                                                Message::Assistant(AssistantMessage {
                                                    content: final_msg.content.clone(),
                                                    usage: final_msg.usage.clone(),
                                                });
                                            options.messages.push(TaggedMessage::new(
                                                options.current_step_id,
                                                assistant_msg,
                                            ));
                                            options.stop_reason = Some(StopReason::Finish);
                                        }
                                        LanguageModelResponseContentType::Reasoning(ref reason) => {
                                            options.messages.push(TaggedMessage::new(
                                                options.current_step_id,
                                                Message::Assistant(AssistantMessage {
                                                    content:
                                                        LanguageModelResponseContentType::Reasoning(
                                                            reason.clone(),
                                                        ),
                                                    usage: final_msg.usage.clone(),
                                                }),
                                            ))
                                        }
                                        LanguageModelResponseContentType::ToolCall(
                                            ref tool_info,
                                        ) => {
                                            // add tool message
                                            let usage = final_msg.usage.clone();
                                            let _ = &options.messages.push(TaggedMessage::new(
                                                options.current_step_id.to_owned(),
                                                Message::Assistant(AssistantMessage::new(
                                                    LanguageModelResponseContentType::ToolCall(
                                                        tool_info.clone(),
                                                    ),
                                                    usage,
                                                )),
                                            ));
                                            options.handle_tool_call(tool_info).await;
                                        }
                                        _ => {}
                                    }

                                    // Finish the step
                                    if let Some(ref hook) = options.on_step_finish {
                                        hook(&options);
                                    }

                                    // Stop If
                                    if let Some(hook) = &options.stop_when.clone()
                                        && hook(&options)
                                    {
                                        let _ = tx.send(LanguageModelStreamChunkType::Incomplete(
                                            "Stopped by hook".to_string(),
                                        ));
                                        options.stop_reason = Some(StopReason::Hook);
                                        break;
                                    }

                                    let _ = tx
                                        .send(LanguageModelStreamChunkType::End(final_msg.clone()));
                                }
                                LanguageModelStreamChunk::Delta(other) => {
                                    let _ = tx.send(other.clone()); // propagate chunks
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(LanguageModelStreamChunkType::Failed(e.to_string()));
                        options.stop_reason = Some(StopReason::Error(e.clone()));
                        break;
                    }
                }

                match options.stop_reason {
                    None => {}
                    _ => break,
                };
            }

            match options.stop_reason {
                None => {}
                _ => break,
            };
        }

        drop(tx);

        let result = StreamTextResponse { stream, options };

        Ok(result)
    }
}

// ============================================================================
// Section: response types
// ============================================================================

// Response from a stream call on `StreamText`.
pub struct StreamTextResponse {
    /// A stream of responses from the language model.
    pub stream: LanguageModelStream,
    /// The reason the model stopped generating text.
    options: LanguageModelOptions,
}

impl StreamTextResponse {
    #[cfg(any(test, feature = "test-access"))]
    pub fn step_ids(&self) -> Vec<usize> {
        self.options.messages.iter().map(|t| t.step_id).collect()
    }
}

impl Deref for StreamTextResponse {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}
