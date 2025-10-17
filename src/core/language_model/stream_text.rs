use crate::core::{
    AssistantMessage, LanguageModelStreamChunkType, Message,
    language_model::{
        LanguageModel, LanguageModelResponseContentType, LanguageModelStream,
        LanguageModelStreamChunk, StopReason, request::LanguageModelRequest,
    },
    messages::TaggedMessage,
    utils::resolve_message,
};
use crate::error::Result;
use futures::StreamExt;
use std::ops::{Deref, DerefMut};

impl<M: LanguageModel> LanguageModelRequest<M> {
    /// Generates Streaming text using a specified language model.
    ///
    /// Generate a text and call tools for a given prompt using a language model.
    /// This function streams the output. If you do not want to stream the output, use `GenerateText` instead.
    ///
    /// Returns an `Error` if the underlying model fails to generate a response.
    pub async fn stream_text(&mut self) -> Result<StreamTextResponse<M>> {
        let (system_prompt, messages) = resolve_message(self);
        self.system = Some(system_prompt);
        self.messages = messages;

        let (tx, stream) = LanguageModelStream::new();
        let _ = tx.send(LanguageModelStreamChunkType::Start);

        loop {
            // Update the current step
            self.current_step_id += 1;

            // Prepare the next step
            if let Some(hook) = self.prepare_step.clone() {
                *self = hook(self.clone()).unwrap_or(self.clone());
            }

            let mut response = self
                .model
                .stream_text(self.clone())
                .await
                .inspect_err(|e| {
                    self.stop_reason = Some(StopReason::Error(e.clone()));
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
                                            self.messages.push(TaggedMessage::new(
                                                self.current_step_id,
                                                assistant_msg,
                                            ));
                                            self.stop_reason = Some(StopReason::Finish);
                                        }
                                        LanguageModelResponseContentType::Reasoning(ref reason) => {
                                            self.messages.push(TaggedMessage::new(
                                                self.current_step_id,
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
                                            self.messages.push(TaggedMessage::new(
                                                self.current_step_id.to_owned(),
                                                Message::Assistant(AssistantMessage::new(
                                                    LanguageModelResponseContentType::ToolCall(
                                                        tool_info.clone(),
                                                    ),
                                                    usage,
                                                )),
                                            ));
                                            self.handle_tool_call(tool_info).await;
                                        }
                                        _ => {}
                                    }

                                    // Finish the step
                                    if let Some(ref hook) = self.on_step_finish {
                                        *self = hook(self.clone()).unwrap_or(self.clone());
                                    }

                                    // Stop If
                                    if let Some(hook) = &self.stop_when.clone()
                                        && hook(self)
                                    {
                                        let _ = tx.send(LanguageModelStreamChunkType::Incomplete(
                                            "Stopped by hook".to_string(),
                                        ));
                                        self.stop_reason = Some(StopReason::Hook);
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
                        self.stop_reason = Some(StopReason::Error(e.clone()));
                        break;
                    }
                }

                match self.stop_reason {
                    None => {}
                    _ => break,
                };
            }

            match self.stop_reason {
                None => {}
                _ => break,
            };
        }

        drop(tx);

        let result = StreamTextResponse {
            stream,
            request: self.clone(),
        };

        Ok(result)
    }
}

// ============================================================================
// Section: response types
// ============================================================================

// Response from a stream call on `StreamText`.
pub struct StreamTextResponse<M: LanguageModel> {
    /// A stream of responses from the language model.
    pub stream: LanguageModelStream,
    request: LanguageModelRequest<M>,
}

impl<M: LanguageModel> StreamTextResponse<M> {
    #[cfg(any(test, feature = "test-access"))]
    pub fn step_ids(&self) -> Vec<usize> {
        self.request.messages.iter().map(|t| t.step_id).collect()
    }
}

impl<M: LanguageModel> Deref for StreamTextResponse<M> {
    type Target = LanguageModelRequest<M>;

    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

impl<M: LanguageModel> DerefMut for StreamTextResponse<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.request
    }
}
