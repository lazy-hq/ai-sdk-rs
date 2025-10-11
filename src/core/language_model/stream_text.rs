use crate::core::{
    AssistantMessage, LanguageModelStreamChunkType, Message,
    language_model::{
        LanguageModel, LanguageModelOptions, LanguageModelResponseContentType,
        LanguageModelResponseMethods, MpmcStream, request::LanguageModelRequest,
    },
    messages::TaggedMessage,
    utils::{handle_tool_call, resolve_message},
};
use crate::error::Result;
use futures::StreamExt;

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
            ..self.options
        };

        options.current_step_id += 1;
        if let Some(hook) = options.prepare_step.clone() {
            hook(&mut options);
        }

        let mut response = self.model.stream_text(options.to_owned()).await?;

        let (tx, stream) = MpmcStream::new();
        let _ = tx.send(LanguageModelStreamChunkType::Start);
        while let Some(chunk) = response.stream.next().await {
            match chunk {
                Ok(LanguageModelStreamChunkType::End(assistant_msg)) => {
                    let usage = assistant_msg.usage.clone();
                    match &assistant_msg.content {
                        LanguageModelResponseContentType::Text(text) => {
                            let assistant_msg = Message::Assistant(AssistantMessage {
                                content: text.clone().into(),
                                usage: assistant_msg.usage.clone(),
                            });
                            options.messages.push(TaggedMessage {
                                step_id: options.current_step_id,
                                message: assistant_msg,
                            });

                            if let Some(ref hook) = options.on_step_finish {
                                hook(&options);
                            }
                        }
                        LanguageModelResponseContentType::ToolCall(tool_info) => {
                            // add tool message
                            let assistant_msg = Message::Assistant(AssistantMessage::new(
                                LanguageModelResponseContentType::ToolCall(tool_info.clone()),
                                usage.clone(),
                            ));
                            let _ = &options
                                .messages
                                .push(TaggedMessage::new(options.current_step_id, assistant_msg));

                            let mut current_tool_steps = Vec::new();
                            handle_tool_call(
                                &mut options,
                                vec![tool_info.clone()],
                                &mut current_tool_steps,
                            )
                            .await;

                            if let Some(ref hook) = options.on_step_finish {
                                hook(&options);
                            }

                            if let Some(ref hook) = options.stop_when
                                && hook(&options)
                            {
                                let _ = tx.send(LanguageModelStreamChunkType::Incomplete(
                                    "Stopped by hook".to_string(),
                                ));
                                break;
                            }

                            // update anything options
                            options.current_step_id += 1;
                            self.options = options.clone();

                            // call the next step
                            let next_res = Box::pin(self.stream_text()).await;

                            match next_res {
                                Ok(StreamTextResponse { mut stream, .. }) => {
                                    while let Some(chunk) = stream.next().await {
                                        let _ = tx.send(chunk);
                                    }
                                }
                                Err(e) => {
                                    // TODO: is this the right error to return. maybe Incomplete is
                                    // correct
                                    let _ = tx
                                        .send(LanguageModelStreamChunkType::Failed(e.to_string()));
                                    break;
                                }
                            };
                        }
                    };
                }
                Ok(other) => {
                    let _ = tx.send(other); // propagate
                }
                Err(e) => {
                    let _ = tx.send(LanguageModelStreamChunkType::Failed(e.to_string()));
                    break;
                }
            }
        }

        drop(tx);

        let result = StreamTextResponse {
            stream,
            model: response.model,
            options,
        };

        Ok(result)
    }
}

// ============================================================================
// Section: response types
// ============================================================================

// TODO: add standard response fields
// Response from a stream call on `StreamText`.
pub struct StreamTextResponse {
    /// A stream of responses from the language model.
    pub stream: MpmcStream,
    /// The model that generated the response.
    pub model: Option<String>,
    /// The reason the model stopped generating text.
    pub options: LanguageModelOptions,
}

impl StreamTextResponse {
    #[cfg(any(test, feature = "test-access"))]
    pub fn step_ids(&self) -> Vec<usize> {
        self.options.messages.iter().map(|t| t.step_id).collect()
    }
}

impl LanguageModelResponseMethods for StreamTextResponse {
    fn options(&self) -> &LanguageModelOptions {
        &self.options
    }
}
