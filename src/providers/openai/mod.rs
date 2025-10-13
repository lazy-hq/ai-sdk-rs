//! This module provides the OpenAI provider, which implements the `LanguageModel`
//! and `Provider` traits for interacting with the OpenAI API.

pub mod conversions;
pub mod settings;
use std::sync::Arc;

use async_openai::error::OpenAIError;
use async_openai::types::responses::{
    Content, CreateResponse, OutputContent, OutputItem, Response, ResponseEvent, ResponseStream,
};
use async_openai::{Client, config::OpenAIConfig};
use futures::{StreamExt, stream::once};

use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream,
};
use crate::core::messages::AssistantMessage;
use crate::error::ProviderError;
use crate::providers::openai::settings::{OpenAIProviderSettings, OpenAIProviderSettingsBuilder};
use crate::{
    core::{language_model::LanguageModel, provider::Provider, tools::ToolCallInfo},
    error::{Error, Result},
};
use async_trait::async_trait;

/// The OpenAI provider.
#[derive(Debug, Clone)]
pub struct OpenAI {
    client: Client<OpenAIConfig>,
    settings: OpenAIProviderSettings,
}

impl OpenAI {
    /// Creates a new `OpenAI` provider with the given settings.
    pub fn new(model_name: impl Into<String>) -> Self {
        OpenAIProviderSettingsBuilder::default()
            .model_name(model_name.into())
            .build()
            .expect("Failed to build OpenAIProviderSettings")
    }

    /// OpenAI provider setting builder.
    pub fn builder() -> OpenAIProviderSettingsBuilder {
        OpenAIProviderSettings::builder()
    }
}

impl Provider for OpenAI {}

impl ProviderError for OpenAIError {}

#[async_trait]
impl LanguageModel for OpenAI {
    fn name(&self) -> String {
        self.settings.model_name.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let mut request: CreateResponse = options.clone().into();

        request.model = self.settings.model_name.to_string();

        let response: Response = self
            .client
            .responses()
            .create(request)
            .await
            .map_err(|e| Error::ProviderError(Arc::new(e)))?;
        let mut collected: Vec<LanguageModelResponseContentType> = Vec::new();

        for out in response.output {
            match out {
                OutputContent::Message(msg) => {
                    for c in msg.content {
                        if let Content::OutputText(t) = c {
                            collected.push(LanguageModelResponseContentType::new(t.text));
                        }
                    }
                }
                OutputContent::FunctionCall(f) => {
                    let mut tool_info = ToolCallInfo::new(f.name);
                    tool_info.id(f.call_id);
                    tool_info.input(serde_json::from_str(&f.arguments).unwrap());
                    collected.push(LanguageModelResponseContentType::ToolCall(tool_info));
                }
                other => collected.push(LanguageModelResponseContentType::NotSupported(format!(
                    "{other:?}"
                ))),
            }
        }

        Ok(LanguageModelResponse {
            contents: collected,
            usage: response.usage.map(|usage| usage.into()),
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let mut request: CreateResponse = options.into();
        request.model = self.settings.model_name.to_string();
        request.stream = Some(true);

        let openai_stream: ResponseStream = self
            .client
            .responses()
            .create_stream(request)
            .await
            .map_err(|e| Error::ProviderError(Arc::new(e)))?;

        let (first, rest) = openai_stream.into_future().await;

        let openai_stream = if let Some(first) = first {
            Box::pin(once(async move { first }).chain(rest))
        } else {
            rest
        };

        #[derive(Default)]
        struct StreamState {
            completed: bool,
        }

        let stream = openai_stream.scan::<_, Result<Vec<LanguageModelStreamChunk>>, _, _>(
            StreamState::default(),
            |state, evt_res| {
                // If already completed, don't emit anything more
                if state.completed {
                    return futures::future::ready(None);
                };

                futures::future::ready(match evt_res {
                    Ok(ResponseEvent::ResponseCompleted(d)) => {
                        state.completed = true;

                        let mut collected: Vec<LanguageModelResponseContentType> = Vec::new();

                        for out in d.response.output.unwrap_or_default() {
                            match out {
                                // TODO: handle in `ResponseEvent::ResponseFunctionCallArgumentsDone` instead
                                OutputItem::FunctionCall(f) => {
                                    let mut tool_info = ToolCallInfo::new(f.name);
                                    tool_info.id(f.call_id);
                                    tool_info.input(serde_json::from_str(&f.arguments).unwrap());
                                    collected.push(LanguageModelResponseContentType::ToolCall(
                                        tool_info,
                                    ));
                                }
                                other => {
                                    collected.push(LanguageModelResponseContentType::NotSupported(
                                        format!("{other:?}"),
                                    ))
                                }
                            }
                        }

                        Some(Ok(collected
                            .into_iter()
                            .map(|ref c| {
                                LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: c.clone(),
                                    usage: d.response.usage.clone().map(|usage| usage.into()),
                                })
                            })
                            .collect()))
                    }
                    Ok(ResponseEvent::ResponseOutputTextDelta(d)) => {
                        Some(Ok(Vec::from([LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Text(d.delta),
                        )])))
                    }
                    Ok(ResponseEvent::ResponseOutputTextDone(d)) => {
                        state.completed = true;
                        Some(Ok(Vec::from([LanguageModelStreamChunk::Done(
                            AssistantMessage {
                                content: LanguageModelResponseContentType::new(d.text),
                                usage: None, // TODO: try to update usage in `ResponseCompleted`
                            },
                        )])))
                    }
                    Ok(ResponseEvent::ResponseFunctionCallArgumentsDelta(d)) => {
                        Some(Ok(Vec::from([LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::ToolCall(d.delta),
                        )])))
                    }
                    Ok(ResponseEvent::ResponseFunctionCallArgumentsDone(d)) => {
                        // TODO: Function calls should be returned here but `d.name`
                        // is not supported by async-openai. currently it is being
                        // handled by the `ResponseEvent::ResponseCompleted` event but
                        // this is not guaranteed leaving function calls to be supressed.
                        Some(Ok(Vec::from([LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::NotSupported(format!("{d:?}")),
                        )])))
                    }
                    Ok(ResponseEvent::ResponseIncomplete(d)) => {
                        Some(Ok(Vec::from([LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Incomplete({
                                if let Some(reason) = d.response.incomplete_details {
                                    reason.reason
                                } else {
                                    "unknown reason".to_string()
                                }
                            }),
                        )])))
                    }
                    Ok(ResponseEvent::ResponseError(e)) => {
                        state.completed = true;
                        let reason =
                            format!("{}: {}", e.code.unwrap_or(" - ".to_string()), e.message);
                        Some(Ok(Vec::from([LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Failed(reason),
                        )])))
                    }
                    Ok(resp) => Some(Ok(Vec::from([LanguageModelStreamChunk::Delta(
                        LanguageModelStreamChunkType::NotSupported(format!("{resp:?}")),
                    )]))),
                    Err(e) => {
                        state.completed = true;
                        Some(Err(Error::ProviderError(Arc::new(e))))
                    }
                })
            },
        );

        Ok(Box::pin(stream))
    }
}
