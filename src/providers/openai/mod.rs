//! This module provides the OpenAI provider, which implements the `LanguageModel`
//! and `Provider` traits for interacting with the OpenAI API.

pub mod conversions;
pub mod settings;
use async_openai::types::responses::{
    Content, CreateResponse, OutputContent, OutputItem, Response, ResponseEvent,
    ResponseOutputItemDone, ResponseStream,
};
use async_openai::{Client, config::OpenAIConfig};
use futures::{StreamExt, stream::once};

use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunkType, LanguageModelStreamResponse,
};
use crate::providers::openai::settings::{OpenAIProviderSettings, OpenAIProviderSettingsBuilder};
use crate::{
    core::{language_model::LanguageModel, provider::Provider, tools::ToolCallInfo},
    error::Result,
};
use async_trait::async_trait;
use serde::Serialize;

/// The OpenAI provider.
#[derive(Debug, Serialize)]
pub struct OpenAI {
    #[serde(skip)]
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

#[async_trait]
impl LanguageModel for OpenAI {
    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let mut request: CreateResponse = options.clone().into();

        request.model = self.settings.model_name.to_string();

        let response: Response = self.client.responses().create(request).await?;
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
                other => {
                    todo!("Unhandled output: {other:?}");
                }
            }
        }

        Ok(LanguageModelResponse {
            model: Some(response.model.to_string()),
            content: collected.first().unwrap().clone(),
            stop_reason: None,
        })
    }

    async fn stream_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelStreamResponse> {
        let mut request: CreateResponse = options.into();
        request.model = self.settings.model_name.to_string();
        request.stream = Some(true);

        let openai_stream: ResponseStream = self.client.responses().create_stream(request).await?;

        let (first, rest) = openai_stream.into_future().await;

        // get the model name from the first response
        let model = match &first {
            Some(Ok(ResponseEvent::ResponseCreated(r))) => Some(
                r.response
                    .model
                    .as_ref()
                    .unwrap_or(&self.settings.model_name)
                    .to_string(),
            ),
            _ => None,
        };

        let openai_stream = if let Some(first) = first {
            Box::pin(once(async move { first }).chain(rest))
        } else {
            rest
        };

        #[derive(Default)]
        struct StreamState {
            completed: bool,
        }

        let stream = openai_stream.scan::<_, Result<LanguageModelStreamChunkType>, _, _>(
            StreamState::default(),
            |state, evt_res| {
                // If already completed, don't emit anything more
                if state.completed {
                    return futures::future::ready(None);
                };

                futures::future::ready(match evt_res {
                    Ok(ResponseEvent::ResponseOutputTextDelta(d)) => {
                        Some(Ok(LanguageModelStreamChunkType::Text(d.delta)))
                    }
                    Ok(ResponseEvent::ResponseCompleted(_)) => {
                        state.completed = true;
                        Some(Ok(LanguageModelStreamChunkType::End))
                    }
                    Ok(ResponseEvent::ResponseFailed(f)) => {
                        state.completed = true;
                        let reason = f
                            .response
                            .error
                            .as_ref()
                            .map(|e| format!("{}: {}", e.code, e.message))
                            .unwrap_or_else(|| "unknown failure".to_string());
                        Some(Ok(LanguageModelStreamChunkType::Failed(reason)))
                    }
                    Ok(ResponseEvent::ResponseOutputItemDone(ResponseOutputItemDone {
                        item: OutputItem::FunctionCall(tool_call),
                        ..
                    })) => {
                        state.completed = true;
                        let mut tool_info = ToolCallInfo::new(tool_call.name);
                        tool_info.id(tool_call.call_id);
                        tool_info.input(serde_json::from_str(&tool_call.arguments).unwrap());

                        Some(Ok(LanguageModelStreamChunkType::ToolCall(tool_info)))
                    }
                    Ok(resp) => Some(Ok(LanguageModelStreamChunkType::NotImplemented(format!(
                        "{resp:?}"
                    )))),
                    Err(e) => {
                        state.completed = true;
                        Some(Err(e.into()))
                    }
                })
            },
        );

        Ok(LanguageModelStreamResponse {
            stream: Box::pin(stream),
            model,
            stop_reason: None,
        })
    }
}
