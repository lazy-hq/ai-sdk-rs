//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use crate::core::messages::{AssistantMessage, TaggedMessage};
use crate::core::tools::{Tool, ToolList};
use crate::core::utils::{handle_tool_call, resolve_message};
use crate::core::{Message, ToolCallInfo, ToolOutputInfo};
use crate::error::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use futures::Stream;
use futures::StreamExt;
use schemars::{JsonSchema, Schema, schema_for};
use serde::de::DeserializeOwned;
use serde::ser::Error as SerdeError;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::mpsc::{self, Receiver, Sender};
use std::task::{Context, Poll};

// ============================================================================
// Section: constants
// ============================================================================
pub const DEFAULT_TOOL_STEP_COUNT: usize = 3;

// ============================================================================
// Section: traits
// ============================================================================

/// The core trait abstracting the capabilities of a language model.
///
/// This trait is the foundation for all text-based AI interactions.
/// Implementors of `LanguageModel` provide the necessary logic to connect to
/// a specific model endpoint and perform operations. The trait is designed to
/// be extensible to support various functionalities, such as single-shot
/// generation and streaming responses.
#[async_trait]
pub trait LanguageModel: Send + Sync + std::fmt::Debug {
    /// Performs a single, non-streaming text generation request.
    ///
    /// This method sends a prompt to the model and returns the entire response at once.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse>;

    /// Performs a streaming text generation request.
    ///
    /// This method sends a prompt to the model and returns a stream of responses.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn stream_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelStreamResponse>;
}

// ============================================================================
// Section: structs and builders
// ============================================================================

/// Options for a language model request.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Builder)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct LanguageModelOptions {
    /// System prompt to be used for the request.
    pub system: Option<String>,

    /// The messages to generate text from.
    /// At least User Message is required.
    pub(crate) messages: Vec<TaggedMessage>,

    /// Output format schema.
    pub schema: Option<Schema>,

    /// The seed (integer) to use for random sampling. If set and supported
    /// by the model, calls will generate deterministic results.
    pub seed: Option<u32>,

    /// Randomness.
    pub temperature: Option<u32>,

    /// Nucleus sampling.
    pub top_p: Option<u32>,

    /// Top-k sampling.
    pub top_k: Option<u32>,

    /// Maximum number of retries.
    pub max_retries: Option<u32>,

    /// Maxoutput tokens.
    pub max_output_tokens: Option<u32>,

    /// Stop sequences.
    /// If set, the model will stop generating text when one of the stop sequences is generated.
    pub stop_sequences: Option<Vec<String>>,

    /// Presence penalty setting. It affects the likelihood of the model to
    /// repeat information that is already in the prompt.
    pub presence_penalty: Option<f32>,

    /// Frequency penalty setting. It affects the likelihood of the model
    /// to repeatedly use the same words or phrases.
    pub frequency_penalty: Option<f32>,

    /// Number tool call cycles to make
    pub step_count: Option<usize>,

    /// List of tools to use.
    pub tools: Option<ToolList>,

    /// Used to track message steps
    pub(crate) current_step_id: usize,
    // Additional provider-specific options. They are passed through
    // to the provider from the AI SDK and enable provider-specific functionality.
    //TODO: add support for provider options
    //pub provider_options: <HashMap<String, <HashMap<String, JsonValue>>>>,
}

impl LanguageModelOptions {
    pub fn builder() -> LanguageModelOptionsBuilder {
        LanguageModelOptionsBuilder::default()
    }

    pub fn messages(&self) -> Vec<Message> {
        self.messages.iter().map(|m| m.message.clone()).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguageModelResponseContentType {
    Text(String),
    ToolCall(ToolCallInfo),
}

impl Default for LanguageModelResponseContentType {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for LanguageModelResponseContentType {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl LanguageModelResponseContentType {
    pub fn new(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
}

// TODO: constract a standard response type
/// Response from a language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelResponse {
    /// The generated text.
    pub content: LanguageModelResponseContentType,

    /// The model that generated the response.
    pub model: Option<String>,

    /// The reason the model stopped generating text.
    /// Might not necessaryly be an error. for errors, handle
    /// the `Result::Err` variant associated with this type.
    pub stop_reason: Option<String>,

    /// Usage information
    pub usage: Option<Usage>,
}

impl LanguageModelResponse {
    /// Creates a new response with the generated text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content: LanguageModelResponseContentType::new(text.into()),
            model: None,
            stop_reason: None,
            usage: None,
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub enum LanguageModelStreamChunkType {
    /// The model has started generating text.
    #[default]
    Start,
    /// Text chunk
    Text(String),
    /// Tool call argument chunk
    ToolCall(String),
    /// The model has stopped generating text successfully.
    End(AssistantMessage),
    /// The model has failed to generate text.
    Failed(String), // TODO: add a type to accomodate provider and aisdk errors
    /// The model finsished generating text with incomplete response.
    Incomplete(String), // TODO: replace with StopReason
    /// Return this for unimplemented features for a specific model.
    NotSupported(String),
}

/// Stream of responses from mapped to a common interface.
pub type LanguageModelStream =
    Pin<Box<dyn Stream<Item = Result<LanguageModelStreamChunkType>> + Send>>;

/// A response from a streaming language model provider.
pub struct LanguageModelStreamResponse {
    /// A stream of responses from the language model.
    pub stream: LanguageModelStream,
    /// The model that generated the response.
    pub model: Option<String>,
}

// Struct wrapper for MPMC channel to act as a stream
pub struct MpmcStream {
    receiver: Receiver<LanguageModelStreamChunkType>,
}

impl MpmcStream {
    // Creates a new MpmcStream with an associated Sender
    pub fn new() -> (Sender<LanguageModelStreamChunkType>, MpmcStream) {
        let (tx, rx) = mpsc::channel();
        (tx, MpmcStream { receiver: rx })
    }
}

impl Stream for MpmcStream {
    type Item = LanguageModelStreamChunkType;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.try_recv() {
            Ok(item) => Poll::Ready(Some(item)),
            Err(mpsc::TryRecvError::Empty) => Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => Poll::Ready(None),
        }
    }
}

/// Options for text generation requests such as `generate_text` and `stream_text`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelRequest<M: LanguageModel> {
    /// The Language Model to use.
    pub model: M,

    /// The prompt to generate text from.
    /// Only one of prompt or messages should be set.
    pub prompt: Option<String>,

    /// Language model call options for the request
    options: LanguageModelOptions,
}

impl<M: LanguageModel> LanguageModelRequest<M> {
    pub fn builder() -> LanguageModelRequestBuilder<M> {
        LanguageModelRequestBuilder::default()
    }
}

impl<M: LanguageModel> Deref for LanguageModelRequest<M> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel> DerefMut for LanguageModelRequest<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

// State for GenerateOptionsBuilder
// Following the type State builder pattern

/// Initial state for setting the model
/// returns SystemStage
pub struct ModelStage {}

/// Secondary state for including system prompt or not
/// returns ConversationStage
pub struct SystemStage {}

/// Third state for conversation, Message or Prompt
/// returns OptionsStage
pub struct ConversationStage {}

/// Final State for setting Options and config
/// returns builder.build
pub struct OptionsStage {}

pub struct LanguageModelRequestBuilder<M: LanguageModel, State = ModelStage> {
    model: Option<M>,
    prompt: Option<String>,
    options: LanguageModelOptions,
    state: std::marker::PhantomData<State>,
}

impl<M: LanguageModel, State> Deref for LanguageModelRequestBuilder<M, State> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel, State> DerefMut for LanguageModelRequestBuilder<M, State> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

impl<M: LanguageModel> LanguageModelRequestBuilder<M> {
    fn default() -> Self {
        LanguageModelRequestBuilder {
            model: None,
            prompt: None,
            options: LanguageModelOptions::default(),
            state: std::marker::PhantomData,
        }
    }
}

/// ModelStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, ModelStage> {
    pub fn model(self, model: M) -> LanguageModelRequestBuilder<M, SystemStage> {
        LanguageModelRequestBuilder {
            model: Some(model),
            prompt: self.prompt,
            options: self.options,
            state: std::marker::PhantomData,
        }
    }
}

/// SystemStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, SystemStage> {
    pub fn system(
        self,
        system: impl Into<String>,
    ) -> LanguageModelRequestBuilder<M, ConversationStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                system: Some(system.into()),
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }

    pub fn prompt(self, prompt: impl Into<String>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: Some(prompt.into()),
            options: self.options,
            state: std::marker::PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                messages: messages.into_iter().map(|msg| msg.into()).collect(),
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }
}

/// ConversationStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, ConversationStage> {
    pub fn prompt(self, prompt: impl Into<String>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: Some(prompt.into()),
            options: self.options,
            state: std::marker::PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                messages: messages.into_iter().map(|msg| msg.into()).collect(),
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }
}
/// OptionsStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, OptionsStage> {
    pub fn schema<T: JsonSchema>(mut self) -> Self {
        self.schema = Some(schema_for!(T));
        self
    }
    pub fn seed(mut self, seed: impl Into<u32>) -> Self {
        self.seed = Some(seed.into());
        self
    }

    pub fn temperature(mut self, temperature: impl Into<u32>) -> Self {
        self.temperature = Some(temperature.into());
        self
    }

    pub fn top_p(mut self, top_p: impl Into<u32>) -> Self {
        self.top_p = Some(top_p.into());
        self
    }

    pub fn top_k(mut self, top_k: impl Into<u32>) -> Self {
        self.top_k = Some(top_k.into());
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: impl Into<Vec<String>>) -> Self {
        self.stop_sequences = Some(stop_sequences.into());
        self
    }

    pub fn max_retries(mut self, max_retries: impl Into<u32>) -> Self {
        self.max_retries = Some(max_retries.into());
        self
    }

    pub fn frequency_penalty(mut self, frequency_penalty: impl Into<f32>) -> Self {
        self.frequency_penalty = Some(frequency_penalty.into());
        self
    }

    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.get_or_insert_default().add_tool(tool);
        self
    }

    pub fn step_count(mut self, step_count: usize) -> Self {
        self.step_count = Some(step_count);
        self
    }

    pub fn build(self) -> LanguageModelRequest<M> {
        let model = self
            .model
            .unwrap_or_else(|| unreachable!("Model must be set"));

        LanguageModelRequest {
            model,
            prompt: self.prompt,
            options: self.options,
        }
    }
}

// ============================================================================
// Section: core function reponse types
// ============================================================================

//TODO: add standard response fields
/// Response from a generate call on `GenerateText`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextResponse {
    /// The options that generated this response
    pub options: LanguageModelOptions, // TODO: implement getters
    /// The model that generated the response.
    pub model: Option<String>,

    /// The reason the model stopped generating text.
    /// Might not necessaryly be an error. for errors, handle
    /// the `Result::Err` variant associated with this type.
    pub stop_reason: Option<String>,

    /// Usage information of the last call
    pub usage: Option<Usage>, // TODO: change to a function for total usage
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

    // TODO: incomplete code
    /// The last content of the response
    pub fn content(&self) -> Option<&LanguageModelResponseContentType> {
        if let Some(msg) = self.options.messages.last() {
            match msg.message {
                Message::Assistant(ref content) => Some(&content.content),
                _ => None,
            }
        } else {
            None
        }
    }

    // TODO: incomplete code
    /// The last text content of the response. returns None if the last
    /// returned content is not a text.
    pub fn text(&self) -> Option<&String> {
        if let Some(msg) = self.options.messages.last() {
            match msg.message {
                Message::Assistant(AssistantMessage {
                    content: LanguageModelResponseContentType::Text(ref c),
                    ..
                }) => Some(c),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn tool_results(&self) -> Option<Vec<ToolOutputInfo>> {
        todo!("The tool results")
    }
    // TODO: Everything that needs to be extracted from the last step
    // should go here ..
}

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
    pub fn step_ids(&self) -> Vec<usize> {
        self.options.messages.iter().map(|t| t.step_id).collect()
    }
}

// ============================================================================
// Section: implementations, Core functions such as generate_text and stream_text
// ============================================================================

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
            ..self.options
        };

        options.current_step_id += 1;
        let response = self.model.generate_text(options.clone()).await?;

        match response.content {
            LanguageModelResponseContentType::Text(text) => {
                let assistant_msg = Message::Assistant(AssistantMessage {
                    content: text.into(),
                    usage: response.usage.clone(),
                });
                options
                    .messages
                    .push(TaggedMessage::new(options.current_step_id, assistant_msg));

                Ok(GenerateTextResponse {
                    options,
                    model: response.model,
                    stop_reason: response.stop_reason,
                    usage: response.usage,
                })
            }
            LanguageModelResponseContentType::ToolCall(tool_info) => {
                // add tool message
                let _ = &options.messages.push(TaggedMessage::new(
                    options.current_step_id,
                    Message::Assistant(AssistantMessage::new(
                        LanguageModelResponseContentType::ToolCall(tool_info.clone()),
                        response.usage,
                    )),
                ));

                let mut tool_steps = Vec::new();
                handle_tool_call(&mut options, vec![tool_info], &mut tool_steps).await;

                // update anything options
                options.current_step_id += 1;
                self.options = options;

                // call the next step with the tool results
                Box::pin(self.generate_text()).await
            }
        }
    }

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
            ..self.options
        };

        options.current_step_id += 1;
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
