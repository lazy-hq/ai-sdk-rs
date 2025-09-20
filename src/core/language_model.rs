//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use std::ops::{Deref, DerefMut};

use crate::core::types::Message;
use crate::core::utils::resolve_message;
use crate::core::{LanguageModelResponse, LanguageModelStreamResponse};
use crate::error::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

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
    async fn generate(&mut self, options: LanguageModelOptions) -> Result<LanguageModelResponse>;

    /// Performs a streaming text generation request.
    ///
    /// This method sends a prompt to the model and returns a stream of responses.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn generate_stream(
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
    pub messages: Vec<Message>,

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
    // TODO: add support for reponse format
    // pub response_format: Option<ResponseFormat>,

    // Additional provider-specific options. They are passed through
    // to the provider from the AI SDK and enable provider-specific functionality.
    //TODO: add support for provider options
    //pub provider_options: <HashMap<String, <HashMap<String, JsonValue>>>>,
}

impl LanguageModelOptions {
    pub fn builder() -> LanguageModelOptionsBuilder {
        LanguageModelOptionsBuilder::default()
    }
}

/// Options for text generation requests such as `generate_text` and `stream_text`.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions<M: LanguageModel> {
    /// The Language Model to use.
    pub model: M,

    /// The prompt to generate text from.
    /// Only one of prompt or messages should be set.
    pub prompt: Option<String>,

    /// Language model call options for the request
    options: LanguageModelOptions,
}

#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptions<M> {
    fn builder() -> GenerateOptionsBuilder<M> {
        GenerateOptionsBuilder::default()
    }
}

impl<M: LanguageModel> Deref for GenerateOptions<M> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel> DerefMut for GenerateOptions<M> {
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

#[allow(dead_code)]
pub struct GenerateOptionsBuilder<M: LanguageModel, State = ModelStage> {
    model: Option<M>,
    prompt: Option<String>,
    options: LanguageModelOptions,
    state: std::marker::PhantomData<State>,
}

impl<M: LanguageModel, State> Deref for GenerateOptionsBuilder<M, State> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel, State> DerefMut for GenerateOptionsBuilder<M, State> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptionsBuilder<M> {
    fn default() -> Self {
        GenerateOptionsBuilder {
            model: None,
            prompt: None,
            options: LanguageModelOptions::default(),
            state: std::marker::PhantomData,
        }
    }
}

/// ModelStage Builder
#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptionsBuilder<M, ModelStage> {
    pub fn model(self, model: M) -> GenerateOptionsBuilder<M, ConversationStage> {
        GenerateOptionsBuilder {
            model: Some(model),
            prompt: self.prompt,
            options: self.options,
            state: std::marker::PhantomData,
        }
    }
}

/// SystemStage Builder
#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptionsBuilder<M, SystemStage> {
    fn system(self, system: impl Into<String>) -> GenerateOptionsBuilder<M, ConversationStage> {
        GenerateOptionsBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                system: Some(system.into()),
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }
}

/// ConversationStage Builder
#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptionsBuilder<M, ConversationStage> {
    fn prompt(self, prompt: impl Into<String>) -> GenerateOptionsBuilder<M, OptionsStage> {
        GenerateOptionsBuilder {
            model: self.model,
            prompt: Some(prompt.into()),
            options: self.options,
            state: std::marker::PhantomData,
        }
    }

    fn messages(self, messages: Vec<Message>) -> GenerateOptionsBuilder<M, OptionsStage> {
        GenerateOptionsBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                messages,
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }
}

/// OptionsStage Builder
#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptionsBuilder<M, OptionsStage> {
    fn seed(mut self, seed: impl Into<u32>) -> Self {
        self.seed = Some(seed.into());
        self
    }

    fn temperature(mut self, temperature: impl Into<u32>) -> Self {
        self.temperature = Some(temperature.into());
        self
    }

    fn top_p(mut self, top_p: impl Into<u32>) -> Self {
        self.top_p = Some(top_p.into());
        self
    }

    fn top_k(mut self, top_k: impl Into<u32>) -> Self {
        self.top_k = Some(top_k.into());
        self
    }

    fn stop_sequences(mut self, stop_sequences: impl Into<Vec<String>>) -> Self {
        self.stop_sequences = Some(stop_sequences.into());
        self
    }

    fn max_retries(mut self, max_retries: impl Into<u32>) -> Self {
        self.max_retries = Some(max_retries.into());
        self
    }

    fn frequency_penalty(mut self, frequency_penalty: impl Into<f32>) -> Self {
        self.frequency_penalty = Some(frequency_penalty.into());
        self
    }

    fn build(self) -> GenerateOptions<M> {
        let model = self
            .model
            .unwrap_or_else(|| unreachable!("Model must be set"));

        GenerateOptions {
            model,
            prompt: self.prompt,
            options: self.options,
        }
    }
}

/// Core struct for generating text using a language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateText<M: LanguageModel> {
    pub options: GenerateOptions<M>,
}

impl<M: LanguageModel> GenerateText<M> {
    pub fn builder() -> GenerateOptionsBuilder<M> {
        GenerateOptionsBuilder::default()
    }
}

//TODO: add standard response fields
/// Response from a generate call on `GenerateText`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextResponse {
    /// The generated text.
    pub text: String,
}

impl GenerateTextResponse {
    /// Creates a new response with the generated text.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

// ============================================================================
// Section: implementations
// ============================================================================

impl<M: LanguageModel> GenerateText<M> {
    /// Generates text using a specified language model.
    ///
    /// Generate a text and call tools for a given prompt using a language model.
    /// This function does not stream the output. If you want to stream the output, use `StreamText` instead.
    ///
    /// # Arguments
    ///
    /// * `model` - A language model that implements the `LanguageModel` trait.
    ///
    /// * `options` - A `GenerateTextCallOptions` struct containing the model, prompt,
    ///   and other parameters for the request.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the underlying model fails to generate a response.
    pub async fn generate(&mut self) -> Result<GenerateTextResponse> {
        let (system_prompt, messages) = resolve_message(
            &self.options.system,
            &self.options.prompt,
            &self.options.messages,
        );

        let response = self
            .options
            .model
            .generate(
                LanguageModelOptions::builder()
                    .system(system_prompt)
                    .messages(messages)
                    .max_output_tokens(self.options.max_output_tokens)
                    .temperature(self.options.temperature)
                    .top_p(self.options.top_p)
                    .top_k(self.options.top_k)
                    .stop_sequences(self.options.stop_sequences.clone())
                    .build()?,
            )
            .await?;

        let result = GenerateTextResponse::new(response.text);

        Ok(result)
    }
}
