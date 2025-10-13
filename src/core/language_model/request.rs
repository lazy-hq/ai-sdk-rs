//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use crate::core::Message;
use crate::core::language_model::{LanguageModel, LanguageModelOptions};
use crate::core::tools::Tool;
use schemars::{JsonSchema, schema_for};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Options for text generation requests such as `generate_text` and `stream_text`.
#[derive(Debug, Clone)]
pub struct LanguageModelRequest<M: LanguageModel> {
    /// The Language Model to use.
    pub model: M,

    /// The prompt to generate text from.
    /// Only one of prompt or messages should be set.
    pub prompt: Option<String>,

    /// Language model call options for the request
    pub(crate) options: LanguageModelOptions,
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

    pub fn stop_when<F>(mut self, hook: F) -> Self
    where
        F: Fn(&LanguageModelOptions) -> bool + Send + Sync + 'static,
    {
        self.stop_when = Some(Arc::new(hook));
        self
    }

    pub fn prepare_step<F>(mut self, hook: F) -> Self
    where
        F: Fn(&mut LanguageModelOptions) + Send + Sync + 'static,
    {
        self.prepare_step = Some(Arc::new(hook));
        self
    }

    pub fn on_step_finish<F>(mut self, hook: F) -> Self
    where
        F: Fn(&LanguageModelOptions) + Send + Sync + 'static,
    {
        self.on_step_finish = Some(Arc::new(hook));
        self
    }

    pub fn reasoning_effort(
        mut self,
        reasoning_effort: impl Into<crate::core::language_model::ReasoningEffort>,
    ) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
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
