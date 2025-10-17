//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use crate::Error;
use crate::core::language_model::{
    LanguageModel, LanguageModelResponseContentType, OnStepFinishHook, PrepareStepHook, Step,
    StopReason, StopWhenHook, Usage,
};
use crate::core::messages::TaggedMessage;
use crate::core::messages::TaggedMessageHelpers;
use crate::core::tools::{Tool, ToolList};
use crate::core::{AssistantMessage, Message, ToolCallInfo, ToolResultInfo};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Options for text generation requests such as `generate_text` and `stream_text`.
#[derive(Clone, Default)]
pub struct LanguageModelRequest<M: LanguageModel> {
    /// The prompt to generate text from.
    /// Only one of prompt or messages should be set.
    pub prompt: Option<String>,

    pub system: Option<String>,

    /// The Language Model to use.
    pub(crate) model: M,

    pub(crate) options: M::Options,

    /// Hook to stop tool calling if returns true
    pub(crate) stop_when: Option<StopWhenHook<M>>,

    /// Hook called before each step (language model request)
    pub(crate) prepare_step: Option<PrepareStepHook<M>>,

    /// Hook called after each step finishes
    pub(crate) on_step_finish: Option<OnStepFinishHook<M>>,

    /// List of tools to use.
    pub(crate) tools: Option<ToolList>,

    /// Used to track message steps
    pub(crate) current_step_id: usize,

    /// The messages to generate text from.
    /// At least User Message is required.
    pub(crate) messages: Vec<TaggedMessage>,

    // The stop reasons. should be updated after each step.
    pub(crate) stop_reason: Option<StopReason>,
}

impl<M: LanguageModel> Debug for LanguageModelRequest<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanguageModelRequest")
            .field("model", &self.model)
            .field("prompt", &self.prompt)
            .field("messages", &self.messages)
            .finish()
    }
}

impl<M: LanguageModel> LanguageModelRequest<M> {
    pub fn builder() -> LanguageModelRequestBuilder<M> {
        LanguageModelRequestBuilder::default()
    }

    pub(crate) async fn handle_tool_call(&mut self, input: &ToolCallInfo) -> &mut Self {
        if let Some(tools) = &self.tools {
            let tool_result_task = tools.execute(input.clone()).await;
            let tool_result = tool_result_task
                .await
                .map_err(|err| Error::ToolCallError(format!("Error executing tool: {}", err)))
                .and_then(|result| result);

            let mut tool_output_infos = Vec::new();

            let mut tool_output_info = ToolResultInfo::new(&input.tool.name);
            let output = match tool_result {
                Ok(result) => serde_json::Value::String(result),
                Err(err) => serde_json::Value::String(format!("Error: {}", err)),
            };
            tool_output_info.output(output);
            tool_output_info.id(&input.tool.id);
            tool_output_infos.push(tool_output_info.clone());

            // update messages
            self.messages.push(TaggedMessage::new(
                self.current_step_id,
                Message::Tool(tool_output_info),
            ));

            self
        } else {
            self
        }
    }

    pub fn messages(&self) -> Vec<Message> {
        self.messages.iter().map(|t| t.clone().into()).collect()
    }

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason.clone()
    }

    pub fn step(&self, index: usize) -> Option<Step> {
        let messages: Vec<Message> = self
            .messages
            .iter()
            .filter(|t| t.step_id == index)
            .map(|t| t.message.clone())
            .collect();
        if messages.is_empty() {
            None
        } else {
            Some(Step::new(index, messages))
        }
    }

    pub fn last_step(&self) -> Option<Step> {
        let max_step = self.messages.iter().map(|t| t.step_id).max()?;
        self.step(max_step)
    }

    pub fn steps(&self) -> Vec<Step> {
        let mut step_map: HashMap<usize, Vec<Message>> = HashMap::new();
        for tagged in &self.messages {
            step_map
                .entry(tagged.step_id)
                .or_default()
                .push(tagged.message.clone());
        }
        let mut steps: Vec<Step> = step_map
            .into_iter()
            .map(|(id, msgs)| Step::new(id, msgs))
            .collect();
        steps.sort_by_key(|s| s.step_id);
        steps
    }

    pub fn usage(&self) -> Usage {
        self.steps()
            .iter()
            .map(|s| s.usage())
            .fold(Usage::default(), |acc, u| &acc + &u)
    }

    pub fn content(&self) -> Option<&LanguageModelResponseContentType> {
        if let Some(msg) = self.messages.last() {
            match msg.message {
                Message::Assistant(ref assistant_msg) => {
                    if let LanguageModelResponseContentType::Reasoning(_) = assistant_msg.content {
                        None
                    } else {
                        Some(&assistant_msg.content)
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn text(&self) -> Option<String> {
        if let Some(msg) = self.messages.last() {
            match msg.message {
                Message::Assistant(AssistantMessage {
                    content: LanguageModelResponseContentType::Text(ref text),
                    ..
                }) => Some(text.clone()),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn tool_results(&self) -> Option<Vec<ToolResultInfo>> {
        self.messages.as_slice().extract_tool_results()
    }

    pub fn tool_calls(&self) -> Option<Vec<ToolCallInfo>> {
        self.messages.as_slice().extract_tool_calls()
    }
}

impl<M: LanguageModel> Deref for LanguageModelRequest<M> {
    type Target = M::Options;

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
    pub(crate) model: Option<M>,
    pub options: M::Options, // TODO: use pub(crate) once clippy issue is fixed
    pub(crate) prompt: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) messages: Vec<TaggedMessage>,
    pub(crate) stop_when: Option<StopWhenHook<M>>,
    pub(crate) prepare_step: Option<PrepareStepHook<M>>,
    pub(crate) on_step_finish: Option<OnStepFinishHook<M>>,
    pub(crate) tools: Option<ToolList>,
    state: std::marker::PhantomData<State>,
}

impl<M: LanguageModel> LanguageModelRequestBuilder<M> {
    fn default() -> Self {
        LanguageModelRequestBuilder {
            model: None,
            options: M::Options::default(),
            prompt: None,
            system: None,
            messages: vec![],
            stop_when: None,
            prepare_step: None,
            on_step_finish: None,
            tools: None,
            state: std::marker::PhantomData,
        }
    }
}

/// ModelStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, ModelStage> {
    pub fn model(self, model: M) -> LanguageModelRequestBuilder<M, SystemStage> {
        LanguageModelRequestBuilder {
            model: Some(model),
            options: M::Options::default(),
            prompt: self.prompt,
            system: self.system,
            messages: self.messages,
            stop_when: self.stop_when,
            prepare_step: self.prepare_step,
            on_step_finish: self.on_step_finish,
            tools: self.tools,
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
            options: M::Options::default(),
            prompt: self.prompt,
            system: Some(system.into()),
            messages: self.messages,
            stop_when: self.stop_when,
            prepare_step: self.prepare_step,
            on_step_finish: self.on_step_finish,
            tools: self.tools,
            state: std::marker::PhantomData,
        }
    }

    pub fn prompt(self, prompt: impl Into<String>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            options: M::Options::default(),
            prompt: Some(prompt.into()),
            system: self.system,
            messages: self.messages,
            stop_when: self.stop_when,
            prepare_step: self.prepare_step,
            on_step_finish: self.on_step_finish,
            tools: self.tools,
            state: std::marker::PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            options: M::Options::default(),
            prompt: self.prompt,
            system: self.system,
            messages: messages.into_iter().map(|msg| msg.into()).collect(),
            stop_when: self.stop_when,
            prepare_step: self.prepare_step,
            on_step_finish: self.on_step_finish,
            tools: self.tools,
            state: std::marker::PhantomData,
        }
    }
}

/// ConversationStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, ConversationStage> {
    pub fn prompt(self, prompt: impl Into<String>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            options: M::Options::default(),
            prompt: Some(prompt.into()),
            system: self.system,
            messages: self.messages,
            stop_when: self.stop_when,
            prepare_step: self.prepare_step,
            on_step_finish: self.on_step_finish,
            tools: self.tools,
            state: std::marker::PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            options: M::Options::default(),
            prompt: self.prompt,
            messages: messages.into_iter().map(|msg| msg.into()).collect(),
            system: self.system,
            stop_when: self.stop_when,
            prepare_step: self.prepare_step,
            on_step_finish: self.on_step_finish,
            tools: self.tools,
            state: std::marker::PhantomData,
        }
    }
}

impl<M: LanguageModel> LanguageModelRequestBuilder<M, OptionsStage> {
    pub fn with_tool(&mut self, tool: Tool) -> &mut Self {
        self.tools.get_or_insert_default().add_tool(tool);
        self
    }

    pub fn stop_when<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(&LanguageModelRequest<M>) -> bool + Send + Sync + 'static,
    {
        self.stop_when = Some(Arc::new(hook));
        self
    }

    pub fn prepare_step<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(LanguageModelRequest<M>) -> Option<LanguageModelRequest<M>> + Send + Sync + 'static,
    {
        self.prepare_step = Some(Arc::new(hook));
        self
    }

    pub fn on_step_finish<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(LanguageModelRequest<M>) -> Option<LanguageModelRequest<M>> + Send + Sync + 'static,
    {
        self.on_step_finish = Some(Arc::new(hook));
        self
    }
}
