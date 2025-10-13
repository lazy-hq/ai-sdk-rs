//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

pub mod generate_text;
pub mod request;
pub mod stream_text;

use crate::core::messages::{AssistantMessage, TaggedMessage, TaggedMessageHelpers};
use crate::core::tools::ToolList;
use crate::core::utils;
use crate::core::{Message, ToolCallInfo, ToolResultInfo};
use crate::error::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use futures::Stream;
use schemars::Schema;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Add;
use std::pin::Pin;
use std::sync::Arc;
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
    fn name(&self) -> String;
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
    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream>;
}

// ============================================================================
// Section: hook types
// ============================================================================

pub type StopWhenHook = Arc<dyn Fn(&LanguageModelOptions) -> bool + Send + Sync>;
pub type PrepareStepHook = Arc<dyn Fn(&mut LanguageModelOptions) + Send + Sync>;
pub type OnStepFinishHook = Arc<dyn Fn(&LanguageModelOptions) + Send + Sync>;

// ============================================================================
// Section: structs and impls
// ============================================================================

/// A "step" represents a single cycle of model interaction.
pub struct Step {
    pub step_id: usize,
    pub messages: Vec<Message>,
}

impl Step {
    pub fn new(step_id: usize, messages: Vec<Message>) -> Self {
        Self { step_id, messages }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn usage(&self) -> Usage {
        self.messages()
            .iter()
            .filter_map(|m| match m {
                Message::Assistant(AssistantMessage { usage, .. }) => usage.as_ref(),
                _ => None,
            })
            .fold(Usage::default(), |acc, u| &acc + u)
    }

    pub fn tool_calls(&self) -> Option<Vec<ToolCallInfo>> {
        let calls: Vec<ToolCallInfo> = self
            .messages()
            .iter()
            .filter_map(|msg| match msg {
                Message::Assistant(AssistantMessage {
                    content: LanguageModelResponseContentType::ToolCall(info),
                    ..
                }) => Some(Some(info.clone())),
                _ => None,
            })
            .flatten()
            .collect();
        if calls.is_empty() { None } else { Some(calls) }
    }

    pub fn tool_results(&self) -> Option<Vec<ToolResultInfo>> {
        let results: Vec<ToolResultInfo> = self
            .messages()
            .iter()
            .filter_map(|msg| match msg {
                Message::Tool(info) => Some(info.clone()),
                _ => None,
            })
            .collect();
        if results.is_empty() {
            None
        } else {
            Some(results)
        }
    }
}

// ============================================================================
// Section: options
// ============================================================================

/// Options for a language model request.
#[derive(Clone, Default, Builder)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct LanguageModelOptions {
    /// System prompt to be used for the request.
    pub system: Option<String>,

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

    /// Hook to stop tool calling if returns true
    pub stop_when: Option<StopWhenHook>,

    /// Hook called before each step (language model request)
    pub prepare_step: Option<PrepareStepHook>,

    /// Hook called after each step finishes
    pub on_step_finish: Option<OnStepFinishHook>,

    /// Reasoning effort
    pub reasoning_effort: Option<ReasoningEffort>,

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

impl Debug for LanguageModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanguageModelOptions")
            .field("system", &self.system)
            .field("messages", &self.messages)
            .field("schema", &self.schema)
            .field("seed", &self.seed)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("max_retries", &self.max_retries)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("stop_sequences", &self.stop_sequences)
            .field("presence_penalty", &self.presence_penalty)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("tools", &self.tools)
            .field("current_step_id", &self.current_step_id)
            .field("stop_when", &self.stop_when.is_some())
            .field("prepare_step", &self.prepare_step.is_some())
            .field("on_step_finish", &self.on_step_finish.is_some())
            .finish()
    }
}

impl LanguageModelOptions {
    pub fn builder() -> LanguageModelOptionsBuilder {
        LanguageModelOptionsBuilder::default()
    }

    pub fn messages(&self) -> Vec<Message> {
        self.messages.iter().map(|m| m.message.clone()).collect()
    }

    /// Calls the requested tools, adds tool ouput message to messages,
    /// and decrements the step count. uses the previous step id for tagging
    /// the created messages.
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

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason.clone()
    }
}

// ============================================================================
// Section: response types
// ============================================================================

#[derive(Debug, Clone)]
pub enum LanguageModelResponseContentType {
    Text(String),
    ToolCall(ToolCallInfo),
    Reasoning(String),
    NotSupported(String),
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

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    pub total_tokens: Option<usize>,
    pub reasoning_tokens: Option<usize>,
    pub cached_tokens: Option<usize>,
}

impl Add for &Usage {
    type Output = Usage;

    fn add(self, rhs: Self) -> Self::Output {
        Usage {
            input_tokens: utils::sum_options(self.input_tokens, rhs.input_tokens),
            output_tokens: utils::sum_options(self.output_tokens, rhs.output_tokens),
            total_tokens: utils::sum_options(self.total_tokens, rhs.total_tokens),
            reasoning_tokens: utils::sum_options(self.reasoning_tokens, rhs.reasoning_tokens),
            cached_tokens: utils::sum_options(self.cached_tokens, rhs.cached_tokens),
        }
    }
}

/// Response from a language model.
#[derive(Debug, Clone)]
pub struct LanguageModelResponse {
    /// The generated contents (supports multiple outputs).
    pub contents: Vec<LanguageModelResponseContentType>,

    /// Usage information
    pub usage: Option<Usage>,
}

impl LanguageModelResponse {
    /// Creates a new response with the generated text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            contents: vec![LanguageModelResponseContentType::new(text.into())],
            usage: None,
        }
    }
}

#[derive(Default, Debug, Clone)]
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
    /// The model has failed to generate text. error specified by
    /// the language model
    Failed(String),
    /// The model finsished generating text with incomplete response.
    Incomplete(String),
    /// Return this for unimplemented features for a specific model.
    NotSupported(String),
}

#[derive(Debug, Clone)]
pub enum LanguageModelStreamChunk {
    Delta(LanguageModelStreamChunkType),
    Done(AssistantMessage),
}

/// A common interface for stream responses generated by providers (e.g. OpenAI)
pub(crate) type ProviderStream =
    Pin<Box<dyn Stream<Item = Result<Vec<LanguageModelStreamChunk>>> + Send>>;

// A mapping of `ProviderStream` to a channel like stream.
pub struct LanguageModelStream {
    receiver: Receiver<LanguageModelStreamChunkType>,
}

impl LanguageModelStream {
    // Creates a new MpmcStream with an associated Sender
    pub fn new() -> (Sender<LanguageModelStreamChunkType>, LanguageModelStream) {
        let (tx, rx) = mpsc::channel();
        (tx, LanguageModelStream { receiver: rx })
    }
}

impl Stream for LanguageModelStream {
    type Item = LanguageModelStreamChunkType;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.try_recv() {
            Ok(item) => Poll::Ready(Some(item)),
            Err(mpsc::TryRecvError::Empty) => Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => Poll::Ready(None),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum StopReason {
    #[default]
    // The model has finished generating text
    Finish,
    // Provider specific reasons like timeout, rate limit etc
    Provider(String),
    // The user has explicitly provided a hook causing to stop
    Hook,
    // Problematic errors. Providers specific errors can be accessed
    // through `Error::ProviderError`
    Error(Error),
    // Anything that is not supported by the above reasons
    Other(String),
}

// will be converted to the appropriate level of reasoning
// for a language model
#[derive(Debug, Clone, Copy, Default)]
pub enum ReasoningEffort {
    #[default]
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_add_both_some() {
        let u1 = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(30),
            reasoning_tokens: Some(5),
            cached_tokens: Some(2),
        };
        let u2 = Usage {
            input_tokens: Some(15),
            output_tokens: Some(25),
            total_tokens: Some(40),
            reasoning_tokens: Some(10),
            cached_tokens: Some(3),
        };
        let result = &u1 + &u2;
        assert_eq!(result.input_tokens, Some(25));
        assert_eq!(result.output_tokens, Some(45));
        assert_eq!(result.total_tokens, Some(70));
        assert_eq!(result.reasoning_tokens, Some(15));
        assert_eq!(result.cached_tokens, Some(5));
    }

    #[test]
    fn test_usage_add_first_some_second_none() {
        let u1 = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(30),
            reasoning_tokens: Some(5),
            cached_tokens: Some(2),
        };
        let u2 = Usage {
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
            reasoning_tokens: None,
            cached_tokens: None,
        };
        let result = &u1 + &u2;
        assert_eq!(result.input_tokens, Some(10));
        assert_eq!(result.output_tokens, Some(20));
        assert_eq!(result.total_tokens, Some(30));
        assert_eq!(result.reasoning_tokens, Some(5));
        assert_eq!(result.cached_tokens, Some(2));
    }

    #[test]
    fn test_usage_add_first_none_second_some() {
        let u1 = Usage {
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
            reasoning_tokens: None,
            cached_tokens: None,
        };
        let u2 = Usage {
            input_tokens: Some(15),
            output_tokens: Some(25),
            total_tokens: Some(40),
            reasoning_tokens: Some(10),
            cached_tokens: Some(3),
        };
        let result = &u1 + &u2;
        assert_eq!(result.input_tokens, Some(15));
        assert_eq!(result.output_tokens, Some(25));
        assert_eq!(result.total_tokens, Some(40));
        assert_eq!(result.reasoning_tokens, Some(10));
        assert_eq!(result.cached_tokens, Some(3));
    }

    #[test]
    fn test_usage_add_both_none() {
        let u1 = Usage::default();
        let u2 = Usage::default();
        let result = &u1 + &u2;
        assert_eq!(result.input_tokens, None);
        assert_eq!(result.output_tokens, None);
        assert_eq!(result.total_tokens, None);
        assert_eq!(result.reasoning_tokens, None);
        assert_eq!(result.cached_tokens, None);
    }

    #[test]
    fn test_usage_add_mixed() {
        let u1 = Usage {
            input_tokens: Some(10),
            output_tokens: None,
            total_tokens: Some(30),
            reasoning_tokens: None,
            cached_tokens: Some(2),
        };
        let u2 = Usage {
            input_tokens: None,
            output_tokens: Some(25),
            total_tokens: Some(40),
            reasoning_tokens: Some(10),
            cached_tokens: None,
        };
        let result = &u1 + &u2;
        assert_eq!(result.input_tokens, Some(10));
        assert_eq!(result.output_tokens, Some(25));
        assert_eq!(result.total_tokens, Some(70));
        assert_eq!(result.reasoning_tokens, Some(10));
        assert_eq!(result.cached_tokens, Some(2));
    }

    #[test]
    fn test_usage_add_zero_values() {
        let u1 = Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            total_tokens: Some(0),
            reasoning_tokens: Some(0),
            cached_tokens: Some(0),
        };
        let u2 = Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            total_tokens: Some(0),
            reasoning_tokens: Some(0),
            cached_tokens: Some(0),
        };
        let result = &u1 + &u2;
        assert_eq!(result.input_tokens, Some(0));
        assert_eq!(result.output_tokens, Some(0));
        assert_eq!(result.total_tokens, Some(0));
        assert_eq!(result.reasoning_tokens, Some(0));
        assert_eq!(result.cached_tokens, Some(0));
    }

    #[test]
    fn test_step_usage() {
        let messages = vec![
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text("Hello".to_string()),
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    total_tokens: Some(15),
                    reasoning_tokens: Some(2),
                    cached_tokens: Some(1),
                }),
            }),
            Message::User("Hi".to_string().into()),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text("How are you?".to_string()),
                usage: Some(Usage {
                    input_tokens: Some(5),
                    output_tokens: Some(3),
                    total_tokens: Some(8),
                    reasoning_tokens: Some(1),
                    cached_tokens: Some(0),
                }),
            }),
        ];
        let step = Step::new(1, messages);
        let usage = step.usage();
        assert_eq!(usage.input_tokens, Some(15));
        assert_eq!(usage.output_tokens, Some(8));
        assert_eq!(usage.total_tokens, Some(23));
        assert_eq!(usage.reasoning_tokens, Some(3));
        assert_eq!(usage.cached_tokens, Some(1));
    }

    #[test]
    fn test_step_usage_no_assistant() {
        let messages = vec![
            Message::User("Hello".to_string().into()),
            Message::System("System".to_string().into()),
        ];
        let step = Step::new(0, messages);
        let usage = step.usage();
        assert_eq!(usage, Usage::default());
    }

    #[test]
    fn test_step_tool_calls_empty_messages() {
        let step = Step::new(0, vec![]);
        assert_eq!(step.tool_calls(), None);
    }

    #[test]
    fn test_step_tool_calls_only_non_assistant_messages() {
        let messages = vec![
            Message::System("System".to_string().into()),
            Message::User("User".to_string().into()),
            Message::Tool(ToolResultInfo::new("tool1")),
        ];
        let step = Step::new(0, messages);
        assert_eq!(step.tool_calls(), None);
    }

    #[test]
    fn test_step_tool_calls_single_assistant_with_tool_call() {
        let tool_call = ToolCallInfo::new("test_tool");
        let messages = vec![Message::Assistant(AssistantMessage {
            content: LanguageModelResponseContentType::ToolCall(tool_call.clone()),
            usage: None,
        })];
        let step = Step::new(0, messages);
        let calls = step.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool.name, "test_tool");
    }

    #[test]
    fn test_step_tool_calls_multiple_assistant_with_tool_calls() {
        let tool_call1 = ToolCallInfo::new("tool1");
        let tool_call2 = ToolCallInfo::new("tool2");
        let messages = vec![
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::ToolCall(tool_call1.clone()),
                usage: None,
            }),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::ToolCall(tool_call2.clone()),
                usage: None,
            }),
        ];
        let step = Step::new(0, messages);
        let calls = step.tool_calls().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool.name, "tool1");
        assert_eq!(calls[1].tool.name, "tool2");
    }

    #[test]
    fn test_step_tool_calls_assistant_without_tool_call() {
        let messages = vec![Message::Assistant(AssistantMessage {
            content: LanguageModelResponseContentType::Text("Hello".to_string()),
            usage: None,
        })];
        let step = Step::new(0, messages);
        assert_eq!(step.tool_calls(), None);
    }

    #[test]
    fn test_step_tool_calls_mixed_message_types() {
        let tool_call = ToolCallInfo::new("test_tool");
        let messages = vec![
            Message::System("System".to_string().into()),
            Message::User("User".to_string().into()),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::ToolCall(tool_call.clone()),
                usage: None,
            }),
            Message::Tool(ToolResultInfo::new("other_tool")),
        ];
        let step = Step::new(0, messages);
        let calls = step.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool.name, "test_tool");
    }

    #[test]
    fn test_step_tool_calls_duplicate_tool_calls() {
        let tool_call1 = ToolCallInfo::new("tool1");
        let tool_call2 = ToolCallInfo::new("tool1"); // Same name
        let messages = vec![
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::ToolCall(tool_call1.clone()),
                usage: None,
            }),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::ToolCall(tool_call2.clone()),
                usage: None,
            }),
        ];
        let step = Step::new(0, messages);
        let calls = step.tool_calls().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool.name, "tool1");
        assert_eq!(calls[1].tool.name, "tool1");
    }

    #[test]
    fn test_step_tool_results_empty_messages() {
        let step = Step::new(0, vec![]);
        assert!(step.tool_results().is_none());
    }

    #[test]
    fn test_step_tool_results_only_non_tool_messages() {
        let messages = vec![
            Message::System("System".to_string().into()),
            Message::User("User".to_string().into()),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text("Assistant".to_string()),
                usage: None,
            }),
        ];
        let step = Step::new(0, messages);
        assert!(step.tool_results().is_none());
    }

    #[test]
    fn test_step_tool_results_single_tool_message() {
        let tool_result = ToolResultInfo::new("test_tool");
        let messages = vec![Message::Tool(tool_result.clone())];
        let step = Step::new(0, messages);
        let results = step.tool_results().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool.name, "test_tool");
    }

    #[test]
    fn test_step_tool_results_multiple_tool_messages() {
        let tool_result1 = ToolResultInfo::new("tool1");
        let tool_result2 = ToolResultInfo::new("tool2");
        let messages = vec![
            Message::Tool(tool_result1.clone()),
            Message::Tool(tool_result2.clone()),
        ];
        let step = Step::new(0, messages);
        let results = step.tool_results().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool.name, "tool1");
        assert_eq!(results[1].tool.name, "tool2");
    }

    #[test]
    fn test_step_tool_results_mixed_message_types() {
        let tool_result = ToolResultInfo::new("test_tool");
        let messages = vec![
            Message::System("System".to_string().into()),
            Message::User("User".to_string().into()),
            Message::Tool(tool_result.clone()),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text("Assistant".to_string()),
                usage: None,
            }),
        ];
        let step = Step::new(0, messages);
        let results = step.tool_results().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool.name, "test_tool");
    }

    #[test]
    fn test_step_tool_results_no_tool_messages_but_others_present() {
        let messages = vec![
            Message::System("System".to_string().into()),
            Message::User("User".to_string().into()),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text("Assistant".to_string()),
                usage: None,
            }),
        ];
        let step = Step::new(0, messages);
        assert!(step.tool_results().is_none());
    }

    #[test]
    fn test_step_tool_results_duplicate_tool_entries() {
        let tool_result1 = ToolResultInfo::new("tool1");
        let tool_result2 = ToolResultInfo::new("tool1"); // Same name
        let messages = vec![
            Message::Tool(tool_result1.clone()),
            Message::Tool(tool_result2.clone()),
        ];
        let step = Step::new(0, messages);
        let results = step.tool_results().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool.name, "tool1");
        assert_eq!(results[1].tool.name, "tool1");
    }

    #[test]
    fn test_step_tool_results_preserving_original_message_order() {
        let tool_result1 = ToolResultInfo::new("tool1");
        let tool_result2 = ToolResultInfo::new("tool2");
        let tool_result3 = ToolResultInfo::new("tool3");
        let messages = vec![
            Message::System("System".to_string().into()),
            Message::Tool(tool_result1.clone()),
            Message::User("User".to_string().into()),
            Message::Tool(tool_result2.clone()),
            Message::Assistant(AssistantMessage {
                content: LanguageModelResponseContentType::Text("Assistant".to_string()),
                usage: None,
            }),
            Message::Tool(tool_result3.clone()),
        ];
        let step = Step::new(0, messages);
        let results = step.tool_results().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].tool.name, "tool1");
        assert_eq!(results[1].tool.name, "tool2");
        assert_eq!(results[2].tool.name, "tool3");
    }

    #[test]
    fn test_step_tool_results_large_number_of_messages() {
        let mut messages = Vec::new();
        // Add 1000 messages with tool results interspersed
        for i in 0..1000 {
            messages.push(Message::Tool(ToolResultInfo::new(format!("tool{i}"))));
            if i % 100 == 0 {
                messages.push(Message::User(format!("User message {}", i).into()));
            }
        }
        let step = Step::new(0, messages);
        let results = step.tool_results().unwrap();
        assert_eq!(results.len(), 1000);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.tool.name, format!("tool{}", i));
        }
    }
}
