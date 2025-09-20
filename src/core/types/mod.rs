//! Core types for AI SDK functions.
mod messages;
mod responses;

// re-export key components for better public API.
pub use messages::{AssistantMessage, Message, Role, SystemMessage, UserMessage};
pub use responses::{
    GenerateTextResponse, LanguageModelResponse, LanguageModelStreamResponse, StreamChunk,
    StreamChunkData,
};
