//! Core types for AI SDK functions.
mod messages;

// re-export key components for better public API.
pub use messages::{AssistantMessage, Message, Role, SystemMessage, UserMessage};
