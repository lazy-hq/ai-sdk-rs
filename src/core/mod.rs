//! The core components of the AI SDK, including traits, types, and the main generation function.
//!
//! This module provides the essential building blocks for interacting with language models.
//! It defines the `LanguageModel` trait, which all model providers must implement,
//! and includes the primary `generate_text` function for initiating text generation.
//!
//! Key types like `GenerateTextCallOptions` and `GenerateTextResponse` are also
//! re-exported for convenient access.

pub mod language_model;
pub mod messages;
pub mod provider;
pub mod utils;

// Re-export key components to provide a clean public API.
pub use language_model::{
    GenerateText, GenerateTextResponse, LanguageModel, StreamText, StreamTextResponse,
};
pub use messages::{AssistantMessage, Message, Role, SystemMessage, UserMessage};
pub use provider::Provider;
