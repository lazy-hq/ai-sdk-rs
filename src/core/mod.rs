//! The core components of the AI SDK, including traits, types, and the main generation function.
//!
//! This module provides the essential building blocks for interacting with language models.
//! It defines the `LanguageModel` trait, which all model providers must implement,
//! and includes the primary `generate_text` function for initiating text generation.
//!
//! Key types like `GenerateTextCallOptions` and `GenerateTextResponse` are also
//! re-exported for convenient access.

pub mod generate_stream;
pub mod generate_text;
pub mod language_model;
pub mod provider;
pub mod types;
pub mod utils;

// Re-export key components to provide a clean public API.
pub use generate_stream::generate_stream;
pub use generate_text::generate_text;
pub use language_model::LanguageModel;
pub use provider::Provider;
//pub use types::{
//GenerateTextCallOptions, GenerateTextResponse, Message, Role, SystemMessage,
//UserMessage,
//};
pub use types::{AssistantMessage, Message, Role, SystemMessage, UserMessage};
