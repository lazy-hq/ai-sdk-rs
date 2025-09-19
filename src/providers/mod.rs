//! This module provides the `Provider` trait, which defines the interface for
//! interacting with different AI providers.

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;
