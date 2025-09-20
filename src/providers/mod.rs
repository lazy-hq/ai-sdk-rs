//! This module provides the `Provider` trait, which defines the interface for
//! interacting with different AI providers.

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "models-dev")]
pub use crate::models_dev::traits::ModelsDevAware;
