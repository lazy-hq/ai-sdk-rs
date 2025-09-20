//! Models.dev API client with caching support.
//!
//! This module provides a client for interacting with the models.dev API,
//! featuring both memory and disk caching to minimize API calls and improve performance.

#[cfg(feature = "models-dev")]
pub mod client;

#[cfg(feature = "models-dev")]
pub mod error;

#[cfg(feature = "models-dev")]
pub mod types;

#[cfg(feature = "models-dev")]
pub mod traits;

#[cfg(feature = "models-dev")]
pub mod registry;

#[cfg(feature = "models-dev")]
pub use client::ModelsDevClient;

#[cfg(feature = "models-dev")]
pub use error::ModelsDevError;

#[cfg(feature = "models-dev")]
pub use registry::{
    EnvVarInfo, ModelCostInfo, ModelInfo, ModelLimitInfo, ProviderInfo, ProviderRegistry,
};

#[cfg(feature = "models-dev")]
pub use types::{
    ApiInfo, DocInfo, EnvVar, Modalities, Model, ModelCost, ModelLimit, ModelsDevResponse, NpmInfo,
    Provider,
};

#[cfg(feature = "models-dev")]
pub use traits::{ModelsDevAware, ProviderConnectionInfo};
