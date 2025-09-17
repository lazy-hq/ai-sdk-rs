//! Options for AI SDK functions and traits.

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::core::types::messages::Message;
use crate::error::{Error, Result};

/// Shortens the definition of the `GenerateTextCallOptions` and
/// `LanguageModelCallOptions` because all the fields from the first are also
/// second.
macro_rules! define_with_lm_call_options {
        ( $( ($field:ident, $typ:ty, $default:expr, $comment:expr) ),* ) => {
            #[derive(Debug, Clone, Serialize, Deserialize, Builder)]
            #[builder(pattern = "owned", setter(into), build_fn(name = "build_inner", error = "Error"))]
            pub struct GenerateTextCallOptions  {
                $(
                    #[doc = $comment]
                    #[builder(default = $default)]
                    pub $field: $typ,
                )*
                // Define `GenerateTextCallOptions` specific entries here

                /// The prompt to generate text from. Uses the completion format.
                /// Only one of prompt or messages should be set.
                #[builder(default = "None")]
                pub prompt: Option<String>,

                /// Maximum number of retries.
                #[builder(default = "100")]
                pub max_retries: u32,
            }

            /// Options for a language model request. The ones directly passed to the
            /// provider,`None` is used for the provider default.
            #[derive(Debug, Clone, Serialize, Deserialize, Builder)]
            #[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
            pub struct LanguageModelCallOptions {
                $(
                    #[doc = $comment]
                    #[builder(default = $default)]
                    pub $field: $typ,
                )*
            }
        };
}

// TODO: add support for main options
define_with_lm_call_options!(
    // identifier, type, default, comment
    (
        system,
        Option<String>,
        None,
        "System prompt to be used for the request."
    ),
    (
        messages,
        Option<Vec<Message>>,
        None,
        "The messages to generate text from. Uses the chat format. Only one of prompt or messages should be set."
    ),
    (
        max_tokens,
        Option<u32>,
        None,
        "The maximum number of tokens to generate."
    ),
    (temperature, Option<u32>, None, "Randomness."),
    (top_p, Option<u32>, None, "Nucleus sampling."),
    (top_k, Option<u32>, None, "Top-k sampling."),
    (stop, Option<Vec<String>>, None, "Stop sequence.")
);

impl LanguageModelCallOptions {
    /// Creates a new builder for `LanguageModelCallOptions`.
    pub fn builder() -> LanguageModelCallOptionsBuilder {
        LanguageModelCallOptionsBuilder::default()
    }
}

/*
 *CORE function options
 */

/*Generate Text Options and builder*/

impl GenerateTextCallOptions {
    /// Creates a new builder for `GenerateTextCallOptions`.
    pub fn builder() -> GenerateTextCallOptionsBuilder {
        GenerateTextCallOptionsBuilder::default()
    }
}

impl GenerateTextCallOptionsBuilder {
    pub fn build(self) -> Result<GenerateTextCallOptions> {
        let options = self.build_inner()?;

        if options.prompt.is_some() && options.messages.is_some() {
            return Err(Error::InvalidInput(
                "Cannot set both prompt and messages".to_string(),
            ));
        }

        if options.messages.is_none() && options.prompt.is_none() {
            return Err(Error::InvalidInput(
                "Messages or prompt must be set".to_string(),
            ));
        }

        Ok(options)
    }
}

/*Stream Text Options and builder*/

//TODO: add separate options for generate_stream and generate_text, currently they are using the
//same GenerateTextCallOptions
