//! Helper functions and conversions for the OpenAI provider.

use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequestArgs,
};

use crate::core::types::LanguageModelCallOptions;

fn user_message(message: &str) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage::from(message))
}

fn system_message(message: &str) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage::from(message))
}

struct OpenAiMessage(ChatCompletionRequestMessage);

impl From<OpenAiMessage> for String {
    /// Handle the conversion from any `OpenAiMessage` to `String`. Currently it only handles
    /// user messages that are texts or part of a text. returns empty string if it is not.
    fn from(value: OpenAiMessage) -> Self {
        match value.0 {
            ChatCompletionRequestMessage::User(user_message) => match &user_message.content {
                ChatCompletionRequestUserMessageContent::Text(text) => text.to_string(),
                ChatCompletionRequestUserMessageContent::Array(arr) => match arr.first().unwrap() {
                    ChatCompletionRequestUserMessageContentPart::Text(text) => {
                        text.text.to_string()
                    }
                    _ => "".to_string(),
                },
            },
            _ => "".to_string(),
        }
    }
}

impl From<LanguageModelCallOptions> for CreateChatCompletionRequestArgs {
    fn from(options: LanguageModelCallOptions) -> Self {
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        //request_builder.model(self.model_name().to_string());

        if let Some(max_tokens) = options.max_tokens {
            request_builder.max_tokens(max_tokens);
        };

        if let Some(temprature) = options.temprature {
            request_builder.temperature(temprature as f32 / 100_f32);
        };

        if let Some(top_p) = options.top_p {
            request_builder.top_p(top_p as f32 / 100_f32);
        };

        if options.top_k.is_some() {
            log::warn!("WrongProviderInput: top_k is not supported by OpenAI");
        };

        if let Some(stop) = options.stop {
            request_builder.stop(stop);
        };

        let msg: ChatCompletionRequestMessage =
            OpenAiMessage(user_message(&options.prompt)).0;
        let mut msgs = vec![msg];

        if let Some(system_prompt) = options.system_prompt {
            msgs.push(system_message(&system_prompt));
        }
        request_builder.messages(msgs);

        request_builder
    }
}