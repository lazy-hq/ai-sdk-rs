use serde::{Deserialize, Serialize};

/// Role for model messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Message Type for model messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
}

//TODO: add Message impl for easy creation of messages + type safety that guarantees system
//prompts are always first

/// System model message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    role: Role,
    pub content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
}

/// User model message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    role: Role,
    pub content: String,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
}

/// Assistant model message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    role: Role,
    pub content: String,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}
