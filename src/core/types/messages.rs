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

impl Message {
    pub fn builder() -> MessageBuilder<Initial> {
        MessageBuilder::builder()
    }
}

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

/// Message State for type safe message list construction.
/// Initial state for initial message builder with either system or user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Initial;

/// Message State for type safe message list construction.
/// Conversation state is used for only user and assistant message builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation;

/// Message Builder with state for type safe message list construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBuilder<State = Initial> {
    messages: Vec<Message>,
    state: std::marker::PhantomData<State>,
}

impl MessageBuilder {
    pub fn builder() -> MessageBuilder<Initial> {
        MessageBuilder {
            messages: Vec::new(),
            state: std::marker::PhantomData,
        }
    }
}

impl<State> MessageBuilder<State> {
    pub fn build(self) -> Vec<Message> {
        self.messages
    }
}

impl MessageBuilder<Initial> {
    pub fn system(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages
            .push(Message::System(SystemMessage::new(content)));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }

    pub fn user(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages.push(Message::User(UserMessage::new(content)));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }
}

impl MessageBuilder<Conversation> {
    pub fn user(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages.push(Message::User(UserMessage::new(content)));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }
    pub fn assistant(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages
            .push(Message::Assistant(AssistantMessage::new(content)));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }
}
