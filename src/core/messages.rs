use crate::core::{ToolCallInfo, ToolOutputInfo};
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
    Tool(ToolOutputInfo),
    Developer(String),
}

impl Message {
    /// Start a new conversation with an empty message list.
    ///
    /// Returns a `MessageBuilder<Conversation>`, allowing any number of
    /// `user`/`assistant` calls without forcing an initial system or user message.
    ///
    /// # Example
    /// ```
    /// use aisdk::core::Message;
    ///
    /// let mut msg = Message::conversation_builder();
    /// for _ in 0..10 {
    ///     msg = msg.user("hello");
    /// }
    /// let messages = msg.build(); // messages is a Vec<Message>
    /// ```
    pub fn conversation_builder() -> MessageBuilder<Conversation> {
        MessageBuilder::conversation_builder()
    }

    /// Create a new message builder in the initial state.
    ///
    /// Returns a `MessageBuilder<Initial>` that enforces type-safe order:
    /// the first message **must** be either a system prompt or a user message.
    /// After that, the builder transitions to `Conversation` state and allows
    /// free mixing of user/assistant messages but not system prompts.
    ///
    /// # Example
    /// ```
    /// use aisdk::core::Message;
    ///
    /// let msgs = Message::builder()
    ///     .system("You are helpful.")
    ///     .user("Hello!")
    ///     .assistant("Hi there.")
    ///     .build();
    /// ```
    pub fn builder() -> MessageBuilder<Initial> {
        MessageBuilder::default()
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

impl From<String> for SystemMessage {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SystemMessage {
    fn from(value: &str) -> Self {
        Self::new(value)
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

impl From<String> for UserMessage {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for UserMessage {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantMessage {
    Text(String),
    ToolCall(ToolCallInfo),
}

impl From<String> for AssistantMessage {
    fn from(value: String) -> Self {
        Self::Text(value)
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
    pub fn conversation_builder() -> MessageBuilder<Conversation> {
        MessageBuilder {
            messages: Vec::new(),
            state: std::marker::PhantomData,
        }
    }
}

impl Default for MessageBuilder {
    fn default() -> Self {
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
        self.messages.push(Message::System(content.into().into()));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }

    pub fn user(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages.push(Message::User(content.into().into()));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }
}

impl MessageBuilder<Conversation> {
    pub fn user(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages.push(Message::User(content.into().into()));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }
    pub fn assistant(mut self, content: impl Into<String>) -> MessageBuilder<Conversation> {
        self.messages
            .push(Message::Assistant(content.into().into()));
        MessageBuilder {
            messages: self.messages,
            state: std::marker::PhantomData,
        }
    }
}
