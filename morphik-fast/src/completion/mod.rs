pub mod openai;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Completion response.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub content: String,
    pub usage: Option<Usage>,
}

/// Token usage information.
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Abstract completion model interface.
#[async_trait]
pub trait CompletionModel: Send + Sync {
    /// Generate a completion for the given messages.
    async fn complete(
        &self,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> anyhow::Result<CompletionResult>;

    /// Generate a streaming completion. Returns a stream of text chunks.
    async fn complete_stream(
        &self,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<String>>;
}
