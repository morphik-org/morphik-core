use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::error;

use super::{CompletionModel, CompletionResult, Message, Usage};

/// OpenAI completion model via API (supports GPT-4.1-mini and others).
pub struct OpenAICompletionModel {
    model_name: String,
    api_key: String,
    default_max_tokens: u32,
    default_temperature: f64,
    http_client: reqwest::Client,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<UsageResponse>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct UsageResponse {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl OpenAICompletionModel {
    pub fn new(
        model_name: &str,
        api_key: &str,
        default_max_tokens: u32,
        default_temperature: f64,
    ) -> Self {
        Self {
            model_name: model_name.to_string(),
            api_key: api_key.to_string(),
            default_max_tokens,
            default_temperature,
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl CompletionModel for OpenAICompletionModel {
    async fn complete(
        &self,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> anyhow::Result<CompletionResult> {
        let request = ChatRequest {
            model: self.model_name.clone(),
            messages: messages
                .iter()
                .map(|m| ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens: Some(max_tokens.unwrap_or(self.default_max_tokens)),
            temperature: Some(temperature.unwrap_or(self.default_temperature)),
            stream: None,
        };

        let resp = self
            .http_client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI completion API error ({status}): {body}");
        }

        let response: ChatResponse = resp.json().await?;
        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(CompletionResult { content, usage })
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let request = ChatRequest {
            model: self.model_name.clone(),
            messages: messages
                .iter()
                .map(|m| ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens: Some(max_tokens.unwrap_or(self.default_max_tokens)),
            temperature: Some(temperature.unwrap_or(self.default_temperature)),
            stream: Some(true),
        };

        let http_client = self.http_client.clone();
        let api_key = self.api_key.clone();

        tokio::spawn(async move {
            let resp = http_client
                .post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(&api_key)
                .json(&request)
                .send()
                .await;

            match resp {
                Ok(response) => {
                    let text = response.text().await.unwrap_or_default();
                    // Parse SSE response (simplified).
                    for line in text.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                break;
                            }
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(content) = json
                                    .get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("delta"))
                                    .and_then(|d| d.get("content"))
                                    .and_then(|c| c.as_str())
                                {
                                    if tx.send(content.to_string()).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Streaming completion error: {e}");
                }
            }
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialization() {
        let req = ChatRequest {
            model: "gpt-4.1-mini".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.3),
            stream: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4.1-mini");
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn test_chat_response_deserialization() {
        let json = r#"{
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "total_tokens": 16
            }
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello! How can I help?"));
        assert_eq!(resp.usage.unwrap().total_tokens, 16);
    }
}
