use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::EmbeddingModel;

/// OpenAI embedding model via API.
pub struct OpenAIEmbeddingModel {
    model_name: String,
    api_key: String,
    dimensions: u32,
    http_client: reqwest::Client,
    batch_size: usize,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAIEmbeddingModel {
    pub fn new(model_name: &str, api_key: &str, dimensions: u32) -> Self {
        Self {
            model_name: model_name.to_string(),
            api_key: api_key.to_string(),
            dimensions,
            http_client: reqwest::Client::new(),
            batch_size: 100,
        }
    }

    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            model: self.model_name.clone(),
            input: texts.to_vec(),
        };

        let resp = self
            .http_client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI embedding API error ({status}): {body}");
        }

        let response: EmbeddingResponse = resp.json().await?;
        Ok(response.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl EmbeddingModel for OpenAIEmbeddingModel {
    async fn embed_for_ingestion(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            let batch_vec: Vec<String> = batch.to_vec();
            let embeddings = self.embed_batch(&batch_vec).await?;

            // Validate dimensions.
            for emb in &embeddings {
                if emb.len() != self.dimensions as usize {
                    anyhow::bail!(
                        "Embedding dimension mismatch: expected {}, got {}",
                        self.dimensions,
                        emb.len()
                    );
                }
            }

            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    async fn embed_for_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let results = self.embed_batch(&[query.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned for query"))
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_request_serialization() {
        let req = EmbeddingRequest {
            model: "text-embedding-3-small".to_string(),
            input: vec!["hello world".to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "text-embedding-3-small");
        assert_eq!(json["input"][0], "hello world");
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}
            ],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }"#;
        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding.len(), 3);
    }
}
