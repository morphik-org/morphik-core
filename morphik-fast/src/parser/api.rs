use std::collections::HashMap;
use tracing::info;

use crate::models::chunk::Chunk;

/// API-mode parser that offloads parsing to external GPU servers.
pub struct ApiParser {
    api_endpoints: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    http_client: reqwest::Client,
}

impl ApiParser {
    pub fn new(api_endpoints: Vec<String>, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            api_endpoints,
            chunk_size,
            chunk_overlap,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Parse a document by sending it to the API endpoint.
    /// Returns parsed text content and metadata.
    pub async fn parse_document(
        &self,
        file_bytes: &[u8],
        filename: &str,
    ) -> anyhow::Result<(String, HashMap<String, serde_json::Value>)> {
        if self.api_endpoints.is_empty() {
            anyhow::bail!("No API endpoints configured for parser");
        }

        let endpoint = &self.api_endpoints[0];
        let url = format!("{endpoint}/parse");

        let part = reqwest::multipart::Part::bytes(file_bytes.to_vec())
            .file_name(filename.to_string());
        let form = reqwest::multipart::Form::new().part("file", part);

        let resp = self
            .http_client
            .post(&url)
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Parser API error ({status}): {body}");
        }

        let result: serde_json::Value = resp.json().await?;
        let text = result
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let metadata: HashMap<String, serde_json::Value> = result
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok((text, metadata))
    }

    /// Parse a document and split into chunks.
    pub async fn parse_and_chunk(
        &self,
        file_bytes: &[u8],
        filename: &str,
    ) -> anyhow::Result<Vec<Chunk>> {
        let (text, metadata) = self.parse_document(file_bytes, filename).await?;

        let text_chunks = super::split_text(&text, self.chunk_size, self.chunk_overlap);

        let chunks: Vec<Chunk> = text_chunks
            .into_iter()
            .map(|content| Chunk {
                content,
                metadata: metadata.clone(),
            })
            .collect();

        info!("Parsed {} into {} chunks", filename, chunks.len());
        Ok(chunks)
    }

    /// Parse plain text content (no file upload needed).
    pub fn parse_text_content(
        &self,
        text: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Vec<Chunk> {
        let text_chunks = super::split_text(text, self.chunk_size, self.chunk_overlap);
        text_chunks
            .into_iter()
            .map(|content| Chunk {
                content,
                metadata: metadata.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_content() {
        let parser = ApiParser::new(vec![], 100, 10);
        let chunks = parser.parse_text_content("Hello world. This is a test.", HashMap::new());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello world. This is a test.");
    }

    #[test]
    fn test_parse_text_content_multiple_chunks() {
        let parser = ApiParser::new(vec![], 50, 5);
        let text = "a".repeat(100) + "\n\n" + &"b".repeat(100);
        let chunks = parser.parse_text_content(&text, HashMap::new());
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_parse_text_with_metadata() {
        let parser = ApiParser::new(vec![], 1000, 100);
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), serde_json::json!("test"));
        let chunks = parser.parse_text_content("content", metadata);
        assert_eq!(chunks[0].metadata.get("source").unwrap(), "test");
    }
}
