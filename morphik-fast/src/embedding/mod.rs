pub mod openai;

use async_trait::async_trait;

/// Abstract embedding model interface.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Embed a list of text chunks for ingestion. Returns a vector of embeddings.
    async fn embed_for_ingestion(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single query string.
    async fn embed_for_query(&self, query: &str) -> anyhow::Result<Vec<f32>>;

    /// Return the embedding dimensions.
    fn dimensions(&self) -> u32;
}
