pub mod cache;
pub mod fast_multivector;
pub mod pgvector;
pub mod utils;

use async_trait::async_trait;
use crate::models::api::StoreMetrics;
use crate::models::chunk::DocumentChunk;

/// Abstract vector store interface.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Store document chunks and their embeddings. Returns (success, stored_ids, metrics).
    async fn store_embeddings(
        &self,
        chunks: &[DocumentChunk],
        app_id: Option<&str>,
    ) -> anyhow::Result<(bool, Vec<String>, StoreMetrics)>;

    /// Find similar chunks by embedding.
    async fn query_similar(
        &self,
        query_embedding: &[f32],
        k: usize,
        doc_ids: Option<&[String]>,
        app_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentChunk>>;

    /// Retrieve specific chunks by (document_id, chunk_number) pairs.
    async fn get_chunks_by_id(
        &self,
        chunk_identifiers: &[(String, i32)],
        app_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentChunk>>;

    /// Delete all chunks for a document.
    async fn delete_chunks_by_document_id(
        &self,
        document_id: &str,
        app_id: Option<&str>,
    ) -> anyhow::Result<bool>;

    /// Initialize the store (create tables, etc.).
    async fn initialize(&self) -> anyhow::Result<bool>;
}
