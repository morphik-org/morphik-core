pub mod postgres;

use async_trait::async_trait;
use std::collections::HashMap;

use crate::models::api::DocumentInfo;

/// Abstract database interface for document metadata.
#[async_trait]
pub trait Database: Send + Sync {
    /// Insert or update a document record.
    async fn upsert_document(
        &self,
        document_id: &str,
        external_id: Option<&str>,
        filename: Option<&str>,
        app_id: &str,
        chunk_count: i32,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<()>;

    /// Get document information.
    async fn get_document(
        &self,
        document_id: &str,
        app_id: &str,
    ) -> anyhow::Result<Option<DocumentInfo>>;

    /// List documents for an app.
    async fn list_documents(
        &self,
        app_id: &str,
        limit: i64,
        offset: i64,
    ) -> anyhow::Result<Vec<DocumentInfo>>;

    /// Delete a document.
    async fn delete_document(
        &self,
        document_id: &str,
        app_id: &str,
    ) -> anyhow::Result<bool>;

    /// Get document by external ID.
    async fn get_document_by_external_id(
        &self,
        external_id: &str,
        app_id: &str,
    ) -> anyhow::Result<Option<DocumentInfo>>;

    /// Initialize database tables.
    async fn initialize(&self) -> anyhow::Result<()>;
}
