use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use tracing::info;

use crate::models::api::DocumentInfo;
use super::Database;

/// PostgreSQL database for document metadata storage.
pub struct PostgresDatabase {
    pool: PgPool,
}

impl PostgresDatabase {
    pub async fn new(uri: &str, pool_size: u32) -> anyhow::Result<Self> {
        let clean_uri = uri.replace("postgresql+asyncpg://", "postgresql://");
        let pool = PgPoolOptions::new()
            .max_connections(pool_size)
            .acquire_timeout(std::time::Duration::from_secs(10))
            .connect(&clean_uri)
            .await?;

        info!("Connected to PostgreSQL (pool_size={pool_size})");
        Ok(Self { pool })
    }
}

#[async_trait]
impl Database for PostgresDatabase {
    async fn initialize(&self) -> anyhow::Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                document_id VARCHAR(255) NOT NULL UNIQUE,
                external_id VARCHAR(255),
                filename VARCHAR(1024),
                app_id VARCHAR(255) NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_documents_app_id ON documents(app_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_documents_external_id ON documents(external_id)",
        )
        .execute(&self.pool)
        .await?;

        info!("Database tables initialized");
        Ok(())
    }

    async fn upsert_document(
        &self,
        document_id: &str,
        external_id: Option<&str>,
        filename: Option<&str>,
        app_id: &str,
        chunk_count: i32,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<()> {
        let metadata_json = serde_json::to_value(metadata)?;

        sqlx::query(
            "INSERT INTO documents (document_id, external_id, filename, app_id, chunk_count, metadata)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (document_id)
             DO UPDATE SET
                external_id = COALESCE(EXCLUDED.external_id, documents.external_id),
                filename = COALESCE(EXCLUDED.filename, documents.filename),
                chunk_count = EXCLUDED.chunk_count,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP",
        )
        .bind(document_id)
        .bind(external_id)
        .bind(filename)
        .bind(app_id)
        .bind(chunk_count)
        .bind(&metadata_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn get_document(
        &self,
        document_id: &str,
        app_id: &str,
    ) -> anyhow::Result<Option<DocumentInfo>> {
        let row = sqlx::query(
            "SELECT document_id, external_id, filename, chunk_count, metadata, created_at
             FROM documents
             WHERE document_id = $1 AND app_id = $2",
        )
        .bind(document_id)
        .bind(app_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| {
            let metadata: serde_json::Value = r.get("metadata");
            DocumentInfo {
                document_id: r.get("document_id"),
                external_id: r.get("external_id"),
                filename: r.get("filename"),
                chunk_count: r.get("chunk_count"),
                metadata: serde_json::from_value(metadata).unwrap_or_default(),
                created_at: r
                    .get::<Option<chrono::DateTime<chrono::Utc>>, _>("created_at")
                    .map(|dt| dt.to_rfc3339()),
            }
        }))
    }

    async fn list_documents(
        &self,
        app_id: &str,
        limit: i64,
        offset: i64,
    ) -> anyhow::Result<Vec<DocumentInfo>> {
        let rows = sqlx::query(
            "SELECT document_id, external_id, filename, chunk_count, metadata, created_at
             FROM documents
             WHERE app_id = $1
             ORDER BY created_at DESC
             LIMIT $2 OFFSET $3",
        )
        .bind(app_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| {
                let metadata: serde_json::Value = r.get("metadata");
                DocumentInfo {
                    document_id: r.get("document_id"),
                    external_id: r.get("external_id"),
                    filename: r.get("filename"),
                    chunk_count: r.get("chunk_count"),
                    metadata: serde_json::from_value(metadata).unwrap_or_default(),
                    created_at: r
                        .get::<Option<chrono::DateTime<chrono::Utc>>, _>("created_at")
                        .map(|dt| dt.to_rfc3339()),
                }
            })
            .collect())
    }

    async fn delete_document(
        &self,
        document_id: &str,
        app_id: &str,
    ) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM documents WHERE document_id = $1 AND app_id = $2",
        )
        .bind(document_id)
        .bind(app_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn get_document_by_external_id(
        &self,
        external_id: &str,
        app_id: &str,
    ) -> anyhow::Result<Option<DocumentInfo>> {
        let row = sqlx::query(
            "SELECT document_id, external_id, filename, chunk_count, metadata, created_at
             FROM documents
             WHERE external_id = $1 AND app_id = $2",
        )
        .bind(external_id)
        .bind(app_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| {
            let metadata: serde_json::Value = r.get("metadata");
            DocumentInfo {
                document_id: r.get("document_id"),
                external_id: r.get("external_id"),
                filename: r.get("filename"),
                chunk_count: r.get("chunk_count"),
                metadata: serde_json::from_value(metadata).unwrap_or_default(),
                created_at: r
                    .get::<Option<chrono::DateTime<chrono::Utc>>, _>("created_at")
                    .map(|dt| dt.to_rfc3339()),
            }
        }))
    }
}
