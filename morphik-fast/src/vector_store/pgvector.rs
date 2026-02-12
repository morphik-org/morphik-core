use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use std::time::Instant;
use tracing::info;

use crate::models::api::StoreMetrics;
use crate::models::chunk::DocumentChunk;
use crate::vector_store::utils::build_store_metrics;
use crate::vector_store::VectorStore;

/// PostgreSQL with pgvector implementation for vector storage.
pub struct PGVectorStore {
    pool: PgPool,
    ivfflat_probes: u32,
    dimensions: u32,
}

impl PGVectorStore {
    pub async fn new(uri: &str, pool_size: u32, ivfflat_probes: u32, dimensions: u32) -> anyhow::Result<Self> {
        // Strip parameters incompatible with native driver.
        let clean_uri = uri
            .replace("postgresql+asyncpg://", "postgresql://");

        let pool = PgPoolOptions::new()
            .max_connections(pool_size)
            .acquire_timeout(std::time::Duration::from_secs(10))
            .connect(&clean_uri)
            .await?;

        info!("Created PGVector store connection pool (size={pool_size})");

        Ok(Self {
            pool,
            ivfflat_probes,
            dimensions,
        })
    }
}

#[async_trait]
impl VectorStore for PGVectorStore {
    async fn initialize(&self) -> anyhow::Result<bool> {
        // Enable pgvector extension.
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await?;
        info!("Enabled pgvector extension");

        // Check if table exists.
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'vector_embeddings')",
        )
        .fetch_one(&self.pool)
        .await?;

        if !exists {
            let create_sql = format!(
                "CREATE TABLE vector_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    chunk_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    chunk_metadata TEXT,
                    embedding vector({}) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )",
                self.dimensions
            );
            sqlx::query(&create_sql).execute(&self.pool).await?;
            info!(
                "Created vector_embeddings table with vector({})",
                self.dimensions
            );

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_document_id ON vector_embeddings(document_id)")
                .execute(&self.pool)
                .await?;

            sqlx::query(
                "CREATE INDEX IF NOT EXISTS vector_idx ON vector_embeddings
                 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
            )
            .execute(&self.pool)
            .await?;
            info!("Created IVFFlat index on vector_embeddings");
        } else {
            info!("vector_embeddings table already exists");
        }

        Ok(true)
    }

    async fn store_embeddings(
        &self,
        chunks: &[DocumentChunk],
        _app_id: Option<&str>,
    ) -> anyhow::Result<(bool, Vec<String>, StoreMetrics)> {
        if chunks.is_empty() {
            return Ok((true, vec![], build_store_metrics("none", "none", "pgvector")));
        }

        let start = Instant::now();
        let mut stored_ids = Vec::with_capacity(chunks.len());
        let mut payload_bytes: u64 = 0;

        // Batch insert using a transaction.
        let mut tx = self.pool.begin().await?;

        for chunk in chunks {
            if chunk.embedding.is_empty() {
                continue;
            }
            let metadata_json = serde_json::to_string(&chunk.metadata).unwrap_or_default();
            let embedding_str = format!(
                "[{}]",
                chunk
                    .embedding
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            payload_bytes += chunk.content.len() as u64;

            sqlx::query(
                "INSERT INTO vector_embeddings (document_id, chunk_number, content, chunk_metadata, embedding)
                 VALUES ($1, $2, $3, $4, $5::vector)",
            )
            .bind(&chunk.document_id)
            .bind(chunk.chunk_number)
            .bind(&chunk.content)
            .bind(&metadata_json)
            .bind(&embedding_str)
            .execute(&mut *tx)
            .await?;

            stored_ids.push(format!("{}-{}", chunk.document_id, chunk.chunk_number));
        }

        tx.commit().await?;
        let write_duration = start.elapsed().as_secs_f64();

        let mut metrics = build_store_metrics("none", "none", "pgvector");
        metrics.chunk_payload_bytes = payload_bytes;
        metrics.vector_store_write_s = write_duration;
        metrics.vector_store_rows = stored_ids.len() as u64;

        Ok((true, stored_ids, metrics))
    }

    async fn query_similar(
        &self,
        query_embedding: &[f32],
        k: usize,
        doc_ids: Option<&[String]>,
        _app_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        // Set ivfflat probes.
        sqlx::query(&format!(
            "SET LOCAL ivfflat.probes = {}",
            self.ivfflat_probes
        ))
        .execute(&self.pool)
        .await
        .ok();

        let embedding_str = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let rows = if let Some(ids) = doc_ids {
            if ids.is_empty() {
                return Ok(vec![]);
            }
            sqlx::query(
                "SELECT document_id, chunk_number, content, chunk_metadata,
                        embedding <=> $1::vector AS distance
                 FROM vector_embeddings
                 WHERE document_id = ANY($2)
                 ORDER BY distance
                 LIMIT $3",
            )
            .bind(&embedding_str)
            .bind(ids)
            .bind(k as i64)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT document_id, chunk_number, content, chunk_metadata,
                        embedding <=> $1::vector AS distance
                 FROM vector_embeddings
                 ORDER BY distance
                 LIMIT $2",
            )
            .bind(&embedding_str)
            .bind(k as i64)
            .fetch_all(&self.pool)
            .await?
        };

        let mut chunks = Vec::with_capacity(rows.len());
        for row in rows {
            let document_id: String = row.get("document_id");
            let chunk_number: i32 = row.get("chunk_number");
            let content: String = row.get("content");
            let chunk_metadata: Option<String> = row.get("chunk_metadata");
            let distance: f64 = row.get("distance");

            let metadata = chunk_metadata
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();

            chunks.push(DocumentChunk {
                document_id,
                chunk_number,
                content,
                embedding: vec![],
                metadata,
                score: 1.0 - distance / 2.0,
            });
        }

        Ok(chunks)
    }

    async fn get_chunks_by_id(
        &self,
        chunk_identifiers: &[(String, i32)],
        _app_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        if chunk_identifiers.is_empty() {
            return Ok(vec![]);
        }

        // Build a filter for (document_id, chunk_number) pairs.
        let mut doc_ids = Vec::new();
        let mut chunk_nums = Vec::new();
        for (doc_id, num) in chunk_identifiers {
            doc_ids.push(doc_id.clone());
            chunk_nums.push(*num);
        }

        let rows = sqlx::query(
            "SELECT document_id, chunk_number, content, chunk_metadata
             FROM vector_embeddings
             WHERE (document_id, chunk_number) IN (
                 SELECT UNNEST($1::text[]), UNNEST($2::int[])
             )",
        )
        .bind(&doc_ids)
        .bind(&chunk_nums)
        .fetch_all(&self.pool)
        .await?;

        let mut chunks = Vec::with_capacity(rows.len());
        for row in rows {
            let document_id: String = row.get("document_id");
            let chunk_number: i32 = row.get("chunk_number");
            let content: String = row.get("content");
            let chunk_metadata: Option<String> = row.get("chunk_metadata");

            let metadata = chunk_metadata
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();

            chunks.push(DocumentChunk {
                document_id,
                chunk_number,
                content,
                embedding: vec![],
                metadata,
                score: 0.0,
            });
        }

        Ok(chunks)
    }

    async fn delete_chunks_by_document_id(
        &self,
        document_id: &str,
        _app_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        sqlx::query("DELETE FROM vector_embeddings WHERE document_id = $1")
            .bind(document_id)
            .execute(&self.pool)
            .await?;
        info!("Deleted all chunks for document {document_id}");
        Ok(true)
    }
}
