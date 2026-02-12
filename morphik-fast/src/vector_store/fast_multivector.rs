use async_trait::async_trait;
use ndarray::Array1;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};

use crate::models::api::StoreMetrics;
use crate::models::chunk::DocumentChunk;
use crate::storage::Storage;
use crate::vector_store::cache::FileCacheManager;
use crate::vector_store::utils::*;
use crate::vector_store::VectorStore;

/// Configuration for Fixed Dimensional Encoding (FDE).
/// This compresses multi-vector embeddings into a fixed-size representation
/// for efficient ANN search on TurboPuffer.
#[derive(Debug, Clone)]
pub struct FdeConfig {
    pub dimension: usize,
    pub num_repetitions: usize,
    pub num_simhash_projections: usize,
    pub projection_dimension: usize,
}

impl Default for FdeConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            num_repetitions: 20,
            num_simhash_projections: 5,
            projection_dimension: 16,
        }
    }
}

/// Generate FDE encoding for a document embedding (simplified pure-Rust implementation).
/// In production, this would call the `fixed_dimensional_encoding` library.
/// For now, this implements a simhash-based projection scheme.
pub fn fde_encode_document(embedding: &[f32], config: &FdeConfig) -> Vec<f32> {
    let input = Array1::from_vec(embedding.to_vec());
    let input_dim = input.len();

    let total_output_dim = config.dimension * config.num_repetitions;
    let mut output = vec![0.0f32; total_output_dim];

    // Simple deterministic projection using hash-based approach.
    // Each repetition creates a projection of the input into `dimension` space.
    for rep in 0..config.num_repetitions {
        let offset = rep * config.dimension;
        for d in 0..config.dimension {
            let mut sum = 0.0f32;
            // Deterministic pseudo-random projection.
            for (i, &val) in embedding.iter().enumerate() {
                let seed = (rep * config.dimension * input_dim + d * input_dim + i) as u64;
                let sign = if hash_u64(seed) & 1 == 0 { 1.0 } else { -1.0 };
                sum += val * sign;
            }
            output[offset + d] = sum;
        }
    }

    output
}

/// Generate FDE encoding for a query embedding.
pub fn fde_encode_query(embedding: &[f32], config: &FdeConfig) -> Vec<f32> {
    // Query encoding uses the same projection but with different normalization.
    fde_encode_document(embedding, config)
}

/// Simple hash function for deterministic projections.
fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x517cc1b727220a95);
    x = (x >> 32) ^ x;
    x = x.wrapping_mul(0x6c62272e07bb0142);
    x = (x >> 32) ^ x;
    x
}

/// TurboPuffer client for multi-vector storage and ANN search.
/// Uses the TurboPuffer HTTP API.
pub struct TurboPufferClient {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    default_namespace: String,
}

impl TurboPufferClient {
    pub fn new(api_key: &str, _region: &str, default_namespace: &str) -> Self {
        let base_url = format!("https://api.turbopuffer.com/v1");

        Self {
            api_key: api_key.to_string(),
            base_url,
            http_client: reqwest::Client::new(),
            default_namespace: default_namespace.to_string(),
        }
    }

    fn namespace_url(&self, namespace: &str) -> String {
        let ns = if namespace.is_empty() {
            &self.default_namespace
        } else {
            namespace
        };
        format!("{}/namespaces/{ns}", self.base_url)
    }

    /// Write vectors to a TurboPuffer namespace.
    pub async fn write(
        &self,
        namespace: &str,
        data: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/vectors", self.namespace_url(namespace));
        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&data)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("TurboPuffer write failed ({status}): {body}");
        }

        Ok(resp.json().await?)
    }

    /// Query vectors from a TurboPuffer namespace.
    pub async fn query(
        &self,
        namespace: &str,
        data: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/vectors/query", self.namespace_url(namespace));
        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&data)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 404 {
                anyhow::bail!("TurboPuffer namespace not found");
            }
            anyhow::bail!("TurboPuffer query failed ({status}): {body}");
        }

        Ok(resp.json().await?)
    }

    /// Delete vectors by filter.
    pub async fn delete_by_filter(
        &self,
        namespace: &str,
        filter: serde_json::Value,
    ) -> anyhow::Result<()> {
        let url = format!("{}/vectors", self.namespace_url(namespace));
        let body = serde_json::json!({
            "delete_by_filter": filter
        });

        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            if status.as_u16() == 404 {
                info!("TurboPuffer namespace {namespace} not found during delete");
                return Ok(());
            }
            anyhow::bail!("TurboPuffer delete failed ({status}): {body_text}");
        }

        Ok(())
    }
}

/// FastMultiVectorStore: TurboPuffer-backed multi-vector store with FDE compression.
/// External storage always enabled for chunk payloads and multi-vector tensors.
pub struct FastMultiVectorStore {
    tpuf: TurboPufferClient,
    chunk_storage: Arc<dyn Storage>,
    vector_storage: Arc<dyn Storage>,
    chunk_bucket: String,
    vector_bucket: String,
    cache: FileCacheManager,
    fde_config: FdeConfig,
    db_pool: PgPool,
    document_app_id_cache: tokio::sync::Mutex<HashMap<String, String>>,
    upload_semaphore: Arc<Semaphore>,
}

const DEFAULT_APP_ID: &str = "default";

impl FastMultiVectorStore {
    pub async fn new(
        postgres_uri: &str,
        tpuf_api_key: &str,
        chunk_storage: Arc<dyn Storage>,
        vector_storage: Arc<dyn Storage>,
        chunk_bucket: &str,
        vector_bucket: &str,
        cache_enabled: bool,
        cache_path: &str,
        cache_max_bytes: u64,
        upload_concurrency: u32,
    ) -> anyhow::Result<Self> {
        let clean_uri = postgres_uri.replace("postgresql+asyncpg://", "postgresql://");
        let db_pool = PgPoolOptions::new()
            .max_connections(10)
            .acquire_timeout(std::time::Duration::from_secs(60))
            .connect(&clean_uri)
            .await?;

        let tpuf = TurboPufferClient::new(tpuf_api_key, "aws-us-west-2", "default2");

        let cache = FileCacheManager::new(
            cache_enabled,
            std::path::PathBuf::from(cache_path),
            cache_max_bytes,
        );

        Ok(Self {
            tpuf,
            chunk_storage,
            vector_storage,
            chunk_bucket: chunk_bucket.to_string(),
            vector_bucket: vector_bucket.to_string(),
            cache,
            fde_config: FdeConfig::default(),
            db_pool,
            document_app_id_cache: tokio::sync::Mutex::new(HashMap::new()),
            upload_semaphore: Arc::new(Semaphore::new(upload_concurrency.max(1) as usize)),
        })
    }

    /// Get app_id for a document from the database, with caching.
    async fn get_document_app_id(&self, document_id: &str) -> String {
        {
            let cache = self.document_app_id_cache.lock().await;
            if let Some(id) = cache.get(document_id) {
                return id.clone();
            }
        }

        let result: Option<String> = sqlx::query_scalar(
            "SELECT app_id FROM documents WHERE external_id = $1",
        )
        .bind(document_id)
        .fetch_optional(&self.db_pool)
        .await
        .ok()
        .flatten();

        let app_id = result.unwrap_or_else(|| DEFAULT_APP_ID.to_string());
        {
            let mut cache = self.document_app_id_cache.lock().await;
            cache.insert(document_id.to_string(), app_id.clone());
        }
        app_id
    }

    /// Determine file extension based on content and metadata.
    fn determine_file_extension(content: &str, metadata: &HashMap<String, serde_json::Value>) -> &'static str {
        let is_image = metadata
            .get("is_image")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if is_image {
            // For images, auto-detect from content.
            use base64::Engine;
            if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(content.as_bytes()) {
                return crate::storage::detect_file_extension(&decoded);
            }
            ".png"
        } else {
            ".txt"
        }
    }

    /// Store chunk content in external storage. Returns (storage_key, bytes_stored).
    async fn store_content_externally(
        &self,
        content: &str,
        document_id: &str,
        chunk_number: i32,
        metadata: &HashMap<String, serde_json::Value>,
        app_id: &str,
    ) -> anyhow::Result<(String, u64)> {
        let extension = Self::determine_file_extension(content, metadata);
        let storage_key = format!("{app_id}/{document_id}/{chunk_number}{extension}");

        if extension == ".txt" {
            use base64::Engine;
            let content_bytes = content.as_bytes();
            let content_b64 = base64::engine::general_purpose::STANDARD.encode(content_bytes);
            self.chunk_storage
                .upload_from_base64(&content_b64, &storage_key, Some("text/plain"), &self.chunk_bucket)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to upload chunk: {e}"))?;
        } else {
            self.chunk_storage
                .upload_from_base64(content, &storage_key, None, &self.chunk_bucket)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to upload chunk: {e}"))?;
        }

        let size = self
            .chunk_storage
            .get_object_size(&self.chunk_bucket, &storage_key)
            .await
            .unwrap_or(0);

        Ok((storage_key, size))
    }

    /// Save multi-vector tensor to external storage and cache.
    async fn save_multivector_to_storage(
        &self,
        document_id: &str,
        chunk_number: i32,
        embedding: &[f32],
    ) -> anyhow::Result<(String, String, f64, u64)> {
        let save_path = format!("multivector/{document_id}/{chunk_number}.npy");

        // Serialize embedding as raw f32 bytes (simplified .npy format).
        let npy_bytes = serialize_f32_array(embedding);

        self.vector_storage
            .upload_bytes(&npy_bytes, &save_path, Some("application/octet-stream"), &self.vector_bucket)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to upload multivector: {e}"))?;

        let stored_size = self
            .vector_storage
            .get_object_size(&self.vector_bucket, &save_path)
            .await
            .unwrap_or(0);

        // Cache on ingest.
        let cache_start = Instant::now();
        self.cache
            .put("vectors", &self.vector_bucket, &save_path, &npy_bytes)
            .await;
        let cache_write_time = cache_start.elapsed().as_secs_f64();

        Ok((
            self.vector_bucket.clone(),
            save_path,
            cache_write_time,
            stored_size,
        ))
    }

    /// Load multi-vector tensor from cache or storage.
    async fn load_multivector_from_storage(&self, bucket: &str, key: &str) -> anyhow::Result<Vec<f32>> {
        // Try cache first.
        if let Some(cached) = self.cache.get("vectors", bucket, key).await {
            return deserialize_f32_array(&cached);
        }

        // Download from storage.
        let content = self
            .vector_storage
            .download_file(bucket, key)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to download multivector: {e}"))?;

        // Cache for future use.
        self.cache.put("vectors", bucket, key, &content).await;

        deserialize_f32_array(&content)
    }

    /// Retrieve content from external storage.
    async fn retrieve_content_from_storage(
        &self,
        storage_key: &str,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> String {
        let bucket = &self.chunk_bucket;
        match self.chunk_storage.download_file(bucket, storage_key).await {
            Ok(content_bytes) => {
                if storage_key.ends_with(".txt") {
                    String::from_utf8(content_bytes).unwrap_or_else(|_| storage_key.to_string())
                } else {
                    let is_image = metadata
                        .get("is_image")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if is_image {
                        use base64::Engine;
                        let mime = detect_image_mime(&content_bytes);
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&content_bytes);
                        format!("data:{mime};base64,{b64}")
                    } else {
                        String::from_utf8(content_bytes).unwrap_or_else(|_| storage_key.to_string())
                    }
                }
            }
            Err(e) => {
                error!("Failed to retrieve content from {storage_key}: {e}");
                storage_key.to_string()
            }
        }
    }

    /// Compute multi-vector similarity score.
    /// Simplified scoring: dot product between query embedding and stored multi-vector.
    fn score_multivector(query: &[f32], multivector: &[f32]) -> f64 {
        if query.is_empty() || multivector.is_empty() {
            return 0.0;
        }
        // MaxSim: for each query token, find max similarity across document tokens.
        // Simplified as dot product of flattened vectors.
        let min_len = query.len().min(multivector.len());
        let dot: f64 = query[..min_len]
            .iter()
            .zip(multivector[..min_len].iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        let q_norm: f64 = query.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let d_norm: f64 = multivector.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();

        if q_norm == 0.0 || d_norm == 0.0 {
            0.0
        } else {
            dot / (q_norm * d_norm)
        }
    }
}

#[async_trait]
impl VectorStore for FastMultiVectorStore {
    async fn initialize(&self) -> anyhow::Result<bool> {
        Ok(true)
    }

    async fn store_embeddings(
        &self,
        chunks: &[DocumentChunk],
        app_id: Option<&str>,
    ) -> anyhow::Result<(bool, Vec<String>, StoreMetrics)> {
        if chunks.is_empty() {
            return Ok((
                true,
                vec![],
                build_store_metrics(
                    self.chunk_storage.provider_name(),
                    self.vector_storage.provider_name(),
                    "turbopuffer",
                ),
            ));
        }

        let resolved_app_id = match app_id {
            Some(id) => id.to_string(),
            None => {
                let doc_id = &chunks[0].document_id;
                self.get_document_app_id(doc_id).await
            }
        };

        // FDE encode all embeddings.
        let embeddings: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| fde_encode_document(&c.embedding, &self.fde_config))
            .collect();

        let mut metrics = build_store_metrics(
            self.chunk_storage.provider_name(),
            self.vector_storage.provider_name(),
            "turbopuffer",
        );
        metrics.vector_store_rows = chunks.len() as u64;

        // Upload chunk payloads to storage.
        let payload_start = Instant::now();
        let mut storage_keys = Vec::with_capacity(chunks.len());
        let mut total_payload_bytes: u64 = 0;

        for chunk in chunks {
            match self
                .store_content_externally(
                    &chunk.content,
                    &chunk.document_id,
                    chunk.chunk_number,
                    &chunk.metadata,
                    &resolved_app_id,
                )
                .await
            {
                Ok((key, bytes)) => {
                    storage_keys.push(key);
                    total_payload_bytes += bytes;
                }
                Err(e) => {
                    warn!("Failed to store chunk externally: {e}");
                    storage_keys.push(String::new());
                }
            }
        }
        metrics.chunk_payload_upload_s = payload_start.elapsed().as_secs_f64();
        metrics.chunk_payload_objects = storage_keys.iter().filter(|k| !k.is_empty()).count() as u64;
        metrics.chunk_payload_bytes = total_payload_bytes;

        // Upload multi-vector tensors.
        let mv_start = Instant::now();
        let mut multivecs: Vec<(String, String)> = Vec::with_capacity(chunks.len());
        let mut total_mv_bytes: u64 = 0;
        let mut total_cache_write_s: f64 = 0.0;

        for chunk in chunks {
            let _permit = self.upload_semaphore.acquire().await?;
            match self
                .save_multivector_to_storage(
                    &chunk.document_id,
                    chunk.chunk_number,
                    &chunk.embedding,
                )
                .await
            {
                Ok((bucket, key, cache_s, size)) => {
                    multivecs.push((bucket, key));
                    total_mv_bytes += size;
                    total_cache_write_s += cache_s;
                }
                Err(e) => {
                    warn!("Failed to save multivector: {e}");
                    multivecs.push((String::new(), String::new()));
                }
            }
        }
        metrics.multivector_upload_s = mv_start.elapsed().as_secs_f64();
        metrics.multivector_objects = multivecs.len() as u64;
        metrics.multivector_bytes = total_mv_bytes;
        metrics.cache_write_s = total_cache_write_s;
        metrics.cache_write_objects = multivecs.len() as u64;

        // Build stored IDs and write to TurboPuffer.
        let stored_ids: Vec<String> = chunks
            .iter()
            .map(|c| format!("{}-{}", c.document_id, c.chunk_number))
            .collect();

        let doc_ids: Vec<&str> = chunks.iter().map(|c| c.document_id.as_str()).collect();
        let chunk_numbers: Vec<i32> = chunks.iter().map(|c| c.chunk_number).collect();
        let metadatas: Vec<String> = chunks
            .iter()
            .map(|c| serde_json::to_string(&c.metadata).unwrap_or_default())
            .collect();
        let mv_tuples: Vec<(&str, &str)> = multivecs
            .iter()
            .map(|(b, k)| (b.as_str(), k.as_str()))
            .collect();

        let write_start = Instant::now();
        let write_body = serde_json::json!({
            "upsert_columns": {
                "id": stored_ids,
                "vector": embeddings,
                "document_id": doc_ids,
                "chunk_number": chunk_numbers,
                "content": storage_keys,
                "metadata": metadatas,
                "multivector": mv_tuples,
            },
            "distance_metric": "cosine_distance",
        });

        self.tpuf.write(&resolved_app_id, write_body).await?;
        metrics.vector_store_write_s = write_start.elapsed().as_secs_f64();

        debug!("Stored {} chunks in TurboPuffer", chunks.len());
        Ok((true, stored_ids, metrics))
    }

    async fn query_similar(
        &self,
        query_embedding: &[f32],
        k: usize,
        doc_ids: Option<&[String]>,
        app_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        let t0 = Instant::now();

        // 1) Encode query embedding.
        let encoded_query = fde_encode_query(query_embedding, &self.fde_config);
        let t1 = Instant::now();
        info!(
            "query_similar timing - encode_query: {:.2} ms",
            t1.duration_since(t0).as_secs_f64() * 1000.0
        );

        // 2) ANN search on TurboPuffer.
        let namespace = app_id.unwrap_or(DEFAULT_APP_ID);
        let top_k = (10 * k).min(75);

        let query_body = serde_json::json!({
            "filters": doc_ids.map(|ids| serde_json::json!(["document_id", "In", ids])),
            "rank_by": ["vector", "ANN", encoded_query],
            "top_k": top_k,
            "include_attributes": ["id", "document_id", "chunk_number", "content", "metadata", "multivector"],
            "consistency": {"level": "eventual"},
        });

        let result = self.tpuf.query(namespace, query_body).await?;
        let t2 = Instant::now();
        info!(
            "query_similar timing - ns.query: {:.2} ms",
            t2.duration_since(t1).as_secs_f64() * 1000.0
        );

        // 3) Parse results and download multi-vectors.
        let rows = result
            .get("rows")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if rows.is_empty() {
            info!("query_similar: no rows returned");
            return Ok(vec![]);
        }

        let mut multivectors = Vec::with_capacity(rows.len());
        for row in &rows {
            let mv = row.get("multivector").and_then(|v| v.as_array());
            if let Some(mv_arr) = mv {
                if mv_arr.len() == 2 {
                    let bucket = mv_arr[0].as_str().unwrap_or("");
                    let key = mv_arr[1].as_str().unwrap_or("");
                    match self.load_multivector_from_storage(bucket, key).await {
                        Ok(v) => multivectors.push(v),
                        Err(e) => {
                            warn!("Failed to load multivector: {e}");
                            multivectors.push(vec![]);
                        }
                    }
                } else {
                    multivectors.push(vec![]);
                }
            } else {
                multivectors.push(vec![]);
            }
        }
        let t3 = Instant::now();
        info!(
            "query_similar timing - load_multivectors: {:.2} ms",
            t3.duration_since(t2).as_secs_f64() * 1000.0
        );

        // 4) Rerank using multi-vector scoring.
        let mut scored: Vec<(usize, f64)> = multivectors
            .iter()
            .enumerate()
            .map(|(i, mv)| (i, Self::score_multivector(query_embedding, mv)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let t4 = Instant::now();
        info!(
            "query_similar timing - rerank_scoring: {:.2} ms",
            t4.duration_since(t3).as_secs_f64() * 1000.0
        );

        // 5) Retrieve chunk contents and build results.
        let mut result_chunks = Vec::with_capacity(scored.len());
        for (idx, score) in &scored {
            let row = &rows[*idx];
            let document_id = row
                .get("document_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let chunk_number = row
                .get("chunk_number")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;
            let storage_key = row
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let metadata_str = row
                .get("metadata")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(metadata_str).unwrap_or_default();

            let content = if is_storage_key(storage_key) {
                self.retrieve_content_from_storage(storage_key, &metadata)
                    .await
            } else {
                storage_key.to_string()
            };

            result_chunks.push(DocumentChunk {
                document_id,
                embedding: vec![],
                chunk_number,
                content,
                metadata,
                score: *score,
            });
        }

        let t5 = Instant::now();
        info!(
            "query_similar total time: {:.2} ms",
            t5.duration_since(t0).as_secs_f64() * 1000.0
        );

        Ok(result_chunks)
    }

    async fn get_chunks_by_id(
        &self,
        chunk_identifiers: &[(String, i32)],
        app_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        if chunk_identifiers.is_empty() {
            return Ok(vec![]);
        }

        let namespace = app_id.unwrap_or(DEFAULT_APP_ID);
        let ids: Vec<String> = chunk_identifiers
            .iter()
            .map(|(doc_id, num)| format!("{doc_id}-{num}"))
            .collect();

        let query_body = serde_json::json!({
            "filters": ["id", "In", ids],
            "include_attributes": ["id", "document_id", "chunk_number", "content", "metadata"],
            "top_k": chunk_identifiers.len(),
        });

        let result = self.tpuf.query(namespace, query_body).await?;
        let rows = result
            .get("rows")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut chunks = Vec::with_capacity(rows.len());
        for row in &rows {
            let document_id = row
                .get("document_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let chunk_number = row
                .get("chunk_number")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;
            let storage_key = row
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let metadata_str = row
                .get("metadata")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(metadata_str).unwrap_or_default();

            let content = if is_storage_key(storage_key) {
                self.retrieve_content_from_storage(storage_key, &metadata)
                    .await
            } else {
                storage_key.to_string()
            };

            chunks.push(DocumentChunk {
                document_id,
                embedding: vec![],
                chunk_number,
                content,
                metadata,
                score: 0.0,
            });
        }

        Ok(chunks)
    }

    async fn delete_chunks_by_document_id(
        &self,
        document_id: &str,
        app_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        let namespace = app_id.unwrap_or(DEFAULT_APP_ID);

        let filter = serde_json::json!(["document_id", "Eq", document_id]);
        self.tpuf.delete_by_filter(namespace, filter).await?;

        info!("Deleted chunks for document {document_id} from TurboPuffer");
        Ok(true)
    }
}

/// Serialize a f32 slice to a simple binary format.
/// Format: 4 bytes length (little-endian u32) + raw f32 bytes.
fn serialize_f32_array(data: &[f32]) -> Vec<u8> {
    let len = data.len() as u32;
    let mut bytes = Vec::with_capacity(4 + data.len() * 4);
    bytes.extend_from_slice(&len.to_le_bytes());
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize a f32 array from binary format.
fn deserialize_f32_array(data: &[u8]) -> anyhow::Result<Vec<f32>> {
    if data.len() < 4 {
        anyhow::bail!("Data too short to deserialize f32 array");
    }
    let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let expected_size = 4 + len * 4;
    if data.len() < expected_size {
        anyhow::bail!(
            "Data length {} too short for {} floats (expected {})",
            data.len(),
            len,
            expected_size
        );
    }
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let offset = 4 + i * 4;
        let val = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        result.push(val);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fde_encode_document() {
        let embedding = vec![1.0f32, 2.0, 3.0, 4.0];
        let config = FdeConfig::default();
        let encoded = fde_encode_document(&embedding, &config);
        assert_eq!(encoded.len(), config.dimension * config.num_repetitions);
    }

    #[test]
    fn test_fde_deterministic() {
        let embedding = vec![1.0f32, 2.0, 3.0, 4.0];
        let config = FdeConfig::default();
        let a = fde_encode_document(&embedding, &config);
        let b = fde_encode_document(&embedding, &config);
        assert_eq!(a, b);
    }

    #[test]
    fn test_serialize_deserialize_f32() {
        let data = vec![1.0f32, 2.5, -3.14, 0.0, 100.0];
        let bytes = serialize_f32_array(&data);
        let restored = deserialize_f32_array(&bytes).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_serialize_empty() {
        let data: Vec<f32> = vec![];
        let bytes = serialize_f32_array(&data);
        let restored = deserialize_f32_array(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_score_multivector() {
        let query = vec![1.0f32, 0.0, 0.0];
        let doc = vec![1.0f32, 0.0, 0.0];
        let score = FastMultiVectorStore::score_multivector(&query, &doc);
        assert!((score - 1.0).abs() < 1e-6);

        let orthogonal = vec![0.0f32, 1.0, 0.0];
        let score2 = FastMultiVectorStore::score_multivector(&query, &orthogonal);
        assert!(score2.abs() < 1e-6);
    }

    #[test]
    fn test_hash_u64() {
        // Should be deterministic.
        assert_eq!(hash_u64(42), hash_u64(42));
        // Different inputs should give different outputs.
        assert_ne!(hash_u64(1), hash_u64(2));
    }
}
