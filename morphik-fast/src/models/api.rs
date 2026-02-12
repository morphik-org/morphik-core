use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────── Ingest ────────────────────────────

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    /// Filename (for uploaded files).
    #[serde(default)]
    pub filename: Option<String>,
    /// Raw text content (alternative to file upload).
    #[serde(default)]
    pub content: Option<String>,
    /// Caller-provided metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional document rules.
    #[serde(default)]
    pub rules: Vec<serde_json::Value>,
    /// Whether to run ColPali embedding.
    #[serde(default = "default_true")]
    pub use_colpali: bool,
    /// Folder path for organisation.
    #[serde(default)]
    pub folder_name: Option<String>,
    /// Caller-supplied external ID override.
    #[serde(default)]
    pub external_id: Option<String>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub document_id: String,
    pub chunk_count: usize,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ──────────────────────────── Retrieve ────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RetrieveRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub k: usize,
    #[serde(default)]
    pub min_score: f64,
    #[serde(default)]
    pub document_ids: Option<Vec<String>>,
    #[serde(default)]
    pub folder_name: Option<String>,
    #[serde(default)]
    pub filters: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub use_colpali: bool,
    #[serde(default)]
    pub graph_name: Option<String>,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct RetrieveChunk {
    pub document_id: String,
    pub chunk_number: i32,
    pub content: String,
    pub score: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct RetrieveResponse {
    pub chunks: Vec<RetrieveChunk>,
}

// ──────────────────────────── Documents ────────────────────────────

#[derive(Debug, Serialize)]
pub struct DocumentInfo {
    pub document_id: String,
    pub external_id: Option<String>,
    pub filename: Option<String>,
    pub chunk_count: i32,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub success: bool,
    pub document_id: String,
}

// ──────────────────────────── Chat ────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_top_k")]
    pub k: usize,
    #[serde(default)]
    pub document_ids: Option<Vec<String>>,
    #[serde(default)]
    pub filters: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub completion: String,
    pub sources: Vec<RetrieveChunk>,
}

// ──────────────────────────── Auth ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthContext {
    pub user_id: String,
    #[serde(default)]
    pub app_id: Option<String>,
    #[serde(default)]
    pub permissions: Vec<String>,
}

// ──────────────────────────── Health ────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// ──────────────────────────── Store Metrics ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoreMetrics {
    #[serde(default)]
    pub chunk_payload_upload_s: f64,
    #[serde(default)]
    pub chunk_payload_objects: u64,
    #[serde(default)]
    pub chunk_payload_bytes: u64,
    #[serde(default)]
    pub chunk_payload_backend: String,
    #[serde(default)]
    pub multivector_upload_s: f64,
    #[serde(default)]
    pub multivector_objects: u64,
    #[serde(default)]
    pub multivector_bytes: u64,
    #[serde(default)]
    pub multivector_backend: String,
    #[serde(default)]
    pub vector_store_write_s: f64,
    #[serde(default)]
    pub vector_store_backend: String,
    #[serde(default)]
    pub vector_store_rows: u64,
    #[serde(default)]
    pub cache_write_s: f64,
    #[serde(default)]
    pub cache_write_objects: u64,
}
