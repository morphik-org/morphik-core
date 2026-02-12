use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

/// Global settings singleton.
static SETTINGS: OnceLock<Settings> = OnceLock::new();

// ──────────────────────────── TOML structure ────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct TomlConfig {
    pub api: ApiConfig,
    #[serde(default)]
    pub service: ServiceConfig,
    pub auth: AuthConfig,
    #[serde(default)]
    pub registered_models: HashMap<String, HashMap<String, toml::Value>>,
    pub completion: CompletionConfig,
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub parser: ParserConfig,
    #[serde(default)]
    pub document_analysis: Option<DocumentAnalysisConfig>,
    pub reranker: RerankerConfig,
    pub storage: StorageConfig,
    pub vector_store: VectorStoreConfig,
    #[serde(default)]
    pub multivector_store: MultivectorStoreConfig,
    #[serde(default)]
    pub redis: RedisConfig,
    #[serde(default)]
    pub worker: WorkerConfig,
    #[serde(default)]
    pub pdf: PdfConfig,
    pub morphik: MorphikConfig,
    #[serde(default)]
    pub pdf_viewer: Option<PdfViewerConfig>,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub reload: bool,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ServiceConfig {
    #[serde(default = "default_environment")]
    pub environment: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub enable_profiling: bool,
}

fn default_environment() -> String {
    "development".to_string()
}
fn default_version() -> String {
    "unknown".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct AuthConfig {
    pub jwt_algorithm: String,
    #[serde(default)]
    pub bypass_auth_mode: bool,
    #[serde(default = "default_dev_user_id")]
    pub dev_user_id: String,
}

fn default_dev_user_id() -> String {
    "dev_user".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct CompletionConfig {
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub default_max_tokens: String,
    #[serde(default = "default_temperature")]
    pub default_temperature: f64,
}

fn default_max_tokens() -> String {
    "1000".to_string()
}
fn default_temperature() -> f64 {
    0.3
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub provider: String,
    #[serde(default = "default_pool_size")]
    pub pool_size: u32,
    #[serde(default = "default_max_overflow")]
    pub max_overflow: u32,
    #[serde(default = "default_pool_recycle")]
    pub pool_recycle: u64,
    #[serde(default = "default_pool_timeout")]
    pub pool_timeout: u64,
    #[serde(default = "default_true")]
    pub pool_pre_ping: bool,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_retry_delay")]
    pub retry_delay: f64,
}

fn default_pool_size() -> u32 {
    10
}
fn default_max_overflow() -> u32 {
    15
}
fn default_pool_recycle() -> u64 {
    3600
}
fn default_pool_timeout() -> u64 {
    10
}
fn default_true() -> bool {
    true
}
fn default_max_retries() -> u32 {
    3
}
fn default_retry_delay() -> f64 {
    1.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimensions: u32,
    pub similarity_metric: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ParserConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    #[serde(default)]
    pub use_contextual_chunking: bool,
    #[serde(default)]
    pub contextual_chunking_model: Option<String>,
    #[serde(default)]
    pub xml: Option<ParserXmlConfig>,
    #[serde(default)]
    pub vision: Option<ParserVisionConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ParserXmlConfig {
    #[serde(default = "default_xml_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub preferred_unit_tags: Vec<String>,
    #[serde(default)]
    pub ignore_tags: Vec<String>,
}

fn default_xml_max_tokens() -> u32 {
    350
}

#[derive(Debug, Deserialize, Clone)]
pub struct ParserVisionConfig {
    pub model: String,
    #[serde(default = "default_frame_sample_rate")]
    pub frame_sample_rate: i32,
}

fn default_frame_sample_rate() -> i32 {
    -1
}

#[derive(Debug, Deserialize, Clone)]
pub struct DocumentAnalysisConfig {
    pub model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RerankerConfig {
    pub use_reranker: bool,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub query_max_length: Option<u32>,
    #[serde(default)]
    pub passage_max_length: Option<u32>,
    #[serde(default)]
    pub use_fp16: Option<bool>,
    #[serde(default)]
    pub device: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StorageConfig {
    pub provider: String,
    #[serde(default)]
    pub storage_path: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub bucket_name: Option<String>,
    #[serde(default)]
    pub cache_enabled: bool,
    #[serde(default = "default_cache_max_size_gb")]
    pub cache_max_size_gb: u32,
    #[serde(default)]
    pub cache_path: Option<String>,
    #[serde(default = "default_s3_upload_concurrency")]
    pub s3_upload_concurrency: u32,
}

fn default_cache_max_size_gb() -> u32 {
    10
}
fn default_s3_upload_concurrency() -> u32 {
    16
}

#[derive(Debug, Deserialize, Clone)]
pub struct VectorStoreConfig {
    pub provider: String,
    #[serde(default = "default_ivfflat_probes")]
    pub ivfflat_probes: Option<u32>,
}

fn default_ivfflat_probes() -> Option<u32> {
    Some(100)
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct MultivectorStoreConfig {
    #[serde(default = "default_multivector_provider")]
    pub provider: String,
}

fn default_multivector_provider() -> String {
    "postgres".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct RedisConfig {
    #[serde(default = "default_redis_url")]
    pub url: String,
    #[serde(default = "default_redis_host")]
    pub host: String,
    #[serde(default = "default_redis_port")]
    pub port: u16,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: default_redis_url(),
            host: default_redis_host(),
            port: default_redis_port(),
        }
    }
}

fn default_redis_url() -> String {
    "redis://localhost:6379/0".to_string()
}
fn default_redis_host() -> String {
    "localhost".to_string()
}
fn default_redis_port() -> u16 {
    6379
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct WorkerConfig {
    #[serde(default = "default_arq_max_jobs")]
    pub arq_max_jobs: u32,
    #[serde(default = "default_colpali_store_batch_size")]
    pub colpali_store_batch_size: u32,
}

fn default_arq_max_jobs() -> u32 {
    1
}
fn default_colpali_store_batch_size() -> u32 {
    16
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct PdfConfig {
    #[serde(default = "default_colpali_pdf_dpi")]
    pub colpali_pdf_dpi: u32,
}

fn default_colpali_pdf_dpi() -> u32 {
    150
}

#[derive(Debug, Deserialize, Clone)]
pub struct MorphikConfig {
    pub enable_colpali: bool,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_secret_manager")]
    pub secret_manager: String,
    #[serde(default = "default_api_domain")]
    pub api_domain: String,
    #[serde(default)]
    pub morphik_embedding_api_domain: Vec<String>,
    #[serde(default = "default_colpali_mode")]
    pub colpali_mode: String,
    #[serde(default = "default_parser_mode")]
    pub parser_mode: String,
}

fn default_mode() -> String {
    "self_hosted".to_string()
}
fn default_secret_manager() -> String {
    "env".to_string()
}
fn default_api_domain() -> String {
    "api.morphik.ai".to_string()
}
fn default_colpali_mode() -> String {
    "local".to_string()
}
fn default_parser_mode() -> String {
    "local".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct PdfViewerConfig {
    #[serde(default = "default_frontend_url")]
    pub frontend_url: String,
}

fn default_frontend_url() -> String {
    "http://localhost:3000/api/pdf".to_string()
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TelemetryConfig {
    #[serde(default = "default_service_name")]
    pub service_name: String,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default = "default_upload_interval_hours")]
    pub upload_interval_hours: f64,
    #[serde(default = "default_max_local_bytes")]
    pub max_local_bytes: u64,
}

fn default_service_name() -> String {
    "databridge-core".to_string()
}
fn default_upload_interval_hours() -> f64 {
    4.0
}
fn default_max_local_bytes() -> u64 {
    1_073_741_824
}

// ──────────────────────────── Resolved Settings ────────────────────────────

/// Flat settings structure resolved from TOML + environment variables.
#[derive(Debug, Clone)]
pub struct Settings {
    // API
    pub host: String,
    pub port: u16,

    // Service
    pub environment: String,

    // Auth
    pub jwt_algorithm: String,
    pub jwt_secret_key: String,
    pub bypass_auth_mode: bool,
    pub dev_user_id: String,

    // Registered models
    pub registered_models: HashMap<String, HashMap<String, toml::Value>>,

    // Completion
    pub completion_model: String,
    pub default_max_tokens: u32,
    pub default_temperature: f64,

    // Database
    pub postgres_uri: String,
    pub db_pool_size: u32,

    // Embedding
    pub embedding_model: String,
    pub vector_dimensions: u32,

    // Parser
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub parser_mode: String,

    // Storage
    pub storage_provider: String,
    pub storage_path: String,
    pub aws_region: Option<String>,
    pub s3_bucket: Option<String>,
    pub aws_access_key: Option<String>,
    pub aws_secret_access_key: Option<String>,
    pub s3_upload_concurrency: u32,
    pub cache_enabled: bool,
    pub cache_max_bytes: u64,
    pub cache_path: String,

    // Vector store
    pub vector_store_provider: String,
    pub ivfflat_probes: u32,

    // Multivector store
    pub multivector_store_provider: String,
    pub turbopuffer_api_key: Option<String>,

    // ColPali
    pub enable_colpali: bool,
    pub colpali_mode: String,
    pub morphik_embedding_api_domain: Vec<String>,

    // Mode
    pub mode: String,

    // Redis
    pub redis_url: String,

    // Document analysis
    pub document_analysis_model: Option<String>,
}

impl Settings {
    /// Resolve a model name from the registered_models table.
    pub fn resolve_model_name(&self, key: &str) -> Option<String> {
        self.registered_models
            .get(key)
            .and_then(|m| m.get("model_name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

/// Load and cache settings from morphik.toml and env.
pub fn get_settings() -> &'static Settings {
    SETTINGS.get_or_init(|| load_settings_from_path("morphik.toml").expect("Failed to load settings"))
}

/// Load settings from a given TOML path. Useful for testing.
pub fn load_settings_from_path(path: impl AsRef<Path>) -> anyhow::Result<Settings> {
    // Load .env if present (ignore errors)
    let _ = dotenvy::dotenv();

    let content = std::fs::read_to_string(path.as_ref())?;
    let config: TomlConfig = toml::from_str(&content)?;

    let jwt_secret_key = std::env::var("JWT_SECRET_KEY").unwrap_or_else(|_| "dev-secret-key".to_string());

    if !config.auth.bypass_auth_mode && jwt_secret_key == "dev-secret-key" {
        anyhow::bail!("JWT_SECRET_KEY is required when bypass_auth_mode is disabled");
    }

    let postgres_uri = std::env::var("POSTGRES_URI")
        .map_err(|_| anyhow::anyhow!("POSTGRES_URI environment variable is required"))?;

    // Storage
    let (aws_access_key, aws_secret_access_key, aws_region, s3_bucket) =
        if config.storage.provider == "aws-s3" {
            let ak = std::env::var("AWS_ACCESS_KEY")
                .map_err(|_| anyhow::anyhow!("AWS_ACCESS_KEY required for aws-s3 provider"))?;
            let sk = std::env::var("AWS_SECRET_ACCESS_KEY")
                .map_err(|_| anyhow::anyhow!("AWS_SECRET_ACCESS_KEY required for aws-s3 provider"))?;
            (
                Some(ak),
                Some(sk),
                config.storage.region.clone(),
                config.storage.bucket_name.clone(),
            )
        } else {
            (None, None, None, None)
        };

    // Turbopuffer
    let turbopuffer_api_key = if config.multivector_store.provider == "morphik" {
        Some(
            std::env::var("TURBOPUFFER_API_KEY")
                .map_err(|_| anyhow::anyhow!("TURBOPUFFER_API_KEY required for morphik provider"))?,
        )
    } else {
        std::env::var("TURBOPUFFER_API_KEY").ok()
    };

    // Cache
    let cache_path = config
        .storage
        .cache_path
        .clone()
        .unwrap_or_else(|| {
            let base = config.storage.storage_path.as_deref().unwrap_or("./storage");
            format!("{base}/cache")
        });
    let cache_max_bytes = (config.storage.cache_max_size_gb as u64) * 1024 * 1024 * 1024;

    let max_tokens: u32 = config
        .completion
        .default_max_tokens
        .parse()
        .unwrap_or(1000);

    let document_analysis_model = config.document_analysis.as_ref().map(|d| d.model.clone());

    Ok(Settings {
        host: config.api.host,
        port: config.api.port,
        environment: config.service.environment,
        jwt_algorithm: config.auth.jwt_algorithm,
        jwt_secret_key,
        bypass_auth_mode: config.auth.bypass_auth_mode,
        dev_user_id: config.auth.dev_user_id,
        registered_models: config.registered_models,
        completion_model: config.completion.model,
        default_max_tokens: max_tokens,
        default_temperature: config.completion.default_temperature,
        postgres_uri,
        db_pool_size: config.database.pool_size,
        embedding_model: config.embedding.model,
        vector_dimensions: config.embedding.dimensions,
        chunk_size: config.parser.chunk_size,
        chunk_overlap: config.parser.chunk_overlap,
        parser_mode: config.morphik.parser_mode,
        storage_provider: config.storage.provider,
        storage_path: config
            .storage
            .storage_path
            .unwrap_or_else(|| "./storage".to_string()),
        aws_region,
        s3_bucket,
        aws_access_key,
        aws_secret_access_key,
        s3_upload_concurrency: config.storage.s3_upload_concurrency,
        cache_enabled: config.storage.cache_enabled,
        cache_max_bytes,
        cache_path,
        vector_store_provider: config.vector_store.provider,
        ivfflat_probes: config.vector_store.ivfflat_probes.unwrap_or(100),
        multivector_store_provider: config.multivector_store.provider,
        turbopuffer_api_key,
        enable_colpali: config.morphik.enable_colpali,
        colpali_mode: config.morphik.colpali_mode,
        morphik_embedding_api_domain: config.morphik.morphik_embedding_api_domain,
        mode: config.morphik.mode,
        redis_url: config.redis.url,
        document_analysis_model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn minimal_toml() -> String {
        r#"
[api]
host = "0.0.0.0"
port = 8000

[auth]
jwt_algorithm = "HS256"
bypass_auth_mode = true

[completion]
model = "openai_gpt4-1-mini"

[database]
provider = "postgres"

[embedding]
model = "openai_embedding"
dimensions = 1536
similarity_metric = "cosine"

[parser]
chunk_size = 6000
chunk_overlap = 300

[reranker]
use_reranker = false

[storage]
provider = "local"
storage_path = "./storage"

[vector_store]
provider = "pgvector"

[morphik]
enable_colpali = true
colpali_mode = "api"
parser_mode = "api"
morphik_embedding_api_domain = ["http://localhost:6000"]
"#
        .to_string()
    }

    #[test]
    fn test_parse_minimal_toml() {
        unsafe { std::env::set_var("POSTGRES_URI", "postgresql://test:test@localhost/test") };
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(minimal_toml().as_bytes()).unwrap();
        let settings = load_settings_from_path(tmp.path()).unwrap();

        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 8000);
        assert!(settings.bypass_auth_mode);
        assert_eq!(settings.completion_model, "openai_gpt4-1-mini");
        assert_eq!(settings.embedding_model, "openai_embedding");
        assert_eq!(settings.vector_dimensions, 1536);
        assert_eq!(settings.chunk_size, 6000);
        assert_eq!(settings.storage_provider, "local");
        assert_eq!(settings.vector_store_provider, "pgvector");
        assert!(settings.enable_colpali);
        assert_eq!(settings.colpali_mode, "api");
        assert_eq!(settings.parser_mode, "api");
    }

    #[test]
    fn test_parse_production_toml() {
        unsafe { std::env::set_var("POSTGRES_URI", "postgresql://test:test@localhost/test") };
        unsafe { std::env::set_var("JWT_SECRET_KEY", "production-secret") };
        unsafe { std::env::set_var("AWS_ACCESS_KEY", "AKIATEST") };
        unsafe { std::env::set_var("AWS_SECRET_ACCESS_KEY", "secret123") };
        unsafe { std::env::set_var("TURBOPUFFER_API_KEY", "tpuf-test") };

        let toml_content = r#"
[api]
host = "0.0.0.0"
port = 8000

[service]
environment = "production"

[auth]
jwt_algorithm = "HS256"
bypass_auth_mode = false

[registered_models]
openai_gpt4-1-mini = { model_name = "gpt-4.1-mini" }
openai_embedding = { model_name = "text-embedding-3-small" }

[completion]
model = "openai_gpt4-1-mini"

[database]
provider = "postgres"

[embedding]
model = "openai_embedding"
dimensions = 1536
similarity_metric = "cosine"

[parser]
chunk_size = 6000
chunk_overlap = 300

[reranker]
use_reranker = false

[storage]
provider = "aws-s3"
region = "us-east-2"
bucket_name = "morphik-s3-storage"
storage_path = "morphik-storage"
cache_enabled = true
cache_max_size_gb = 40

[vector_store]
provider = "pgvector"

[multivector_store]
provider = "morphik"

[morphik]
enable_colpali = true
colpali_mode = "api"
parser_mode = "api"
morphik_embedding_api_domain = ["https://embedding-api.morphik.ai"]
"#;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(toml_content.as_bytes()).unwrap();
        let settings = load_settings_from_path(tmp.path()).unwrap();

        assert_eq!(settings.environment, "production");
        assert!(!settings.bypass_auth_mode);
        assert_eq!(settings.storage_provider, "aws-s3");
        assert_eq!(settings.aws_region.as_deref(), Some("us-east-2"));
        assert_eq!(settings.s3_bucket.as_deref(), Some("morphik-s3-storage"));
        assert_eq!(settings.multivector_store_provider, "morphik");
        assert_eq!(settings.turbopuffer_api_key.as_deref(), Some("tpuf-test"));
        assert_eq!(settings.cache_max_bytes, 40 * 1024 * 1024 * 1024);
        assert_eq!(
            settings.resolve_model_name("openai_gpt4-1-mini"),
            Some("gpt-4.1-mini".to_string())
        );
    }
}
