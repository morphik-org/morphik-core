mod app;
mod auth;
mod completion;
mod config;
mod database;
mod embedding;
mod models;
mod parser;
mod routes;
mod storage;
mod vector_store;

use std::net::SocketAddr;
use std::sync::Arc;

use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

use app::AppState;
use database::Database;
use config::load_settings_from_path;
use completion::openai::OpenAICompletionModel;
use database::postgres::PostgresDatabase;
use embedding::openai::OpenAIEmbeddingModel;
use parser::api::ApiParser;
use storage::local::LocalStorage;
use storage::s3::S3Storage;
use vector_store::fast_multivector::FastMultiVectorStore;
use vector_store::pgvector::PGVectorStore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting morphik-fast server...");

    // Load configuration.
    let settings = load_settings_from_path("morphik.toml")?;
    info!(
        "Configuration loaded: environment={}, host={}, port={}",
        settings.environment, settings.host, settings.port
    );

    // Initialize database.
    let database = Arc::new(
        PostgresDatabase::new(&settings.postgres_uri, settings.db_pool_size).await?,
    );
    database.initialize().await?;
    info!("Database initialized");

    // Initialize storage.
    let chunk_storage: Arc<dyn storage::Storage> = match settings.storage_provider.as_str() {
        "aws-s3" => {
            let s3 = S3Storage::new(
                settings.aws_access_key.as_deref().unwrap_or(""),
                settings.aws_secret_access_key.as_deref().unwrap_or(""),
                settings.aws_region.as_deref().unwrap_or("us-east-2"),
                settings.s3_bucket.as_deref().unwrap_or("morphik-s3-storage"),
                settings.s3_upload_concurrency,
            )
            .await?;
            Arc::new(s3)
        }
        "local" | _ => Arc::new(LocalStorage::new(&settings.storage_path)),
    };
    info!("Storage initialized: {}", settings.storage_provider);

    // Initialize PGVector store.
    let vector_store: Arc<dyn vector_store::VectorStore> = Arc::new(
        PGVectorStore::new(
            &settings.postgres_uri,
            settings.db_pool_size,
            settings.ivfflat_probes,
            settings.vector_dimensions,
        )
        .await?,
    );
    vector_store.initialize().await?;
    info!("PGVector store initialized");

    // Initialize ColPali multi-vector store if enabled.
    let colpali_vector_store: Option<Arc<dyn vector_store::VectorStore>> =
        if settings.enable_colpali && settings.multivector_store_provider == "morphik" {
            let tpuf_key = settings
                .turbopuffer_api_key
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("TURBOPUFFER_API_KEY required for morphik provider"))?;

            let s3_bucket = settings.s3_bucket.as_deref().unwrap_or("morphik-s3-storage");

            let fast_store = FastMultiVectorStore::new(
                &settings.postgres_uri,
                tpuf_key,
                chunk_storage.clone(),
                chunk_storage.clone(),
                s3_bucket,
                s3_bucket,
                settings.cache_enabled,
                &settings.cache_path,
                settings.cache_max_bytes,
                settings.s3_upload_concurrency,
            )
            .await?;
            info!("FastMultiVectorStore initialized (TurboPuffer)");
            Some(Arc::new(fast_store))
        } else {
            None
        };

    // Resolve model names from registered models.
    let embedding_model_name = settings
        .resolve_model_name(&settings.embedding_model)
        .unwrap_or_else(|| settings.embedding_model.clone());
    let completion_model_name = settings
        .resolve_model_name(&settings.completion_model)
        .unwrap_or_else(|| settings.completion_model.clone());

    // Initialize embedding model.
    let openai_api_key =
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string());
    let embedding_model: Arc<dyn embedding::EmbeddingModel> = Arc::new(
        OpenAIEmbeddingModel::new(
            &embedding_model_name,
            &openai_api_key,
            settings.vector_dimensions,
        ),
    );
    info!("Embedding model initialized: {embedding_model_name}");

    // Initialize completion model.
    let completion_model: Arc<dyn completion::CompletionModel> = Arc::new(
        OpenAICompletionModel::new(
            &completion_model_name,
            &openai_api_key,
            settings.default_max_tokens,
            settings.default_temperature,
        ),
    );
    info!("Completion model initialized: {completion_model_name}");

    // Initialize parser.
    let parser = ApiParser::new(
        settings.morphik_embedding_api_domain.clone(),
        settings.chunk_size,
        settings.chunk_overlap,
    );
    info!("Parser initialized (mode={})", settings.parser_mode);

    // Build application state.
    let state = Arc::new(AppState {
        settings: settings.clone(),
        vector_store,
        colpali_vector_store,
        embedding_model,
        completion_model,
        parser,
        database,
    });

    // Build router.
    let app = routes::build_router(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http());

    // Start server.
    let addr: SocketAddr = format!("{}:{}", settings.host, settings.port).parse()?;
    info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
