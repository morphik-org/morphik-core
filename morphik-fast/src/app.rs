use std::sync::Arc;

use crate::completion::CompletionModel;
use crate::config::Settings;
use crate::database::Database;
use crate::embedding::EmbeddingModel;
use crate::parser::api::ApiParser;
use crate::vector_store::VectorStore;

/// Shared application state passed to all route handlers.
pub struct AppState {
    pub settings: Settings,
    pub vector_store: Arc<dyn VectorStore>,
    pub colpali_vector_store: Option<Arc<dyn VectorStore>>,
    pub embedding_model: Arc<dyn EmbeddingModel>,
    pub completion_model: Arc<dyn CompletionModel>,
    pub parser: ApiParser,
    pub database: Arc<dyn Database>,
}
