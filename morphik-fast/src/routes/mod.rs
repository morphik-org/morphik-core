pub mod documents;
pub mod ingest;
pub mod retrieve;

use axum::Router;
use std::sync::Arc;

use crate::app::AppState;

/// Build all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .merge(ingest::routes())
        .merge(retrieve::routes())
        .merge(documents::routes())
        .with_state(state)
}
