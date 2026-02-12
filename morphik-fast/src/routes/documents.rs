use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{delete, get};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;
use tracing::{error, info};

use crate::app::AppState;
use crate::auth::extract_auth_from_header;
use crate::models::api::{DeleteResponse, DocumentInfo, HealthResponse};

/// Document management routes.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(health))
        .route("/v2/documents", get(list_documents))
        .route("/v2/documents/{document_id}", get(get_document))
        .route("/v2/documents/{document_id}", delete(delete_document))
}

/// GET /health
async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: state.settings.environment.clone(),
    })
}

#[derive(Debug, Deserialize)]
struct ListParams {
    #[serde(default = "default_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}

fn default_limit() -> i64 {
    100
}

/// GET /v2/documents - List documents.
async fn list_documents(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<ListParams>,
) -> Result<Json<Vec<DocumentInfo>>, (StatusCode, String)> {
    let auth = extract_auth_from_header(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        &state.settings.jwt_secret_key,
        &state.settings.jwt_algorithm,
        state.settings.bypass_auth_mode,
        &state.settings.dev_user_id,
    )?;

    let app_id = auth.app_id.as_deref().ok_or_else(|| {
        (StatusCode::FORBIDDEN, "app_id required".to_string())
    })?;

    let docs = state
        .database
        .list_documents(app_id, params.limit, params.offset)
        .await
        .map_err(|e| {
            error!("Database error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}"))
        })?;

    Ok(Json(docs))
}

/// GET /v2/documents/:document_id - Get document info.
async fn get_document(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(document_id): Path<String>,
) -> Result<Json<DocumentInfo>, (StatusCode, String)> {
    let auth = extract_auth_from_header(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        &state.settings.jwt_secret_key,
        &state.settings.jwt_algorithm,
        state.settings.bypass_auth_mode,
        &state.settings.dev_user_id,
    )?;

    let app_id = auth.app_id.as_deref().ok_or_else(|| {
        (StatusCode::FORBIDDEN, "app_id required".to_string())
    })?;

    let doc = state
        .database
        .get_document(&document_id, app_id)
        .await
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}"))
        })?;

    let info = doc.ok_or_else(|| (StatusCode::NOT_FOUND, "Document not found".to_string()))?;
    Ok(Json(info))
}

/// DELETE /v2/documents/:document_id - Delete a document.
async fn delete_document(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(document_id): Path<String>,
) -> Result<Json<DeleteResponse>, (StatusCode, String)> {
    let auth = extract_auth_from_header(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        &state.settings.jwt_secret_key,
        &state.settings.jwt_algorithm,
        state.settings.bypass_auth_mode,
        &state.settings.dev_user_id,
    )?;

    let app_id = auth.app_id.as_deref().ok_or_else(|| {
        (StatusCode::FORBIDDEN, "app_id required".to_string())
    })?;

    // Delete from database.
    state
        .database
        .delete_document(&document_id, app_id)
        .await
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {e}"))
        })?;

    // Delete from vector store.
    state
        .vector_store
        .delete_chunks_by_document_id(&document_id, Some(app_id))
        .await
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Vector store error: {e}"))
        })?;

    // Delete from ColPali store.
    if let Some(colpali_store) = &state.colpali_vector_store {
        if let Err(e) = colpali_store
            .delete_chunks_by_document_id(&document_id, Some(app_id))
            .await
        {
            error!("ColPali delete error: {e}");
        }
    }

    info!("Deleted document {document_id}");

    Ok(Json(DeleteResponse {
        success: true,
        document_id,
    }))
}
