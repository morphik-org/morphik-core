use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use std::sync::Arc;
use tracing::error;

use crate::app::AppState;
use crate::auth::extract_auth_from_header;
use crate::models::api::{ChatRequest, ChatResponse, RetrieveChunk, RetrieveRequest, RetrieveResponse};

/// Retrieve routes.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v2/retrieve/chunks", post(retrieve_chunks))
        .route("/v2/retrieve/chat", post(retrieve_chat))
}

/// POST /v2/retrieve/chunks - Retrieve similar chunks.
async fn retrieve_chunks(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RetrieveRequest>,
) -> Result<Json<RetrieveResponse>, (StatusCode, String)> {
    let auth = extract_auth_from_header(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        &state.settings.jwt_secret_key,
        &state.settings.jwt_algorithm,
        state.settings.bypass_auth_mode,
        &state.settings.dev_user_id,
    )?;

    let app_id = auth.app_id.as_deref().ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            "app_id required for V2 endpoints".to_string(),
        )
    })?;

    // 1. Embed the query.
    let query_embedding = state
        .embedding_model
        .embed_for_query(&req.query)
        .await
        .map_err(|e| {
            error!("Embedding error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Embedding error: {e}"),
            )
        })?;

    // 2. Query the appropriate vector store.
    let doc_ids = req.document_ids.as_deref();

    let chunks = if req.use_colpali {
        if let Some(colpali_store) = &state.colpali_vector_store {
            colpali_store
                .query_similar(&query_embedding, req.k, doc_ids, Some(app_id))
                .await
                .map_err(|e| {
                    error!("ColPali query error: {e}");
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Query error: {e}"),
                    )
                })?
        } else {
            state
                .vector_store
                .query_similar(&query_embedding, req.k, doc_ids, Some(app_id))
                .await
                .map_err(|e| {
                    error!("Query error: {e}");
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Query error: {e}"),
                    )
                })?
        }
    } else {
        state
            .vector_store
            .query_similar(&query_embedding, req.k, doc_ids, Some(app_id))
            .await
            .map_err(|e| {
                error!("Query error: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Query error: {e}"),
                )
            })?
    };

    // 3. Filter by minimum score.
    let filtered: Vec<RetrieveChunk> = chunks
        .into_iter()
        .filter(|c| c.score >= req.min_score)
        .map(|c| RetrieveChunk {
            document_id: c.document_id,
            chunk_number: c.chunk_number,
            content: c.content,
            score: c.score,
            metadata: c.metadata,
        })
        .collect();

    Ok(Json(RetrieveResponse { chunks: filtered }))
}

/// POST /v2/retrieve/chat - Retrieve + chat completion.
async fn retrieve_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let auth = extract_auth_from_header(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        &state.settings.jwt_secret_key,
        &state.settings.jwt_algorithm,
        state.settings.bypass_auth_mode,
        &state.settings.dev_user_id,
    )?;

    let app_id = auth.app_id.as_deref().ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            "app_id required".to_string(),
        )
    })?;

    // Get the last user message as the query.
    let query = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // 1. Embed query and retrieve chunks.
    let query_embedding = state
        .embedding_model
        .embed_for_query(&query)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Embedding error: {e}"),
            )
        })?;

    let doc_ids = req.document_ids.as_deref();
    let chunks = state
        .vector_store
        .query_similar(&query_embedding, req.k, doc_ids, Some(app_id))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {e}"),
            )
        })?;

    // 2. Build context from retrieved chunks.
    let context: String = chunks
        .iter()
        .enumerate()
        .map(|(_i, c)| format!("[Document {}: chunk {}]\n{}", c.document_id, c.chunk_number, c.content))
        .collect::<Vec<_>>()
        .join("\n\n");

    // 3. Build messages for completion.
    let system_msg = crate::completion::Message {
        role: "system".to_string(),
        content: format!(
            "Use the following retrieved context to answer the user's question. \
             If the context doesn't contain relevant information, say so.\n\n\
             Context:\n{context}"
        ),
    };

    let mut messages = vec![system_msg];
    for msg in &req.messages {
        messages.push(crate::completion::Message {
            role: msg.role.clone(),
            content: msg.content.clone(),
        });
    }

    // 4. Generate completion.
    let result = state
        .completion_model
        .complete(&messages, req.max_tokens, req.temperature)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Completion error: {e}"),
            )
        })?;

    let sources: Vec<RetrieveChunk> = chunks
        .into_iter()
        .map(|c| RetrieveChunk {
            document_id: c.document_id,
            chunk_number: c.chunk_number,
            content: c.content,
            score: c.score,
            metadata: c.metadata,
        })
        .collect();

    Ok(Json(ChatResponse {
        completion: result.content,
        sources,
    }))
}
