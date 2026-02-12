use axum::extract::{Multipart, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};

use crate::app::AppState;
use crate::auth::extract_auth_from_header;
use crate::models::api::{IngestRequest, IngestResponse};

/// Document ingestion routes.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v2/documents", post(ingest_document))
        .route("/v2/documents/text", post(ingest_text))
}

/// POST /v2/documents - Ingest a document from file upload.
async fn ingest_document(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
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

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to read multipart field: {e}"),
        )
    })? {
        let field_name = field.name().unwrap_or("").to_string();
        match field_name.as_str() {
            "file" => {
                filename = field.file_name().map(|s| s.to_string());
                file_bytes = Some(field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read file: {e}"),
                    )
                })?.to_vec());
            }
            "metadata" => {
                let text = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read metadata: {e}"),
                    )
                })?;
                metadata = serde_json::from_str(&text).unwrap_or_default();
            }
            _ => {}
        }
    }

    let file_data = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "No file provided".to_string(),
        )
    })?;
    let fname = filename.clone().unwrap_or_else(|| "upload".to_string());

    // 1. Parse document.
    let chunks = state
        .parser
        .parse_and_chunk(&file_data, &fname)
        .await
        .map_err(|e| {
            error!("Parse error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Parse error: {e}"))
        })?;

    if chunks.is_empty() {
        return Ok(Json(IngestResponse {
            document_id: String::new(),
            chunk_count: 0,
            filename,
            metadata,
        }));
    }

    // 2. Generate embeddings.
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let embeddings = state
        .embedding_model
        .embed_for_ingestion(&texts)
        .await
        .map_err(|e| {
            error!("Embedding error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Embedding error: {e}"),
            )
        })?;

    // 3. Create document ID and convert chunks.
    let document_id = uuid::Uuid::new_v4().to_string();
    let doc_chunks: Vec<crate::models::chunk::DocumentChunk> = chunks
        .into_iter()
        .zip(embeddings.into_iter())
        .enumerate()
        .map(|(i, (chunk, embedding))| chunk.to_document_chunk(document_id.clone(), i as i32, embedding))
        .collect();

    let chunk_count = doc_chunks.len();

    // 4. Store in vector store.
    state
        .vector_store
        .store_embeddings(&doc_chunks, Some(app_id))
        .await
        .map_err(|e| {
            error!("Vector store error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Storage error: {e}"),
            )
        })?;

    // 5. Store in ColPali vector store if enabled.
    if let Some(colpali_store) = &state.colpali_vector_store {
        if let Err(e) = colpali_store
            .store_embeddings(&doc_chunks, Some(app_id))
            .await
        {
            error!("ColPali store error: {e}");
        }
    }

    // 6. Upsert document metadata.
    state
        .database
        .upsert_document(
            &document_id,
            None,
            filename.as_deref(),
            app_id,
            chunk_count as i32,
            &metadata,
        )
        .await
        .map_err(|e| {
            error!("Database error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {e}"),
            )
        })?;

    info!("Ingested document {document_id} with {chunk_count} chunks");

    Ok(Json(IngestResponse {
        document_id,
        chunk_count,
        filename,
        metadata,
    }))
}

/// POST /v2/documents/text - Ingest plain text content.
async fn ingest_text(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
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

    let content = req.content.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "content field is required".to_string(),
        )
    })?;

    // 1. Parse text into chunks.
    let chunks = state
        .parser
        .parse_text_content(&content, req.metadata.clone());

    if chunks.is_empty() {
        return Ok(Json(IngestResponse {
            document_id: String::new(),
            chunk_count: 0,
            filename: req.filename,
            metadata: req.metadata,
        }));
    }

    // 2. Generate embeddings.
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let embeddings = state
        .embedding_model
        .embed_for_ingestion(&texts)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Embedding error: {e}"),
            )
        })?;

    // 3. Create document chunks.
    let document_id = req
        .external_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let doc_chunks: Vec<crate::models::chunk::DocumentChunk> = chunks
        .into_iter()
        .zip(embeddings.into_iter())
        .enumerate()
        .map(|(i, (chunk, embedding))| chunk.to_document_chunk(document_id.clone(), i as i32, embedding))
        .collect();

    let chunk_count = doc_chunks.len();

    // 4. Store embeddings.
    state
        .vector_store
        .store_embeddings(&doc_chunks, Some(app_id))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Storage error: {e}"),
            )
        })?;

    // 5. ColPali store.
    if let Some(colpali_store) = &state.colpali_vector_store {
        if let Err(e) = colpali_store
            .store_embeddings(&doc_chunks, Some(app_id))
            .await
        {
            error!("ColPali store error: {e}");
        }
    }

    // 6. Database.
    state
        .database
        .upsert_document(
            &document_id,
            None,
            req.filename.as_deref(),
            app_id,
            chunk_count as i32,
            &req.metadata,
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {e}"),
            )
        })?;

    Ok(Json(IngestResponse {
        document_id,
        chunk_count,
        filename: req.filename,
        metadata: req.metadata,
    }))
}
