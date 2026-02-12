pub mod local;
pub mod s3;

use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("File not found: {bucket}/{key}")]
    NotFound { bucket: String, key: String },
    #[error("Storage I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("S3 error: {0}")]
    S3(String),
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("{0}")]
    Other(String),
}

/// Abstract storage backend interface.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Upload base64-encoded content.
    async fn upload_from_base64(
        &self,
        content: &str,
        key: &str,
        content_type: Option<&str>,
        bucket: &str,
    ) -> Result<(String, String), StorageError>;

    /// Upload raw bytes.
    async fn upload_bytes(
        &self,
        data: &[u8],
        key: &str,
        content_type: Option<&str>,
        bucket: &str,
    ) -> Result<(String, String), StorageError>;

    /// Download file content.
    async fn download_file(&self, bucket: &str, key: &str) -> Result<Vec<u8>, StorageError>;

    /// Generate presigned download URL (for S3).
    async fn get_download_url(
        &self,
        bucket: &str,
        key: &str,
        expires_in: u64,
    ) -> Result<String, StorageError>;

    /// Delete a file.
    async fn delete_file(&self, bucket: &str, key: &str) -> Result<bool, StorageError>;

    /// Get stored object size in bytes.
    async fn get_object_size(&self, bucket: &str, key: &str) -> Result<u64, StorageError>;

    /// Return the provider name for metrics.
    fn provider_name(&self) -> &str;

    /// Return the default bucket name.
    fn default_bucket(&self) -> &str;
}

/// Detect file type from raw bytes (simplified).
pub fn detect_file_extension(data: &[u8]) -> &'static str {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        ".png"
    } else if data.starts_with(b"\xff\xd8") {
        ".jpg"
    } else if data.starts_with(b"GIF8") {
        ".gif"
    } else if data.starts_with(b"BM") {
        ".bmp"
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        ".webp"
    } else if data.starts_with(b"%PDF") {
        ".pdf"
    } else {
        ".bin"
    }
}
