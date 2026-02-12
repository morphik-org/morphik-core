use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;

use super::{Storage, StorageError};

/// Local filesystem storage backend.
pub struct LocalStorage {
    base_path: PathBuf,
}

impl LocalStorage {
    pub fn new(storage_path: &str) -> Self {
        let base_path = PathBuf::from(storage_path);
        // Ensure directory exists (best-effort at construction time).
        std::fs::create_dir_all(&base_path).ok();
        Self { base_path }
    }

    fn resolve_path(&self, _bucket: &str, key: &str) -> PathBuf {
        self.base_path.join(key)
    }
}

#[async_trait]
impl Storage for LocalStorage {
    async fn upload_from_base64(
        &self,
        content: &str,
        key: &str,
        _content_type: Option<&str>,
        bucket: &str,
    ) -> Result<(String, String), StorageError> {
        use base64::Engine;
        // Handle data URI
        let payload = if content.starts_with("data:") {
            content
                .split_once(',')
                .map(|(_, b)| b)
                .unwrap_or(content)
        } else {
            content
        };
        let decoded = base64::engine::general_purpose::STANDARD.decode(payload)?;
        self.upload_bytes(&decoded, key, _content_type, bucket).await
    }

    async fn upload_bytes(
        &self,
        data: &[u8],
        key: &str,
        _content_type: Option<&str>,
        bucket: &str,
    ) -> Result<(String, String), StorageError> {
        let path = self.resolve_path(bucket, key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        fs::write(&path, data).await?;
        Ok((bucket.to_string(), key.to_string()))
    }

    async fn download_file(&self, bucket: &str, key: &str) -> Result<Vec<u8>, StorageError> {
        let path = self.resolve_path(bucket, key);
        if !path.exists() {
            return Err(StorageError::NotFound {
                bucket: bucket.to_string(),
                key: key.to_string(),
            });
        }
        Ok(fs::read(&path).await?)
    }

    async fn get_download_url(
        &self,
        _bucket: &str,
        key: &str,
        _expires_in: u64,
    ) -> Result<String, StorageError> {
        let path = self.resolve_path(_bucket, key);
        Ok(format!("file://{}", path.display()))
    }

    async fn delete_file(&self, bucket: &str, key: &str) -> Result<bool, StorageError> {
        let path = self.resolve_path(bucket, key);
        match fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(StorageError::Io(e)),
        }
    }

    async fn get_object_size(&self, bucket: &str, key: &str) -> Result<u64, StorageError> {
        let path = self.resolve_path(bucket, key);
        let meta = fs::metadata(&path).await?;
        Ok(meta.len())
    }

    fn provider_name(&self) -> &str {
        "local"
    }

    fn default_bucket(&self) -> &str {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_local_upload_download() {
        let dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(dir.path().to_str().unwrap());

        let data = b"hello world";
        let (bucket, key) = storage
            .upload_bytes(data, "test/file.txt", Some("text/plain"), "")
            .await
            .unwrap();
        assert_eq!(key, "test/file.txt");

        let downloaded = storage.download_file(&bucket, &key).await.unwrap();
        assert_eq!(downloaded, data);
    }

    #[tokio::test]
    async fn test_local_delete() {
        let dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(dir.path().to_str().unwrap());

        storage
            .upload_bytes(b"data", "del.txt", None, "")
            .await
            .unwrap();
        assert!(storage.delete_file("", "del.txt").await.unwrap());
        assert!(!storage.delete_file("", "del.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_local_object_size() {
        let dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(dir.path().to_str().unwrap());

        let data = b"0123456789";
        storage.upload_bytes(data, "size.bin", None, "").await.unwrap();
        let size = storage.get_object_size("", "size.bin").await.unwrap();
        assert_eq!(size, 10);
    }

    #[tokio::test]
    async fn test_local_upload_from_base64() {
        use base64::Engine;
        let dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(dir.path().to_str().unwrap());

        let original = b"test data";
        let encoded = base64::engine::general_purpose::STANDARD.encode(original);

        storage
            .upload_from_base64(&encoded, "b64.txt", None, "")
            .await
            .unwrap();
        let downloaded = storage.download_file("", "b64.txt").await.unwrap();
        assert_eq!(downloaded, original);
    }

    #[tokio::test]
    async fn test_local_not_found() {
        let dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(dir.path().to_str().unwrap());

        let result = storage.download_file("", "nonexistent.txt").await;
        assert!(result.is_err());
    }
}
