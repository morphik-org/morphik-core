use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tracing::debug;

/// Manage local on-disk cache for blobs with LRU eviction.
pub struct FileCacheManager {
    enabled: bool,
    base_dir: PathBuf,
    max_bytes: u64,
    index: Mutex<CacheIndex>,
}

struct CacheIndex {
    initialized: bool,
    entries: HashMap<PathBuf, (f64, u64)>, // (atime, size)
    total_size: u64,
}

impl FileCacheManager {
    pub fn new(enabled: bool, base_dir: PathBuf, max_bytes: u64) -> Self {
        if enabled {
            std::fs::create_dir_all(&base_dir).ok();
        }
        Self {
            enabled,
            base_dir,
            max_bytes,
            index: Mutex::new(CacheIndex {
                initialized: false,
                entries: HashMap::new(),
                total_size: 0,
            }),
        }
    }

    fn path_for(&self, namespace: &str, bucket: &str, key: &str) -> PathBuf {
        let ns = if namespace.is_empty() { "_default" } else { namespace };
        let bkt = if bucket.is_empty() { "_default" } else { bucket };
        self.base_dir.join(ns).join(bkt).join(key)
    }

    fn now_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Get cached data if available.
    pub async fn get(&self, namespace: &str, bucket: &str, key: &str) -> Option<Vec<u8>> {
        if !self.enabled {
            return None;
        }
        let path = self.path_for(namespace, bucket, key);
        match tokio::fs::read(&path).await {
            Ok(data) => {
                let now = Self::now_secs();
                // Update access time in index.
                let mut idx = self.index.lock().await;
                if let Some(entry) = idx.entries.get_mut(&path) {
                    entry.0 = now;
                }
                Some(data)
            }
            Err(_) => None,
        }
    }

    /// Write data to cache.
    pub async fn put(&self, namespace: &str, bucket: &str, key: &str, data: &[u8]) {
        if !self.enabled {
            return;
        }
        let path = self.path_for(namespace, bucket, key);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }

        // Write atomically via temp file.
        let tmp_name = format!(".{}.tmp-{}", path.file_name().unwrap_or_default().to_string_lossy(), uuid::Uuid::new_v4().as_hyphenated());
        let tmp_path = path.parent().unwrap_or(Path::new(".")).join(&tmp_name);

        if let Err(e) = tokio::fs::write(&tmp_path, data).await {
            debug!("Failed to write cache entry {}: {e}", path.display());
            return;
        }
        if let Err(e) = tokio::fs::rename(&tmp_path, &path).await {
            debug!("Failed to rename cache entry {}: {e}", path.display());
            tokio::fs::remove_file(&tmp_path).await.ok();
            return;
        }

        let size = data.len() as u64;
        let now = Self::now_secs();
        {
            let mut idx = self.index.lock().await;
            if let Some(prev) = idx.entries.get(&path) {
                idx.total_size = idx.total_size.saturating_sub(prev.1);
            }
            idx.entries.insert(path, (now, size));
            idx.total_size += size;
        }

        self.enforce_budget().await;
    }

    /// Delete a cache entry.
    pub async fn delete(&self, namespace: &str, bucket: &str, key: &str) {
        if !self.enabled {
            return;
        }
        let path = self.path_for(namespace, bucket, key);
        tokio::fs::remove_file(&path).await.ok();
        let mut idx = self.index.lock().await;
        if let Some(prev) = idx.entries.remove(&path) {
            idx.total_size = idx.total_size.saturating_sub(prev.1);
        }
    }

    /// Delete multiple cache entries.
    pub async fn delete_many(&self, namespace: &str, items: &[(String, String)]) {
        if !self.enabled {
            return;
        }
        for (bucket, key) in items {
            self.delete(namespace, bucket, key).await;
        }
    }

    /// Evict LRU entries until under budget.
    async fn enforce_budget(&self) {
        if !self.enabled {
            return;
        }
        let mut idx = self.index.lock().await;
        if idx.total_size <= self.max_bytes {
            return;
        }

        // Sort by access time ascending (oldest first).
        let mut files: Vec<(f64, u64, PathBuf)> = idx
            .entries
            .iter()
            .map(|(p, (atime, size))| (*atime, *size, p.clone()))
            .collect();
        files.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_, _, path) in files {
            if idx.total_size <= self.max_bytes {
                break;
            }
            tokio::fs::remove_file(&path).await.ok();
            if let Some(prev) = idx.entries.remove(&path) {
                idx.total_size = idx.total_size.saturating_sub(prev.1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cache_put_get() {
        let dir = TempDir::new().unwrap();
        let cache = FileCacheManager::new(true, dir.path().to_path_buf(), 1024 * 1024);

        cache.put("ns", "bucket", "key1", b"hello").await;
        let result = cache.get("ns", "bucket", "key1").await;
        assert_eq!(result, Some(b"hello".to_vec()));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let dir = TempDir::new().unwrap();
        let cache = FileCacheManager::new(true, dir.path().to_path_buf(), 1024 * 1024);

        let result = cache.get("ns", "bucket", "nonexistent").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_cache_delete() {
        let dir = TempDir::new().unwrap();
        let cache = FileCacheManager::new(true, dir.path().to_path_buf(), 1024 * 1024);

        cache.put("ns", "b", "k", b"data").await;
        assert!(cache.get("ns", "b", "k").await.is_some());
        cache.delete("ns", "b", "k").await;
        assert!(cache.get("ns", "b", "k").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_disabled() {
        let dir = TempDir::new().unwrap();
        let cache = FileCacheManager::new(false, dir.path().to_path_buf(), 1024);

        cache.put("ns", "b", "k", b"data").await;
        assert!(cache.get("ns", "b", "k").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_budget_eviction() {
        let dir = TempDir::new().unwrap();
        // Max 20 bytes
        let cache = FileCacheManager::new(true, dir.path().to_path_buf(), 20);

        // Each entry is 10 bytes
        cache.put("ns", "b", "k1", &[0u8; 10]).await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        cache.put("ns", "b", "k2", &[0u8; 10]).await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // This should trigger eviction of k1 (oldest)
        cache.put("ns", "b", "k3", &[0u8; 10]).await;

        // k1 should be evicted
        assert!(cache.get("ns", "b", "k1").await.is_none());
        // k3 should exist
        assert!(cache.get("ns", "b", "k3").await.is_some());
    }
}
