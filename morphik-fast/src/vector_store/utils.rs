use crate::models::api::StoreMetrics;

pub const MULTIVECTOR_CHUNKS_BUCKET: &str = "multivector-chunks";

/// Strip bucket prefix if embedded in the key.
pub fn normalize_storage_key(key: &str) -> String {
    let prefix = format!("{MULTIVECTOR_CHUNKS_BUCKET}/");
    if key.starts_with(&prefix) {
        key[prefix.len()..].to_string()
    } else {
        key.to_string()
    }
}

/// Best-effort heuristic to detect storage keys in content fields.
pub fn is_storage_key(value: &str) -> bool {
    if value.len() >= 500 || !value.contains('/') || value.starts_with("data:") || value.starts_with("http") {
        return false;
    }
    if value.chars().any(|c| c.is_whitespace()) {
        return false;
    }
    if value.chars().any(|c| matches!(c, '(' | ')' | ';' | '=' | '{' | '}')) {
        return false;
    }
    true
}

/// Derive a corrected image key when a legacy .txt key contains image data.
pub fn derive_repaired_image_key(storage_key: &str, is_image: bool, mime_type: Option<&str>) -> Option<String> {
    if !is_image || !storage_key.to_lowercase().ends_with(".txt") {
        return None;
    }
    let base = &storage_key[..storage_key.len() - 4]; // strip ".txt"
    let lower = base.to_lowercase();
    let image_exts = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"];
    if image_exts.iter().any(|ext| lower.ends_with(ext)) {
        return Some(base.to_string());
    }
    let ext = match mime_type {
        Some("image/jpeg") | Some("image/jpg") => ".jpg",
        Some("image/png") => ".png",
        Some("image/webp") => ".webp",
        Some("image/gif") => ".gif",
        Some("image/bmp") => ".bmp",
        Some("image/tiff") => ".tiff",
        _ => ".png",
    };
    Some(format!("{base}{ext}"))
}

/// Build store metrics with the given parameters.
pub fn build_store_metrics(
    chunk_payload_backend: &str,
    multivector_backend: &str,
    vector_store_backend: &str,
) -> StoreMetrics {
    StoreMetrics {
        chunk_payload_backend: chunk_payload_backend.to_string(),
        multivector_backend: multivector_backend.to_string(),
        vector_store_backend: vector_store_backend.to_string(),
        ..Default::default()
    }
}

/// Detect MIME type from raw image bytes.
pub fn detect_image_mime(data: &[u8]) -> &'static str {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if data.starts_with(b"\xff\xd8") {
        "image/jpeg"
    } else if data.starts_with(b"GIF8") {
        "image/gif"
    } else if data.starts_with(b"BM") {
        "image/bmp"
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "image/png"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_storage_key() {
        assert_eq!(
            normalize_storage_key("multivector-chunks/app/doc/0.txt"),
            "app/doc/0.txt"
        );
        assert_eq!(
            normalize_storage_key("app/doc/0.txt"),
            "app/doc/0.txt"
        );
    }

    #[test]
    fn test_is_storage_key() {
        assert!(is_storage_key("app/doc/0.txt"));
        assert!(is_storage_key("default/doc123/5.png"));
        assert!(!is_storage_key("data:image/png;base64,abc"));
        assert!(!is_storage_key("http://example.com"));
        assert!(!is_storage_key("no slash here"));
        assert!(!is_storage_key(&"x".repeat(501)));
    }

    #[test]
    fn test_derive_repaired_image_key() {
        assert_eq!(
            derive_repaired_image_key("app/doc/0.png.txt", true, None),
            Some("app/doc/0.png".to_string())
        );
        assert_eq!(
            derive_repaired_image_key("app/doc/0.txt", true, Some("image/jpeg")),
            Some("app/doc/0.jpg".to_string())
        );
        assert_eq!(
            derive_repaired_image_key("app/doc/0.txt", false, None),
            None
        );
    }

    #[test]
    fn test_build_store_metrics() {
        let m = build_store_metrics("aws-s3", "turbopuffer", "pgvector");
        assert_eq!(m.chunk_payload_backend, "aws-s3");
        assert_eq!(m.vector_store_backend, "pgvector");
        assert_eq!(m.multivector_backend, "turbopuffer");
        assert_eq!(m.vector_store_rows, 0);
    }
}
