//! Fast base64 encoding/decoding operations
//!
//! Uses the `base64` crate which provides SIMD-optimized operations
//! for significant speedup over Python's base64 module.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Encode bytes to base64 string.
///
/// Args:
///     data: Bytes to encode
///
/// Returns:
///     Base64-encoded string
#[pyfunction]
pub fn encode_base64(data: &[u8]) -> String {
    STANDARD.encode(data)
}

/// Decode base64 string to bytes.
///
/// Args:
///     encoded: Base64-encoded string
///
/// Returns:
///     Decoded bytes
///
/// Raises:
///     ValueError: If the input is not valid base64
#[pyfunction]
pub fn decode_base64(py: Python<'_>, encoded: &str) -> PyResult<Py<PyBytes>> {
    match STANDARD.decode(encoded) {
        Ok(decoded) => Ok(PyBytes::new_bound(py, &decoded).into()),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid base64: {}",
            e
        ))),
    }
}

/// Encode bytes to a data URI string.
///
/// Args:
///     data: Bytes to encode
///     mime_type: MIME type for the data URI (default: "image/png")
///
/// Returns:
///     Data URI string (e.g., "data:image/png;base64,...")
#[pyfunction]
#[pyo3(signature = (data, mime_type="image/png"))]
pub fn bytes_to_data_uri(data: &[u8], mime_type: &str) -> String {
    let encoded = STANDARD.encode(data);
    format!("data:{};base64,{}", mime_type, encoded)
}

/// Decode a data URI to bytes.
///
/// Handles both data URI format ("data:...;base64,...") and raw base64 strings.
///
/// Args:
///     data_uri: Data URI or raw base64 string
///
/// Returns:
///     Decoded bytes
///
/// Raises:
///     ValueError: If the input is not valid base64
#[pyfunction]
pub fn data_uri_to_bytes(py: Python<'_>, data_uri: &str) -> PyResult<Py<PyBytes>> {
    let base64_str = if data_uri.starts_with("data:") {
        // Extract base64 portion after the comma
        match data_uri.split_once(',') {
            Some((_, b64)) => b64,
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid data URI format",
                ))
            }
        }
    } else {
        data_uri
    };

    match STANDARD.decode(base64_str) {
        Ok(decoded) => Ok(PyBytes::new_bound(py, &decoded).into()),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid base64: {}",
            e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let original = b"Hello, World!";
        let encoded = encode_base64(original);
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");

        // Note: decode_base64 requires Python GIL, so we test encode here
    }

    #[test]
    fn test_data_uri_format() {
        let data = b"test";
        let uri = bytes_to_data_uri(data, "text/plain");
        assert!(uri.starts_with("data:text/plain;base64,"));
    }
}
