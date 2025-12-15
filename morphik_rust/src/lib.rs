//! Morphik Rust Extensions
//!
//! High-performance Rust implementations for CPU-intensive operations:
//! - Base64 encoding/decoding (SIMD-optimized)
//! - Text chunking with overlap
//! - Binary quantization for embeddings
//! - Text processing utilities

use pyo3::prelude::*;

mod base64_ops;
mod binary_ops;
mod chunking;
mod text_utils;

/// Morphik Rust extension module
#[pymodule]
fn morphik_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Base64 operations
    m.add_function(wrap_pyfunction!(base64_ops::encode_base64, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::decode_base64, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::bytes_to_data_uri, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::data_uri_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::decode_base64_batch, m)?)?;
    m.add_function(wrap_pyfunction!(base64_ops::encode_base64_batch, m)?)?;

    // Chunking operations
    m.add_function(wrap_pyfunction!(chunking::split_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::split_text_with_separators, m)?)?;

    // Binary quantization operations
    m.add_function(wrap_pyfunction!(binary_ops::binary_quantize_1d, m)?)?;
    m.add_function(wrap_pyfunction!(binary_ops::binary_quantize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(binary_ops::binary_quantize_packed, m)?)?;
    m.add_function(wrap_pyfunction!(binary_ops::binary_quantize_batch_packed, m)?)?;
    m.add_function(wrap_pyfunction!(binary_ops::hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(binary_ops::hamming_distance_batch, m)?)?;

    // Text utilities
    m.add_function(wrap_pyfunction!(text_utils::count_tokens_whitespace, m)?)?;
    m.add_function(wrap_pyfunction!(text_utils::count_tokens_batch, m)?)?;
    m.add_function(wrap_pyfunction!(text_utils::normalize_whitespace, m)?)?;
    m.add_function(wrap_pyfunction!(text_utils::split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(text_utils::find_all_positions, m)?)?;
    m.add_function(wrap_pyfunction!(text_utils::clean_control_chars, m)?)?;
    m.add_function(wrap_pyfunction!(text_utils::clean_control_chars_batch, m)?)?;

    Ok(())
}
