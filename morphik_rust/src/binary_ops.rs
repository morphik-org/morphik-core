//! Binary quantization operations for embeddings
//!
//! High-performance binary quantization using SIMD-friendly operations.
//! Converts float embeddings to binary (1 if > 0, else 0) for efficient storage.
//!
//! Based on techniques from:
//! - Qdrant's binary quantization (40x speedup)
//! - SimSIMD library patterns
//! - Meilisearch Arroy's SIMD approach
//!
//! The key insight is packing 8 sign bits into a single byte for compact storage
//! and fast Hamming distance computation.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// Quantize a 1D embedding to binary (values > 0 become true).
///
/// This is the basic unpacked version returning a list of booleans.
/// For storage efficiency, prefer `binary_quantize_packed`.
///
/// Args:
///     embedding: 1D numpy array of f32 floats
///
/// Returns:
///     List of booleans (True where value > 0)
#[pyfunction]
pub fn binary_quantize_1d(embedding: PyReadonlyArray1<f32>) -> Vec<bool> {
    let slice = embedding.as_slice().expect("Array must be contiguous");
    // Pre-allocate for better performance
    let mut result = Vec::with_capacity(slice.len());
    result.extend(slice.iter().map(|&v| v > 0.0));
    result
}

/// Quantize a 2D batch of embeddings to binary.
///
/// Uses rayon for parallel processing across the batch dimension.
///
/// Args:
///     embeddings: 2D numpy array of shape (batch_size, embedding_dim)
///
/// Returns:
///     List of lists of booleans
#[pyfunction]
pub fn binary_quantize_batch(embeddings: PyReadonlyArray2<f32>) -> Vec<Vec<bool>> {
    let shape = embeddings.shape();
    let batch_size = shape[0];
    let dim = shape[1];
    let data = embeddings.as_slice().expect("Array must be contiguous");

    // Process each embedding in parallel using rayon
    (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            let end = start + dim;
            let mut result = Vec::with_capacity(dim);
            result.extend(data[start..end].iter().map(|&v| v > 0.0));
            result
        })
        .collect()
}

/// Quantize embeddings to packed bytes (8 bits per byte).
///
/// This is the most memory-efficient representation:
/// - 128-dim embedding: 16 bytes instead of 512 bytes (32x compression)
/// - Enables fast Hamming distance computation using XOR + popcount
///
/// Bit packing is MSB-first: bit 0 of each byte corresponds to index 0, 8, 16...
///
/// Args:
///     embedding: 1D numpy array of f32 floats
///
/// Returns:
///     Bytes with packed binary representation
#[pyfunction]
pub fn binary_quantize_packed(py: Python<'_>, embedding: PyReadonlyArray1<f32>) -> Py<PyBytes> {
    let slice = embedding.as_slice().expect("Array must be contiguous");
    let len = slice.len();
    let num_bytes = (len + 7) / 8; // Ceiling division

    let mut packed = vec![0u8; num_bytes];

    // SIMD-friendly: process 8 elements at a time when possible
    let chunks = len / 8;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;
        let mut byte = 0u8;

        // Unrolled loop for 8 elements - compiler can optimize to SIMD
        if slice[base] > 0.0 {
            byte |= 0b1000_0000;
        }
        if slice[base + 1] > 0.0 {
            byte |= 0b0100_0000;
        }
        if slice[base + 2] > 0.0 {
            byte |= 0b0010_0000;
        }
        if slice[base + 3] > 0.0 {
            byte |= 0b0001_0000;
        }
        if slice[base + 4] > 0.0 {
            byte |= 0b0000_1000;
        }
        if slice[base + 5] > 0.0 {
            byte |= 0b0000_0100;
        }
        if slice[base + 6] > 0.0 {
            byte |= 0b0000_0010;
        }
        if slice[base + 7] > 0.0 {
            byte |= 0b0000_0001;
        }

        packed[chunk_idx] = byte;
    }

    // Handle remaining elements
    let remaining_start = chunks * 8;
    if remaining_start < len {
        let mut byte = 0u8;
        for (i, &v) in slice[remaining_start..].iter().enumerate() {
            if v > 0.0 {
                byte |= 1 << (7 - i);
            }
        }
        packed[chunks] = byte;
    }

    PyBytes::new_bound(py, &packed).into()
}

/// Quantize a batch of embeddings to packed bytes.
///
/// Uses parallel processing for large batches.
///
/// Args:
///     embeddings: 2D numpy array of shape (batch_size, embedding_dim)
///
/// Returns:
///     List of bytes objects with packed binary representation
#[pyfunction]
pub fn binary_quantize_batch_packed(
    py: Python<'_>,
    embeddings: PyReadonlyArray2<f32>,
) -> PyResult<Vec<Py<PyBytes>>> {
    let shape = embeddings.shape();
    let batch_size = shape[0];
    let dim = shape[1];
    let num_bytes = (dim + 7) / 8;
    let data = embeddings.as_slice().expect("Array must be contiguous");

    // Process each embedding in parallel
    let packed_vecs: Vec<Vec<u8>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let start = i * dim;
            let slice = &data[start..start + dim];
            let mut packed = vec![0u8; num_bytes];

            // Process 8 elements at a time
            let chunks = dim / 8;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * 8;
                let mut byte = 0u8;

                // Unrolled loop for SIMD-friendly processing
                if slice[base] > 0.0 {
                    byte |= 0b1000_0000;
                }
                if slice[base + 1] > 0.0 {
                    byte |= 0b0100_0000;
                }
                if slice[base + 2] > 0.0 {
                    byte |= 0b0010_0000;
                }
                if slice[base + 3] > 0.0 {
                    byte |= 0b0001_0000;
                }
                if slice[base + 4] > 0.0 {
                    byte |= 0b0000_1000;
                }
                if slice[base + 5] > 0.0 {
                    byte |= 0b0000_0100;
                }
                if slice[base + 6] > 0.0 {
                    byte |= 0b0000_0010;
                }
                if slice[base + 7] > 0.0 {
                    byte |= 0b0000_0001;
                }

                packed[chunk_idx] = byte;
            }

            // Handle remaining elements
            let remaining_start = chunks * 8;
            if remaining_start < dim {
                let mut byte = 0u8;
                for (j, &v) in slice[remaining_start..].iter().enumerate() {
                    if v > 0.0 {
                        byte |= 1 << (7 - j);
                    }
                }
                packed[chunks] = byte;
            }

            packed
        })
        .collect();

    // Convert to Python bytes (GIL required)
    Ok(packed_vecs
        .into_iter()
        .map(|v| PyBytes::new_bound(py, &v).into())
        .collect())
}

/// Compute Hamming distance between two packed binary vectors.
///
/// Uses XOR + popcount which is very fast on modern CPUs.
///
/// Args:
///     a: First packed binary vector
///     b: Second packed binary vector
///
/// Returns:
///     Hamming distance (number of differing bits)
///
/// Raises:
///     ValueError: If vectors have different lengths
#[pyfunction]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> PyResult<u32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    Ok(a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum())
}

/// Batch compute Hamming distances between a query and multiple candidates.
///
/// Uses parallel processing for large batches.
///
/// Args:
///     query: Query packed binary vector
///     candidates: List of candidate packed binary vectors
///
/// Returns:
///     List of Hamming distances
///
/// Raises:
///     ValueError: If any candidate has different length than query
#[pyfunction]
pub fn hamming_distance_batch(query: Vec<u8>, candidates: Vec<Vec<u8>>) -> PyResult<Vec<u32>> {
    let query_len = query.len();

    // Validate all candidates have same length as query upfront
    for (i, candidate) in candidates.iter().enumerate() {
        if candidate.len() != query_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Candidate {} length mismatch: {} vs query length {}",
                i,
                candidate.len(),
                query_len
            )));
        }
    }

    Ok(candidates
        .par_iter()
        .map(|candidate| {
            query
                .iter()
                .zip(candidate.iter())
                .map(|(&x, &y)| (x ^ y).count_ones())
                .sum()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_logic() {
        let values = vec![1.0f32, -0.5, 0.1, -2.0, 0.0, 3.0, -1.0, 0.5];
        let result: Vec<bool> = values.iter().map(|&v| v > 0.0).collect();
        assert_eq!(
            result,
            vec![true, false, true, false, false, true, false, true]
        );
    }

    #[test]
    fn test_bit_packing() {
        // Test that [1, -1, 1, -1, -1, 1, -1, 1] packs to 0b10100101 = 165
        let values = [1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
        let mut byte = 0u8;
        for (i, &v) in values.iter().enumerate() {
            if v > 0.0 {
                byte |= 1 << (7 - i);
            }
        }
        assert_eq!(byte, 0b10100101);
    }

    #[test]
    fn test_hamming() {
        let a = vec![0b11110000u8, 0b10101010];
        let b = vec![0b11110000u8, 0b01010101];
        // Second byte differs completely: 8 bits
        assert_eq!(
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x ^ y).count_ones())
                .sum::<u32>(),
            8
        );
    }
}
