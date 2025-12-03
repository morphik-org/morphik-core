//! Fast text chunking with overlap
//!
//! Rust implementation of RecursiveCharacterTextSplitter for
//! significantly faster text splitting, especially on large documents.

use pyo3::prelude::*;

/// Default separators for recursive text splitting (same as Python implementation)
const DEFAULT_SEPARATORS: &[&str] = &["\n\n", "\n", ". ", " ", ""];

/// Split text into chunks with specified size and overlap.
///
/// Uses recursive splitting with default separators: ["\n\n", "\n", ". ", " ", ""]
///
/// Args:
///     text: Text to split
///     chunk_size: Maximum size of each chunk
///     chunk_overlap: Number of characters to overlap between chunks
///
/// Returns:
///     List of text chunks
#[pyfunction]
#[pyo3(signature = (text, chunk_size, chunk_overlap))]
pub fn split_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let separators: Vec<&str> = DEFAULT_SEPARATORS.to_vec();
    split_recursive(text, chunk_size, chunk_overlap, &separators)
}

/// Split text into chunks with custom separators.
///
/// Args:
///     text: Text to split
///     chunk_size: Maximum size of each chunk
///     chunk_overlap: Number of characters to overlap between chunks
///     separators: List of separators to use (tried in order)
///
/// Returns:
///     List of text chunks
#[pyfunction]
#[pyo3(signature = (text, chunk_size, chunk_overlap, separators))]
pub fn split_text_with_separators(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
) -> Vec<String> {
    let sep_refs: Vec<&str> = separators.iter().map(|s| s.as_str()).collect();
    split_recursive(text, chunk_size, chunk_overlap, &sep_refs)
}

/// Internal recursive splitting function
fn split_recursive(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    separators: &[&str],
) -> Vec<String> {
    // Base case: text fits in chunk
    if text.len() <= chunk_size {
        if text.is_empty() {
            return vec![];
        }
        return vec![text.to_string()];
    }

    // No separators left - split at chunk_size boundaries
    if separators.is_empty() {
        return text
            .chars()
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|c| c.iter().collect::<String>())
            .collect();
    }

    let sep = separators[0];
    let remaining_seps = &separators[1..];

    // Split by current separator
    let splits: Vec<&str> = if sep.is_empty() {
        // Character-level splitting
        text.char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect()
    } else {
        text.split(sep).collect()
    };

    // Merge splits into chunks
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for (i, part) in splits.iter().enumerate() {
        let add_part = if sep.is_empty() || i == splits.len() - 1 {
            part.to_string()
        } else {
            format!("{}{}", part, sep)
        };

        if current.len() + add_part.len() > chunk_size {
            if !current.is_empty() {
                chunks.push(current);
            }
            current = add_part;
        } else {
            current.push_str(&add_part);
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    // Recurse for oversized chunks
    let mut final_chunks: Vec<String> = Vec::new();
    for chunk in chunks {
        if chunk.len() > chunk_size && !remaining_seps.is_empty() {
            final_chunks.extend(split_recursive(&chunk, chunk_size, chunk_overlap, remaining_seps));
        } else {
            final_chunks.push(chunk);
        }
    }

    // Apply overlap
    if chunk_overlap > 0 && final_chunks.len() > 1 {
        let mut overlapped: Vec<String> = Vec::with_capacity(final_chunks.len());
        for (i, chunk) in final_chunks.iter().enumerate() {
            if i > 0 {
                let prev = &final_chunks[i - 1];
                let overlap_start = prev.len().saturating_sub(chunk_overlap);
                let overlap = &prev[overlap_start..];
                overlapped.push(format!("{}{}", overlap, chunk));
            } else {
                overlapped.push(chunk.clone());
            }
        }
        return overlapped;
    }

    final_chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_split() {
        let text = "Hello world. This is a test. Another sentence here.";
        let chunks = split_text(text, 20, 0);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Allow some tolerance for separator inclusion
            assert!(chunk.len() <= 25, "Chunk too large: {}", chunk.len());
        }
    }

    #[test]
    fn test_with_overlap() {
        let text = "AAAAA BBBBB CCCCC DDDDD EEEEE";
        let chunks = split_text(text, 10, 3);
        assert!(chunks.len() > 1);
        // Check that overlap is present
        if chunks.len() >= 2 {
            let last_of_first = &chunks[0][chunks[0].len().saturating_sub(3)..];
            assert!(chunks[1].starts_with(last_of_first) || chunks[1].contains(last_of_first));
        }
    }

    #[test]
    fn test_empty_text() {
        let chunks = split_text("", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_text() {
        let text = "Small";
        let chunks = split_text(text, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Small");
    }
}
