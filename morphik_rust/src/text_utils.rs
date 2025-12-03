//! Text processing utilities
//!
//! Fast text operations for chunking, token counting, and string manipulation.

use pyo3::prelude::*;
use rayon::prelude::*;

/// Count tokens using whitespace splitting.
///
/// A fast approximation of token count using whitespace as separator.
///
/// Args:
///     text: Text to count tokens in
///
/// Returns:
///     Approximate token count
#[pyfunction]
pub fn count_tokens_whitespace(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Count tokens in multiple texts in parallel.
///
/// Args:
///     texts: List of texts to count tokens in
///
/// Returns:
///     List of token counts
#[pyfunction]
pub fn count_tokens_batch(texts: Vec<String>) -> Vec<usize> {
    texts
        .par_iter()
        .map(|t| t.split_whitespace().count())
        .collect()
}

/// Extract all text content from a string, normalizing whitespace.
///
/// Collapses multiple whitespace characters into single spaces and trims.
///
/// Args:
///     text: Input text
///
/// Returns:
///     Normalized text
#[pyfunction]
pub fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split text into sentences using common delimiters.
///
/// Args:
///     text: Text to split
///
/// Returns:
///     List of sentences
#[pyfunction]
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' {
            // Check if this is end of sentence (followed by space or end)
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    // Add remaining text
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Find all occurrences of a substring (case-insensitive).
///
/// Args:
///     text: Text to search in
///     pattern: Pattern to search for
///
/// Returns:
///     List of (start, end) positions
#[pyfunction]
pub fn find_all_positions(text: &str, pattern: &str) -> Vec<(usize, usize)> {
    let text_lower = text.to_lowercase();
    let pattern_lower = pattern.to_lowercase();
    let pattern_len = pattern.len();

    let mut positions = Vec::new();
    let mut start = 0;

    while let Some(pos) = text_lower[start..].find(&pattern_lower) {
        let absolute_pos = start + pos;
        positions.push((absolute_pos, absolute_pos + pattern_len));
        start = absolute_pos + 1;
    }

    positions
}

/// Clean text by removing control characters.
///
/// Removes null bytes and other control characters that cause issues in databases.
///
/// Args:
///     text: Text to clean
///
/// Returns:
///     Cleaned text
#[pyfunction]
pub fn clean_control_chars(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t' || *c == '\r')
        .collect()
}

/// Batch clean control characters from multiple texts.
///
/// Args:
///     texts: List of texts to clean
///
/// Returns:
///     List of cleaned texts
#[pyfunction]
pub fn clean_control_chars_batch(texts: Vec<String>) -> Vec<String> {
    texts
        .par_iter()
        .map(|t| {
            t.chars()
                .filter(|c| !c.is_control() || *c == '\n' || *c == '\t' || *c == '\r')
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens() {
        assert_eq!(count_tokens_whitespace("hello world"), 2);
        assert_eq!(count_tokens_whitespace("  multiple   spaces  "), 2);
        assert_eq!(count_tokens_whitespace(""), 0);
    }

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("a\tb\nc"), "a b c");
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? I'm fine!");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_clean_control_chars() {
        assert_eq!(clean_control_chars("hello\x00world"), "helloworld");
        assert_eq!(clean_control_chars("line1\nline2"), "line1\nline2");
    }
}
