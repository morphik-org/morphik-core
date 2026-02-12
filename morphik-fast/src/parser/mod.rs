pub mod api;


/// Text chunking using recursive character splitting.
/// Splits text into chunks of approximately `chunk_size` characters
/// with `chunk_overlap` character overlap.
pub fn split_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let separators = ["\n\n", "\n", ". ", " ", ""];
    recursive_split(text, &separators, chunk_size, chunk_overlap)
}

fn recursive_split(
    text: &str,
    separators: &[&str],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    if text.len() <= chunk_size || separators.is_empty() {
        return vec![text.to_string()];
    }

    let separator = separators[0];
    let remaining_separators = &separators[1..];

    if separator.is_empty() {
        // Character-level split as fallback.
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let end = (start + chunk_size).min(text.len());
            chunks.push(text[start..end].to_string());
            if end >= text.len() {
                break;
            }
            start = end.saturating_sub(chunk_overlap);
        }
        return chunks;
    }

    let parts: Vec<&str> = text.split(separator).collect();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for part in parts {
        let candidate = if current.is_empty() {
            part.to_string()
        } else {
            format!("{current}{separator}{part}")
        };

        if candidate.len() > chunk_size {
            if !current.is_empty() {
                // Current chunk is big enough, save it.
                if current.len() > chunk_size {
                    // Need to sub-split.
                    chunks.extend(recursive_split(&current, remaining_separators, chunk_size, chunk_overlap));
                } else {
                    chunks.push(current.clone());
                }
                // Start new chunk with overlap.
                let overlap_start = current.len().saturating_sub(chunk_overlap);
                current = format!("{}{separator}{part}", &current[overlap_start..]);
            } else {
                // Single part bigger than chunk_size.
                chunks.extend(recursive_split(part, remaining_separators, chunk_size, chunk_overlap));
                current = String::new();
            }
        } else {
            current = candidate;
        }
    }

    if !current.is_empty() {
        if current.len() > chunk_size {
            chunks.extend(recursive_split(&current, remaining_separators, chunk_size, chunk_overlap));
        } else {
            chunks.push(current);
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_small_text() {
        let chunks = split_text("hello", 1000, 100);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn test_split_empty() {
        let chunks = split_text("", 1000, 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_multiple_chunks() {
        let text = "a".repeat(500) + "\n\n" + &"b".repeat(500);
        let chunks = split_text(&text, 600, 50);
        assert!(chunks.len() >= 2);
        // Each chunk should be within size limit.
        for chunk in &chunks {
            assert!(chunk.len() <= 650, "Chunk too large: {} chars", chunk.len());
        }
    }

    #[test]
    fn test_split_with_overlap() {
        let text = (0..10)
            .map(|i| format!("Paragraph {i}. {}", "x".repeat(50)))
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks = split_text(&text, 200, 20);
        assert!(chunks.len() > 1);
    }
}
