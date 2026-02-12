use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a chunk stored in VectorStore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// external_id of parent document
    pub document_id: String,
    pub content: String,
    /// Dense embedding vector (may be empty when returned from queries).
    #[serde(default)]
    pub embedding: Vec<f32>,
    pub chunk_number: i32,
    /// Chunk-specific metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub score: f64,
}

/// Represents a raw chunk from parsing (before embedding).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Chunk {
    /// Convert to a DocumentChunk with embedding.
    pub fn to_document_chunk(
        self,
        document_id: String,
        chunk_number: i32,
        embedding: Vec<f32>,
    ) -> DocumentChunk {
        DocumentChunk {
            document_id,
            content: self.content,
            embedding,
            chunk_number,
            metadata: self.metadata,
            score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_to_document_chunk() {
        let chunk = Chunk {
            content: "Hello world".to_string(),
            metadata: HashMap::new(),
        };
        let doc_chunk = chunk.to_document_chunk("doc1".to_string(), 0, vec![1.0, 2.0, 3.0]);
        assert_eq!(doc_chunk.document_id, "doc1");
        assert_eq!(doc_chunk.chunk_number, 0);
        assert_eq!(doc_chunk.embedding.len(), 3);
        assert_eq!(doc_chunk.score, 0.0);
    }

    #[test]
    fn test_document_chunk_serialize() {
        let chunk = DocumentChunk {
            document_id: "doc1".to_string(),
            content: "test content".to_string(),
            embedding: vec![],
            chunk_number: 0,
            metadata: HashMap::new(),
            score: 0.95,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: DocumentChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.document_id, "doc1");
        assert_eq!(deserialized.score, 0.95);
    }
}
