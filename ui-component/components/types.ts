// Common types used across multiple components

export interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, unknown>;
  system_metadata: Record<string, unknown>;
  additional_metadata: Record<string, unknown>;
}

export interface SearchResult {
  document_id: string;
  chunk_number: number;
  content: string;
  content_type: string;
  score: number;
  filename?: string;
  metadata: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface SearchOptions {
  filters: string;
  k: number;
  min_score: number;
  use_reranking: boolean;
  use_colpali: boolean;
}

export interface QueryOptions extends SearchOptions {
  max_tokens: number;
  temperature: number;
  graph_name?: string;
}