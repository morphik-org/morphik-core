// Common types used across multiple components

export interface MorphikUIProps {
  connectionUri?: string;
  apiBaseUrl?: string;
  isReadOnlyUri?: boolean; // Controls whether the URI can be edited
  onUriChange?: (uri: string) => void; // Callback when URI is changed
  onBackClick?: () => void; // Callback when back button is clicked
  appName?: string; // Name of the app to display in UI
  initialFolder?: string | null; // Initial folder to show
}

export interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, unknown>;
  system_metadata: Record<string, unknown>;
  additional_metadata: Record<string, unknown>;
}

export interface Folder {
  id: string;
  name: string;
  description?: string;
  owner: string;
  document_ids: string[];
  system_metadata: Record<string, unknown>;
  access_control?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
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
  folder_name?: string;
}

export interface QueryOptions extends SearchOptions {
  max_tokens: number;
  temperature: number;
  graph_name?: string;
}