import type { UIMessage } from "./chat/ChatMessages";

// Define option types used in callbacks
export interface SearchOptions {
  k?: number;
  min_score?: number;
  filters?: string; // JSON string or object?
  use_reranking?: boolean;
  use_colpali?: boolean;
}

export interface QueryOptions extends SearchOptions {
  max_tokens?: number;
  temperature?: number;
  graph_name?: string;
  folder_name?: string;
  // Note: Consider if streaming or other UI options should be here
}

// Common types used across multiple components

export interface MorphikUIProps {
  connectionUri?: string | null; // Allow null/undefined initially
  apiBaseUrl?: string;
  isReadOnlyUri?: boolean; // Controls whether the URI can be edited
  onUriChange?: (newUri: string) => void; // Callback when URI is changed
  onBackClick?: () => void; // Callback when back button is clicked
  appName?: string; // Name of the app to display in UI
  initialFolder?: string | null; // Initial folder to show
  initialSection?: "documents" | "search" | "chat" | "graphs" | "agent"; // Initial section to show

  // Callbacks for Documents Section tracking
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderCreate?: (folderName: string) => void;
  onFolderDelete?: (folderName: string) => void;
  onFolderClick?: (folderName: string | null) => void; // Allow null

  // Callbacks for Search and Chat tracking
  onSearchSubmit?: (query: string, options: SearchOptions) => void;
  onChatSubmit?: (query: string, options: QueryOptions, initialMessages?: UIMessage[]) => void; // Use UIMessage[]

  // Callback for Agent Chat tracking
  onAgentSubmit?: (query: string) => void;

  // Callbacks for Graph tracking
  onGraphClick?: (graphName: string | undefined) => void;
  onGraphCreate?: (graphName: string, numDocuments: number) => void;
  onGraphUpdate?: (graphName: string, numAdditionalDocuments: number) => void;
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

export interface Source {
  document_id: string;
  chunk_number: number;
  score?: number;
  filename?: string;
  content?: string;
  content_type?: string;
  metadata?: Record<string, unknown>;
  download_url?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}
