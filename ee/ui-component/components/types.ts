import type { UIMessage } from "./chat/ChatMessages";

// Define option types used in callbacks
export interface SearchOptions {
  k?: number;
  min_score?: number;
  filters?: string | object; // JSON string or object with external_id array
  use_reranking?: boolean;
  use_colpali?: boolean;
  padding?: number; // Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only)
  /**
   * Optional folder scoping for retrieval endpoints.
   */
  folder_name?: string | string[];
}

export interface QueryOptions extends SearchOptions {
  max_tokens?: number;
  temperature?: number;
  graph_name?: string;
  folder_name?: string | string[]; // Support single folder or array of folders
  // external_id removed - should be in filters object as external_id: string[]
  llm_config?: Record<string, unknown>; // LiteLLM-compatible model configuration
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
  initialSection?:
    | "documents"
    | "search"
    | "chat"
    | "graphs"
    | "workflows"
    | "connections"
    | "pdf"
    | "settings"
    | "logs"; // Initial section to show

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

  // User profile and auth
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;

  // UI Customization
  logoLight?: string;
  logoDark?: string;
}

export interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, unknown>;
  system_metadata: Record<string, unknown>;
  additional_metadata: Record<string, unknown>;
  folder_name?: string;
  app_id?: string;
  end_user_id?: string;
}

export interface FolderSummary {
  id: string;
  name: string;
  description?: string;
  doc_count?: number;
  updated_at?: string;
}

export interface Folder extends FolderSummary {
  document_ids?: string[];
  system_metadata: Record<string, unknown>;
  created_at?: string;
  app_id?: string;
  end_user_id?: string;
  // updated_at inherited
}

export interface SearchResult {
  document_id: string;
  chunk_number: number;
  content: string;
  content_type: string;
  score: number;
  filename?: string;
  metadata: Record<string, unknown>;
  is_padding?: boolean; // Whether this chunk was added as padding
}

export interface ChunkGroup {
  main_chunk: SearchResult;
  padding_chunks: SearchResult[];
  total_chunks: number;
}

export interface GroupedSearchResponse {
  chunks: SearchResult[]; // Flat list for backward compatibility
  groups: ChunkGroup[]; // Grouped chunks for UI display
  total_results: number;
  has_padding: boolean;
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
  timestamp?: string;
  sources?: Source[];
}

// Model Configuration Types
export interface ModelConfigResponse {
  id: string;
  provider: string;
  config_data: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ModelConfigCreate {
  provider: string;
  config_data: Record<string, unknown>;
}

export interface ModelConfigUpdate {
  config_data: Record<string, unknown>;
}

export interface CustomModel {
  id: string;
  name: string;
  provider: string;
  model_name: string;
  config: Record<string, unknown>;
}

export interface CustomModelCreate {
  name: string;
  provider: string;
  model_name: string;
  config: Record<string, unknown>;
}
