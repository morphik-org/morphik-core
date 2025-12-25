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
  /**
   * Optional nesting depth for folder scoping (-1 for all descendants).
   */
  folder_depth?: number;
  /**
   * Base64-encoded image for visual search (requires use_colpali=true).
   */
  query_image?: string;
}

export interface QueryOptions extends SearchOptions {
  max_tokens?: number;
  temperature?: number;
  graph_name?: string;
  folder_name?: string | string[]; // Support single folder or array of folders
  folder_depth?: number;
  // external_id removed - should be in filters object as external_id: string[]
  llm_config?: Record<string, unknown>; // LiteLLM-compatible model configuration
  inline_citations?: boolean; // Whether to include inline citations with filename and page number
}

// Common types used across multiple components

// Breadcrumb type for custom navigation
export interface Breadcrumb {
  label: string;
  href?: string;
  onClick?: (e: React.MouseEvent) => void;
  current?: boolean;
}

export interface MorphikUIProps {
  connectionUri?: string | null; // Allow null/undefined initially
  apiBaseUrl?: string;
  appId?: string; // App ID for tracking and limits
  isReadOnlyUri?: boolean; // Controls whether the URI can be edited
  onBackClick?: () => void; // Callback when back button is clicked
  appName?: string; // Name of the app to display in UI
  initialFolder?: string | null; // Initial folder to show
  initialSection?: "documents" | "search" | "chat" | "graphs" | "connections" | "pdf" | "settings" | "logs"; // Initial section to show

  // Custom breadcrumbs for organization context
  breadcrumbItems?: Breadcrumb[];

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
  onUpgradeClick?: () => void;

  // UI Customization
  logoLight?: string;
  logoDark?: string;
}

export interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, unknown>;
  system_metadata: SystemMetadata;
  additional_metadata: Record<string, unknown>;
  folder_path?: string;
  folder_name?: string;
  app_id?: string;
  end_user_id?: string;
}

export interface SystemMetadata {
  status?: string;
  progress?: ProcessingProgress;
  created_at?: string;
  updated_at?: string;
  error?: string;
  [key: string]: unknown;
}

export interface FolderSummary {
  id: string;
  name: string;
  full_path?: string;
  parent_id?: string | null;
  depth?: number | null;
  child_count?: number | null;
  description?: string;
  doc_count?: number;
  updated_at?: string;
  document_ids?: string[];
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

// Progress tracking interface for document processing
export interface ProcessingProgress {
  step_name: string;
  current_step: number;
  total_steps: number;
  percentage?: number;
}
