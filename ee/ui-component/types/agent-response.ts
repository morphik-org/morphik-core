export interface CitationBase {
  id: string;
  type: 'text' | 'image';
  sourceDocId: string;
  chunkId: string;
  snippet?: string; // Optional for images if only imageUrl is relevant initially
  grounded?: boolean;
  reasoning?: string; // Added reasoning
}

export interface TextCitation extends CitationBase {
  type: 'text';
  snippet: string; // Snippet is mandatory for text
}

export interface ImageCitation extends CitationBase {
  type: 'image';
  imageUrl: string;
  bbox?: [number, number, number, number]; // [x1, y1, x2, y2] normalized
}

export type Citation = TextCitation | ImageCitation;

export interface AgentRichResponse {
  mode: 'rich';
  body: string; // Markdown content
  citations: Citation[];
}

export interface AgentPlainResponse {
  mode: 'plain';
  body: string;
}

export type AgentResponseData = AgentRichResponse | AgentPlainResponse;

// Wrapper structure if the API returns { response: AgentResponseData, tool_history: any[] }
export interface AgentApiOutput {
  response: AgentResponseData;
  tool_history?: any[]; // Assuming tool_history is still relevant
}
