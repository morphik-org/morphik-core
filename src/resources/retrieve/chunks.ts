// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as ChunksAPI from './chunks';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';

export class Chunks extends APIResource {
  /**
   * Retrieve relevant chunks.
   *
   * Args: request: RetrieveRequest containing: - query: Search query text - filters:
   * Optional metadata filters - k: Number of results (default: 4) - min_score:
   * Minimum similarity threshold (default: 0.0) - use_reranking: Whether to use
   * reranking - use_colpali: Whether to use ColPali-style embedding model -
   * folder_name: Optional folder to scope the search to - end_user_id: Optional
   * end-user ID to scope the search to auth: Authentication context
   *
   * Returns: List[ChunkResult]: List of relevant chunks
   */
  create(body: ChunkCreateParams, options?: RequestOptions): APIPromise<ChunkCreateResponse> {
    return this._client.post('/retrieve/chunks', { body, ...options });
  }

  /**
   * Retrieve relevant chunks with grouped response format.
   *
   * Returns both flat results (for backward compatibility) and grouped results (for
   * UI). When padding > 0, groups chunks by main matches and their padding chunks.
   *
   * Args: request: RetrieveRequest containing query, filters, padding, etc. auth:
   * Authentication context
   *
   * Returns: GroupedChunkResponse: Contains both flat chunks and grouped chunks
   */
  createGrouped(
    body: ChunkCreateGroupedParams,
    options?: RequestOptions,
  ): APIPromise<ChunkCreateGroupedResponse> {
    return this._client.post('/retrieve/chunks/grouped', { body, ...options });
  }
}

/**
 * Query result at chunk level
 */
export interface ChunkResult {
  chunk_number: number;

  content: string;

  content_type: string;

  document_id: string;

  metadata: { [key: string]: unknown };

  score: number;

  download_url?: string | null;

  filename?: string | null;

  /**
   * Whether this chunk was added as padding
   */
  is_padding?: boolean;
}

/**
 * Base retrieve request model
 */
export interface RetrieveRequest {
  query: string;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  filters?: { [key: string]: unknown } | null;

  /**
   * Optional folder scope for the operation. Accepts a single folder name or a list
   * of folder names.
   */
  folder_name?: string | Array<string> | null;

  /**
   * Name of the graph to use for knowledge graph-enhanced retrieval
   */
  graph_name?: string | null;

  /**
   * Number of relationship hops to traverse in the graph
   */
  hop_depth?: number | null;

  /**
   * Whether to include relationship paths in the response
   */
  include_paths?: boolean | null;

  k?: number;

  min_score?: number;

  /**
   * Number of additional chunks/pages to retrieve before and after matched chunks
   * (ColPali only)
   */
  padding?: number;

  use_colpali?: boolean | null;

  use_reranking?: boolean | null;
}

export type ChunkCreateResponse = Array<ChunkResult>;

/**
 * Response that includes both flat results and grouped results for UI
 */
export interface ChunkCreateGroupedResponse {
  /**
   * Flat list of all chunks (for backward compatibility)
   */
  chunks: Array<ChunkResult>;

  /**
   * Grouped chunks for UI display
   */
  groups: Array<ChunkCreateGroupedResponse.Group>;

  /**
   * Whether padding was applied to any results
   */
  has_padding: boolean;

  /**
   * Total number of unique chunks
   */
  total_results: number;
}

export namespace ChunkCreateGroupedResponse {
  /**
   * Represents a group of chunks: one main match + its padding chunks
   */
  export interface Group {
    /**
     * Query result at chunk level
     */
    main_chunk: ChunksAPI.ChunkResult;

    /**
     * Total number of chunks in this group
     */
    total_chunks: number;

    padding_chunks?: Array<ChunksAPI.ChunkResult>;
  }
}

export interface ChunkCreateParams {
  query: string;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  filters?: { [key: string]: unknown } | null;

  /**
   * Optional folder scope for the operation. Accepts a single folder name or a list
   * of folder names.
   */
  folder_name?: string | Array<string> | null;

  /**
   * Name of the graph to use for knowledge graph-enhanced retrieval
   */
  graph_name?: string | null;

  /**
   * Number of relationship hops to traverse in the graph
   */
  hop_depth?: number | null;

  /**
   * Whether to include relationship paths in the response
   */
  include_paths?: boolean | null;

  k?: number;

  min_score?: number;

  /**
   * Number of additional chunks/pages to retrieve before and after matched chunks
   * (ColPali only)
   */
  padding?: number;

  use_colpali?: boolean | null;

  use_reranking?: boolean | null;
}

export interface ChunkCreateGroupedParams {
  query: string;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  filters?: { [key: string]: unknown } | null;

  /**
   * Optional folder scope for the operation. Accepts a single folder name or a list
   * of folder names.
   */
  folder_name?: string | Array<string> | null;

  /**
   * Name of the graph to use for knowledge graph-enhanced retrieval
   */
  graph_name?: string | null;

  /**
   * Number of relationship hops to traverse in the graph
   */
  hop_depth?: number | null;

  /**
   * Whether to include relationship paths in the response
   */
  include_paths?: boolean | null;

  k?: number;

  min_score?: number;

  /**
   * Number of additional chunks/pages to retrieve before and after matched chunks
   * (ColPali only)
   */
  padding?: number;

  use_colpali?: boolean | null;

  use_reranking?: boolean | null;
}

export declare namespace Chunks {
  export {
    type ChunkResult as ChunkResult,
    type RetrieveRequest as RetrieveRequest,
    type ChunkCreateResponse as ChunkCreateResponse,
    type ChunkCreateGroupedResponse as ChunkCreateGroupedResponse,
    type ChunkCreateParams as ChunkCreateParams,
    type ChunkCreateGroupedParams as ChunkCreateGroupedParams,
  };
}
