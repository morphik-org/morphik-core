// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Retrieve extends APIResource {
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
  retrieveChunks(
    body: RetrieveRetrieveChunksParams,
    options?: RequestOptions,
  ): APIPromise<RetrieveRetrieveChunksResponse> {
    return this._client.post('/retrieve/chunks', { body, ...options });
  }

  /**
   * Retrieve relevant documents.
   *
   * Args: request: RetrieveRequest containing: - query: Search query text - filters:
   * Optional metadata filters - k: Number of results (default: 4) - min_score:
   * Minimum similarity threshold (default: 0.0) - use_reranking: Whether to use
   * reranking - use_colpali: Whether to use ColPali-style embedding model -
   * folder_name: Optional folder to scope the search to - end_user_id: Optional
   * end-user ID to scope the search to auth: Authentication context
   *
   * Returns: List[DocumentResult]: List of relevant documents
   */
  retrieveDocs(
    body: RetrieveRetrieveDocsParams,
    options?: RequestOptions,
  ): APIPromise<RetrieveRetrieveDocsResponse> {
    return this._client.post('/retrieve/docs', { body, ...options });
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

export type RetrieveRetrieveChunksResponse = Array<ChunkResult>;

export type RetrieveRetrieveDocsResponse =
  Array<RetrieveRetrieveDocsResponse.RetrieveRetrieveDocsResponseItem>;

export namespace RetrieveRetrieveDocsResponse {
  /**
   * Query result at document level
   */
  export interface RetrieveRetrieveDocsResponseItem {
    additional_metadata: { [key: string]: unknown };

    /**
     * Represents either a URL or content string
     */
    content: RetrieveRetrieveDocsResponseItem.Content;

    document_id: string;

    metadata: { [key: string]: unknown };

    score: number;
  }

  export namespace RetrieveRetrieveDocsResponseItem {
    /**
     * Represents either a URL or content string
     */
    export interface Content {
      type: 'url' | 'string';

      value: string;

      /**
       * Filename when type is url
       */
      filename?: string | null;
    }
  }
}

export interface RetrieveRetrieveChunksParams {
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

export interface RetrieveRetrieveDocsParams {
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

export declare namespace Retrieve {
  export {
    type ChunkResult as ChunkResult,
    type RetrieveRequest as RetrieveRequest,
    type RetrieveRetrieveChunksResponse as RetrieveRetrieveChunksResponse,
    type RetrieveRetrieveDocsResponse as RetrieveRetrieveDocsResponse,
    type RetrieveRetrieveChunksParams as RetrieveRetrieveChunksParams,
    type RetrieveRetrieveDocsParams as RetrieveRetrieveDocsParams,
  };
}
