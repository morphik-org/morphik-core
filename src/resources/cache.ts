// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';
import { path } from '../internal/utils/path';

export class Cache extends APIResource {
  /**
   * Create a persistent cache for low-latency completions.
   *
   * Args: name: Unique identifier for the cache. model: The model name to use when
   * generating completions. gguf_file: Path to the `gguf` weights file to load.
   * filters: Optional metadata filters used to select documents. docs: Explicit list
   * of document IDs to include in the cache. auth: Authentication context used for
   * permission checks.
   *
   * Returns: A dictionary describing the created cache.
   */
  create(params: CacheCreateParams, options?: RequestOptions): APIPromise<CacheCreateResponse> {
    const { gguf_file, model, name, ...body } = params;
    return this._client.post('/cache/create', { query: { gguf_file, model, name }, body, ...options });
  }

  /**
   * Retrieve information about a specific cache.
   *
   * Args: name: Name of the cache to inspect. auth: Authentication context used to
   * authorize the request.
   *
   * Returns: A dictionary with a boolean `exists` field indicating whether the cache
   * is loaded.
   */
  retrieve(name: string, options?: RequestOptions): APIPromise<CacheRetrieveResponse> {
    return this._client.get(path`/cache/${name}`, options);
  }

  /**
   * Refresh an existing cache with newly available documents.
   *
   * Args: name: Identifier of the cache to update. auth: Authentication context used
   * for permission checks.
   *
   * Returns: A dictionary indicating whether any documents were added.
   */
  update(name: string, options?: RequestOptions): APIPromise<CacheUpdateResponse> {
    return this._client.post(path`/cache/${name}/update`, options);
  }

  /**
   * Manually add documents to an existing cache.
   *
   * Args: name: Name of the target cache. document_ids: List of document IDs to
   * insert. auth: Authentication context used for authorization.
   *
   * Returns: A dictionary indicating whether the documents were queued for addition.
   */
  addDocs(
    name: string,
    params: CacheAddDocsParams,
    options?: RequestOptions,
  ): APIPromise<CacheAddDocsResponse> {
    const { body } = params;
    return this._client.post(path`/cache/${name}/add_docs`, { body: body, ...options });
  }

  /**
   * Generate a completion using a pre-populated cache.
   *
   * Args: name: Name of the cache to query. query: Prompt text to send to the model.
   * max_tokens: Optional maximum number of tokens to generate. temperature: Optional
   * sampling temperature for the model. auth: Authentication context for permission
   * checks.
   *
   * Returns: A :class:`CompletionResponse` object containing the model output.
   */
  query(name: string, params: CacheQueryParams, options?: RequestOptions): APIPromise<CompletionResponse> {
    const { query, max_tokens, temperature } = params;
    return this._client.post(path`/cache/${name}/query`, {
      query: { query, max_tokens, temperature },
      ...options,
    });
  }
}

/**
 * Response from completion generation
 */
export interface CompletionResponse {
  /**
   * Structured completion object for schema-based responses
   */
  completion: string | { [key: string]: unknown };

  usage: { [key: string]: number };

  finish_reason?: string | null;

  metadata?: { [key: string]: unknown } | null;

  sources?: Array<CompletionResponse.Source>;
}

export namespace CompletionResponse {
  /**
   * Source information for a chunk used in completion
   */
  export interface Source {
    chunk_number: number;

    document_id: string;

    score?: number | null;
  }
}

export type CacheCreateResponse = { [key: string]: unknown };

export type CacheRetrieveResponse = { [key: string]: unknown };

export type CacheUpdateResponse = { [key: string]: boolean };

export type CacheAddDocsResponse = { [key: string]: boolean };

export interface CacheCreateParams {
  /**
   * Query param:
   */
  gguf_file: string;

  /**
   * Query param:
   */
  model: string;

  /**
   * Query param:
   */
  name: string;

  /**
   * Body param:
   */
  docs?: Array<string> | null;

  /**
   * Body param:
   */
  filters?: { [key: string]: unknown } | null;
}

export interface CacheAddDocsParams {
  body: Array<string>;
}

export interface CacheQueryParams {
  query: string;

  max_tokens?: number | null;

  temperature?: number | null;
}

export declare namespace Cache {
  export {
    type CompletionResponse as CompletionResponse,
    type CacheCreateResponse as CacheCreateResponse,
    type CacheRetrieveResponse as CacheRetrieveResponse,
    type CacheUpdateResponse as CacheUpdateResponse,
    type CacheAddDocsResponse as CacheAddDocsResponse,
    type CacheCreateParams as CacheCreateParams,
    type CacheAddDocsParams as CacheAddDocsParams,
    type CacheQueryParams as CacheQueryParams,
  };
}
