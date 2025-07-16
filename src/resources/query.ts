// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import * as CacheAPI from './cache';
import * as GraphAPI from './graph/graph';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Query extends APIResource {
  /**
   * Generate completion using relevant chunks as context.
   *
   * When graph_name is provided, the query will leverage the knowledge graph to
   * enhance retrieval by finding relevant entities and their connected documents.
   *
   * Args: request: CompletionQueryRequest containing: - query: Query text - filters:
   * Optional metadata filters - k: Number of chunks to use as context (default: 4) -
   * min_score: Minimum similarity threshold (default: 0.0) - max_tokens: Maximum
   * tokens in completion - temperature: Model temperature - use_reranking: Whether
   * to use reranking - use_colpali: Whether to use ColPali-style embedding model -
   * graph_name: Optional name of the graph to use for knowledge graph-enhanced
   * retrieval - hop_depth: Number of relationship hops to traverse in the graph
   * (1-3) - include_paths: Whether to include relationship paths in the response -
   * prompt_overrides: Optional customizations for entity extraction, resolution, and
   * query prompts - folder_name: Optional folder to scope the operation to -
   * end_user_id: Optional end-user ID to scope the operation to - schema: Optional
   * schema for structured output - chat_id: Optional chat conversation identifier
   * for maintaining history auth: Authentication context
   *
   * Returns: CompletionResponse: Generated text completion or structured output
   */
  generateCompletion(
    body: QueryGenerateCompletionParams,
    options?: RequestOptions,
  ): APIPromise<CacheAPI.CompletionResponse> {
    return this._client.post('/query', { body, ...options });
  }
}

export interface QueryGenerateCompletionParams {
  query: string;

  /**
   * Optional chat session ID for persisting conversation history
   */
  chat_id?: string | null;

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

  /**
   * LiteLLM-compatible model configuration (e.g., model name, API key, base URL)
   */
  llm_config?: { [key: string]: unknown } | null;

  max_tokens?: number | null;

  min_score?: number;

  /**
   * Number of additional chunks/pages to retrieve before and after matched chunks
   * (ColPali only)
   */
  padding?: number;

  /**
   * Container for query-related prompt overrides.
   *
   * Use this class when customizing prompts for query operations, which may include
   * customizations for entity extraction, entity resolution, and the query/response
   * generation itself.
   *
   * This is the most feature-complete override class, supporting all customization
   * types.
   *
   * Available customizations:
   *
   * - entity_extraction: Customize how entities are identified in text
   * - entity_resolution: Customize how entity variants are grouped
   * - query: Customize response generation style, format, and tone
   *
   * Each type has its own required placeholders. See the specific class
   * documentation for details and examples.
   */
  prompt_overrides?: QueryGenerateCompletionParams.PromptOverrides | null;

  /**
   * Schema for structured output, can be a Pydantic model or JSON schema dict
   */
  schema?: unknown | { [key: string]: unknown } | null;

  /**
   * Whether to stream the response back in chunks
   */
  stream_response?: boolean | null;

  temperature?: number | null;

  use_colpali?: boolean | null;

  use_reranking?: boolean | null;
}

export namespace QueryGenerateCompletionParams {
  /**
   * Container for query-related prompt overrides.
   *
   * Use this class when customizing prompts for query operations, which may include
   * customizations for entity extraction, entity resolution, and the query/response
   * generation itself.
   *
   * This is the most feature-complete override class, supporting all customization
   * types.
   *
   * Available customizations:
   *
   * - entity_extraction: Customize how entities are identified in text
   * - entity_resolution: Customize how entity variants are grouped
   * - query: Customize response generation style, format, and tone
   *
   * Each type has its own required placeholders. See the specific class
   * documentation for details and examples.
   */
  export interface PromptOverrides {
    /**
     * Configuration for customizing entity extraction prompts.
     *
     * This allows you to override both the prompt template used for entity extraction
     * and provide domain-specific examples of entities to be extracted.
     *
     * If only examples are provided (without a prompt_template), they will be
     * incorporated into the default prompt. If only prompt_template is provided, it
     * will be used with default examples (if any).
     *
     * Required placeholders:
     *
     * - {content}: Will be replaced with the text to analyze for entity extraction
     * - {examples}: Will be replaced with formatted examples of entities to extract
     *
     * Example prompt template:
     *
     * ```
     * Extract entities from the following text. Look for entities similar to these examples:
     *
     * {examples}
     *
     * Text to analyze:
     * {content}
     *
     * Extracted entities (in JSON format):
     * ```
     */
    entity_extraction?: GraphAPI.EntityExtractionPromptOverride | null;

    /**
     * Configuration for customizing entity resolution prompts.
     *
     * Entity resolution identifies and groups variant forms of the same entity. This
     * override allows you to customize how this process works by providing a custom
     * prompt template and/or domain-specific examples.
     *
     * If only examples are provided (without a prompt_template), they will be
     * incorporated into the default prompt. If only prompt_template is provided, it
     * will be used with default examples (if any).
     *
     * Required placeholders:
     *
     * - {entities_str}: Will be replaced with the extracted entities
     * - {examples_json}: Will be replaced with JSON-formatted examples of entity
     *   resolution groups
     *
     * Example prompt template:
     *
     * ```
     * I have extracted the following entities:
     *
     * {entities_str}
     *
     * Below are examples of how different entity references can be grouped together:
     *
     * {examples_json}
     *
     * Group the above entities by resolving which mentions refer to the same entity.
     * Return the results in JSON format.
     * ```
     */
    entity_resolution?: GraphAPI.EntityResolutionPromptOverride | null;

    /**
     * Configuration for customizing query prompts.
     *
     * This allows you to customize how responses are generated during query
     * operations. Query prompts guide the LLM on how to format and style responses,
     * what tone to use, and how to incorporate retrieved information into the
     * response.
     *
     * Required placeholders:
     *
     * - {question}: Will be replaced with the user's query
     * - {context}: Will be replaced with the retrieved content/context
     *
     * Example prompt template:
     *
     * ```
     * Answer the following question based on the provided information.
     *
     * Question: {question}
     *
     * Context:
     * {context}
     *
     * Answer:
     * ```
     */
    query?: PromptOverrides.Query | null;
  }

  export namespace PromptOverrides {
    /**
     * Configuration for customizing query prompts.
     *
     * This allows you to customize how responses are generated during query
     * operations. Query prompts guide the LLM on how to format and style responses,
     * what tone to use, and how to incorporate retrieved information into the
     * response.
     *
     * Required placeholders:
     *
     * - {question}: Will be replaced with the user's query
     * - {context}: Will be replaced with the retrieved content/context
     *
     * Example prompt template:
     *
     * ```
     * Answer the following question based on the provided information.
     *
     * Question: {question}
     *
     * Context:
     * {context}
     *
     * Answer:
     * ```
     */
    export interface Query {
      /**
       * Custom prompt template for generating responses to queries. REQUIRED
       * PLACEHOLDERS: {question} and {context} must be included in the template. The
       * {question} placeholder will be replaced with the user query, and {context} will
       * be replaced with the retrieved content. Use this to control response style,
       * format, and tone.
       */
      prompt_template?: string | null;
    }
  }
}

export declare namespace Query {
  export { type QueryGenerateCompletionParams as QueryGenerateCompletionParams };
}
