// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import * as QueryAPI from './query';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';
import { path } from '../internal/utils/path';

export class Graphs extends APIResource {
  /**
   * Create a new graph based on document contents.
   *
   * The graph is created asynchronously. A stub graph record is returned with
   * `status = "processing"` while a background task extracts entities and
   * relationships.
   *
   * Args: request: Graph creation parameters including name and optional filters.
   * auth: Authentication context authorizing the operation.
   *
   * Returns: The placeholder :class:`Graph` object which clients can poll for
   * status.
   *
   * @example
   * ```ts
   * const graph = await client.graphs.create({ name: 'name' });
   * ```
   */
  create(body: GraphCreateParams, options?: RequestOptions): APIPromise<Graph> {
    return this._client.post('/graph/create', { body, ...options });
  }

  /**
   * Get a graph by name.
   *
   * This endpoint retrieves a graph by its name if the user has access to it.
   *
   * Args: name: Name of the graph to retrieve auth: Authentication context
   * folder_name: Optional folder to scope the operation to end_user_id: Optional
   * end-user ID to scope the operation to
   *
   * Returns: Graph: The requested graph object
   *
   * @example
   * ```ts
   * const graph = await client.graphs.retrieve('name');
   * ```
   */
  retrieve(
    name: string,
    query: GraphRetrieveParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<Graph> {
    return this._client.get(path`/graph/${name}`, { query, ...options });
  }

  /**
   * Update an existing graph with new documents.
   *
   * This endpoint processes additional documents based on the original graph filters
   * and/or new filters/document IDs, extracts entities and relationships, and
   * updates the graph with new information.
   *
   * Args: name: Name of the graph to update request: UpdateGraphRequest
   * containing: - additional_filters: Optional additional metadata filters to
   * determine which new documents to include - additional_documents: Optional list
   * of additional document IDs to include - prompt_overrides: Optional
   * customizations for entity extraction and resolution prompts - folder_name:
   * Optional folder to scope the operation to - end_user_id: Optional end-user ID to
   * scope the operation to auth: Authentication context
   *
   * Returns: Graph: The updated graph object
   *
   * @example
   * ```ts
   * const graph = await client.graphs.update('name');
   * ```
   */
  update(name: string, body: GraphUpdateParams, options?: RequestOptions): APIPromise<Graph> {
    return this._client.post(path`/graph/${name}/update`, { body, ...options });
  }

  /**
   * List all graphs the user has access to.
   *
   * This endpoint retrieves all graphs the user has access to.
   *
   * Args: auth: Authentication context folder_name: Optional folder to scope the
   * operation to end_user_id: Optional end-user ID to scope the operation to
   *
   * Returns: List[Graph]: List of graph objects
   *
   * @example
   * ```ts
   * const graphs = await client.graphs.list();
   * ```
   */
  list(
    query: GraphListParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<GraphListResponse> {
    return this._client.get('/graphs', { query, ...options });
  }
}

/**
 * Represents a knowledge graph
 */
export interface Graph {
  name: string;

  id?: string;

  app_id?: string | null;

  created_at?: string;

  document_ids?: Array<string>;

  end_user_id?: string | null;

  entities?: Array<Graph.Entity>;

  filters?: { [key: string]: unknown } | null;

  folder_name?: string | null;

  metadata?: { [key: string]: unknown };

  relationships?: Array<Graph.Relationship>;

  system_metadata?: { [key: string]: unknown };

  updated_at?: string;
}

export namespace Graph {
  /**
   * Represents an entity in a knowledge graph
   */
  export interface Entity {
    label: string;

    type: string;

    id?: string;

    chunk_sources?: { [key: string]: Array<number> };

    document_ids?: Array<string>;

    properties?: { [key: string]: unknown };
  }

  /**
   * Represents a relationship between entities in a knowledge graph
   */
  export interface Relationship {
    source_id: string;

    target_id: string;

    type: string;

    id?: string;

    chunk_sources?: { [key: string]: Array<number> };

    document_ids?: Array<string>;
  }
}

/**
 * Container for graph-related prompt overrides.
 *
 * Use this class when customizing prompts for graph operations like create_graph()
 * and update_graph(), which only support entity extraction and entity resolution
 * customizations.
 *
 * This class enforces that only graph-relevant override types are used.
 */
export interface GraphPromptOverrides {
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
  entity_extraction?: QueryAPI.EntityExtractionPromptOverride | null;

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
  entity_resolution?: QueryAPI.EntityResolutionPromptOverride | null;
}

export type GraphListResponse = Array<Graph>;

export interface GraphCreateParams {
  /**
   * Name of the graph to create
   */
  name: string;

  /**
   * Optional list of specific document IDs to include
   */
  documents?: Array<string> | null;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  /**
   * Optional metadata filters to determine which documents to include
   */
  filters?: { [key: string]: unknown } | null;

  /**
   * Optional folder scope for the operation. Accepts a single folder name or a list
   * of folder names.
   */
  folder_name?: string | Array<string> | null;

  /**
   * Container for graph-related prompt overrides.
   *
   * Use this class when customizing prompts for graph operations like create_graph()
   * and update_graph(), which only support entity extraction and entity resolution
   * customizations.
   *
   * This class enforces that only graph-relevant override types are used.
   */
  prompt_overrides?: GraphPromptOverrides | null;
}

export interface GraphRetrieveParams {
  end_user_id?: string | null;
}

export interface GraphUpdateParams {
  /**
   * Optional list of additional document IDs to include
   */
  additional_documents?: Array<string> | null;

  /**
   * Optional additional metadata filters to determine which new documents to include
   */
  additional_filters?: { [key: string]: unknown } | null;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  /**
   * Optional folder scope for the operation. Accepts a single folder name or a list
   * of folder names.
   */
  folder_name?: string | Array<string> | null;

  /**
   * Container for graph-related prompt overrides.
   *
   * Use this class when customizing prompts for graph operations like create_graph()
   * and update_graph(), which only support entity extraction and entity resolution
   * customizations.
   *
   * This class enforces that only graph-relevant override types are used.
   */
  prompt_overrides?: GraphPromptOverrides | null;
}

export interface GraphListParams {
  end_user_id?: string | null;
}

export declare namespace Graphs {
  export {
    type Graph as Graph,
    type GraphPromptOverrides as GraphPromptOverrides,
    type GraphListResponse as GraphListResponse,
    type GraphCreateParams as GraphCreateParams,
    type GraphRetrieveParams as GraphRetrieveParams,
    type GraphUpdateParams as GraphUpdateParams,
    type GraphListParams as GraphListParams,
  };
}
