// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Agent extends APIResource {
  /**
   * Execute an agent-style query using the :class:`MorphikAgent`.
   *
   * Args: request: The query payload containing the natural language question and
   * optional chat_id. auth: Authentication context used to enforce limits and access
   * control. redis: Redis connection for chat history storage.
   *
   * Returns: A dictionary with the agent's full response.
   */
  executeQuery(
    body: AgentExecuteQueryParams,
    options?: RequestOptions,
  ): APIPromise<AgentExecuteQueryResponse> {
    return this._client.post('/agent', { body, ...options });
  }
}

export type AgentExecuteQueryResponse = { [key: string]: unknown };

export interface AgentExecuteQueryParams {
  /**
   * Natural language query for the Morphik agent
   */
  query: string;

  /**
   * Optional chat session ID for persisting conversation history
   */
  chat_id?: string | null;

  /**
   * Display mode for images: 'formatted' (default) creates bounding boxes with
   * Gemini, 'raw' returns uncropped images
   */
  display_mode?: 'formatted' | 'raw';
}

export declare namespace Agent {
  export {
    type AgentExecuteQueryResponse as AgentExecuteQueryResponse,
    type AgentExecuteQueryParams as AgentExecuteQueryParams,
  };
}
