// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';
import { path } from '../internal/utils/path';

export class Models extends APIResource {
  /**
   * Save a custom model configuration.
   */
  create(body: ModelCreateParams, options?: RequestOptions): APIPromise<ModelResponse> {
    return this._client.post('/models', { body, ...options });
  }

  /**
   * Get list of available models from configuration.
   *
   * Returns models grouped by type (chat, embedding, etc.) with their metadata.
   */
  list(options?: RequestOptions): APIPromise<ModelListResponse> {
    return this._client.get('/models', options);
  }

  /**
   * Delete a custom model.
   */
  delete(modelID: string, options?: RequestOptions): APIPromise<ModelDeleteResponse> {
    return this._client.delete(path`/models/${modelID}`, options);
  }

  /**
   * Get list of available models for UI selection.
   *
   * Returns a list of models that can be used for queries. Each model includes:
   *
   * - id: Model identifier to use in llm_config
   * - name: Display name for the model
   * - provider: The LLM provider (e.g., openai, anthropic, ollama)
   * - description: Optional description of the model
   */
  listAvailable(options?: RequestOptions): APIPromise<unknown> {
    return this._client.get('/models/available', options);
  }

  /**
   * List all custom models for the authenticated user.
   */
  listCustom(options?: RequestOptions): APIPromise<ModelListCustomResponse> {
    return this._client.get('/models/custom', options);
  }
}

/**
 * Response for a saved model.
 */
export interface ModelResponse {
  id: string;

  config: { [key: string]: unknown };

  created_at: string;

  name: string;

  provider: string;

  updated_at: string;
}

/**
 * Response for available models endpoint
 */
export interface ModelListResponse {
  chat_models: Array<{ [key: string]: unknown }>;

  default_models: { [key: string]: string | null };

  embedding_models: Array<{ [key: string]: unknown }>;

  providers: Array<string>;
}

export type ModelDeleteResponse = { [key: string]: string };

export type ModelListAvailableResponse = unknown;

export type ModelListCustomResponse = Array<ModelResponse>;

export interface ModelCreateParams {
  config: { [key: string]: unknown };

  name: string;

  provider: string;
}

export declare namespace Models {
  export {
    type ModelResponse as ModelResponse,
    type ModelListResponse as ModelListResponse,
    type ModelDeleteResponse as ModelDeleteResponse,
    type ModelListAvailableResponse as ModelListAvailableResponse,
    type ModelListCustomResponse as ModelListCustomResponse,
    type ModelCreateParams as ModelCreateParams,
  };
}
