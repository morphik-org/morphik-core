// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as CustomModelsAPI from './custom-models';
import { CustomModel, CustomModelCreateParams, CustomModelListResponse, CustomModels } from './custom-models';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class ModelConfigResource extends APIResource {
  customModels: CustomModelsAPI.CustomModels = new CustomModelsAPI.CustomModels(this._client);

  /**
   * Create a new model configuration.
   */
  create(body: ModelConfigCreateParams, options?: RequestOptions): APIPromise<ModelConfig> {
    return this._client.post('/model-config/', { body, ...options });
  }

  /**
   * Get a specific model configuration.
   */
  retrieve(configID: string, options?: RequestOptions): APIPromise<ModelConfig> {
    return this._client.get(path`/model-config/${configID}`, options);
  }

  /**
   * Update an existing model configuration.
   */
  update(configID: string, body: ModelConfigUpdateParams, options?: RequestOptions): APIPromise<ModelConfig> {
    return this._client.put(path`/model-config/${configID}`, { body, ...options });
  }

  /**
   * List all model configurations for the authenticated user and app.
   */
  list(options?: RequestOptions): APIPromise<ModelConfigListResponse> {
    return this._client.get('/model-config/', options);
  }

  /**
   * Delete a model configuration.
   */
  delete(configID: string, options?: RequestOptions): APIPromise<ModelConfigDeleteResponse> {
    return this._client.delete(path`/model-config/${configID}`, options);
  }
}

/**
 * Response model for model configuration.
 */
export interface ModelConfig {
  id: string;

  config_data: { [key: string]: unknown };

  created_at: string;

  provider: string;

  updated_at: string;
}

export type ModelConfigListResponse = Array<ModelConfig>;

export type ModelConfigDeleteResponse = { [key: string]: string };

export interface ModelConfigCreateParams {
  config_data: { [key: string]: unknown };

  provider: string;
}

export interface ModelConfigUpdateParams {
  config_data: { [key: string]: unknown };
}

ModelConfigResource.CustomModels = CustomModels;

export declare namespace ModelConfigResource {
  export {
    type ModelConfig as ModelConfig,
    type ModelConfigListResponse as ModelConfigListResponse,
    type ModelConfigDeleteResponse as ModelConfigDeleteResponse,
    type ModelConfigCreateParams as ModelConfigCreateParams,
    type ModelConfigUpdateParams as ModelConfigUpdateParams,
  };

  export {
    CustomModels as CustomModels,
    type CustomModel as CustomModel,
    type CustomModelListResponse as CustomModelListResponse,
    type CustomModelCreateParams as CustomModelCreateParams,
  };
}
