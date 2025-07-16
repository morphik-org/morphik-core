// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';

export class CustomModels extends APIResource {
  /**
   * Create a new custom model.
   */
  create(body: CustomModelCreateParams, options?: RequestOptions): APIPromise<CustomModel> {
    return this._client.post('/model-config/custom-models', { body, ...options });
  }

  /**
   * List all custom models for the authenticated user.
   */
  list(options?: RequestOptions): APIPromise<CustomModelListResponse> {
    return this._client.get('/model-config/custom-models/list', options);
  }
}

/**
 * Custom model definition.
 */
export interface CustomModel {
  id: string;

  config: { [key: string]: unknown };

  model_name: string;

  name: string;

  provider: string;
}

export type CustomModelListResponse = Array<CustomModel>;

export interface CustomModelCreateParams {
  config: { [key: string]: unknown };

  model_name: string;

  name: string;

  provider: string;
}

export declare namespace CustomModels {
  export {
    type CustomModel as CustomModel,
    type CustomModelListResponse as CustomModelListResponse,
    type CustomModelCreateParams as CustomModelCreateParams,
  };
}
