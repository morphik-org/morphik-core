// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class APIKeys extends APIResource {
  /**
   * Save API key for a provider.
   */
  create(body: APIKeyCreateParams, options?: RequestOptions): APIPromise<APIKeyCreateResponse> {
    return this._client.post('/api-keys', { body, ...options });
  }

  /**
   * List all configured API keys (sanitized).
   */
  list(options?: RequestOptions): APIPromise<APIKeyListResponse> {
    return this._client.get('/api-keys', options);
  }
}

export type APIKeyCreateResponse = { [key: string]: string };

export type APIKeyListResponse = { [key: string]: { [key: string]: unknown } };

export interface APIKeyCreateParams {
  api_key: string;

  provider: string;

  base_url?: string | null;
}

export declare namespace APIKeys {
  export {
    type APIKeyCreateResponse as APIKeyCreateResponse,
    type APIKeyListResponse as APIKeyListResponse,
    type APIKeyCreateParams as APIKeyCreateParams,
  };
}
