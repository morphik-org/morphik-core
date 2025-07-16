// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';

export class Apps extends APIResource {
  /**
   * Provision a **brand-new** Neon database for _request.app_name_.
   *
   * The caller must be authenticated to the dedicated Morphik instance. The
   * authenticated user (represented by the JWT's _user_id_) becomes the owner of the
   * provisioned app.
   */
  create(body: AppCreateParams, options?: RequestOptions): APIPromise<AppCreateResponse> {
    return this._client.post('/ee/apps', { body, ...options });
  }

  /**
   * Destroy the Neon project and metadata associated with _app_name_.
   *
   * Only the owner of the application (identified via _auth.user_id_) may perform
   * this destructive action.
   */
  delete(params: AppDeleteParams, options?: RequestOptions): APIPromise<AppDeleteResponse> {
    const { app_name } = params;
    return this._client.delete('/ee/apps', { query: { app_name }, ...options });
  }
}

export interface AppCreateResponse {
  app_id: string;

  app_name: string;

  morphik_uri: string;

  status: string;
}

export interface AppDeleteResponse {
  app_name: string;

  status?: string;
}

export interface AppCreateParams {
  /**
   * Human-friendly name of the application to create
   */
  app_name: string;

  /**
   * Optional Neon region identifier (defaults to `aws-us-east-1`)
   */
  region?: string | null;
}

export interface AppDeleteParams {
  /**
   * Name of the application to delete
   */
  app_name: string;
}

export declare namespace Apps {
  export {
    type AppCreateResponse as AppCreateResponse,
    type AppDeleteResponse as AppDeleteResponse,
    type AppCreateParams as AppCreateParams,
    type AppDeleteParams as AppDeleteParams,
  };
}
