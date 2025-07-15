// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Cloud extends APIResource {
  /**
   * Delete all resources associated with a given cloud application.
   *
   * Args: app_name: Name of the application whose data should be removed. auth:
   * Authentication context of the requesting user.
   *
   * Returns: A summary describing how many documents and folders were removed.
   */
  deleteApps(params: CloudDeleteAppsParams, options?: RequestOptions): APIPromise<CloudDeleteAppsResponse> {
    const { app_name } = params;
    return this._client.delete('/cloud/apps', { query: { app_name }, ...options });
  }

  /**
   * Generate an authenticated URI for a cloud-hosted Morphik application.
   *
   * Args: request: Parameters for URI generation including `app_id` and `name`.
   * authorization: Bearer token of the user requesting the URI.
   *
   * Returns: A dictionary with the generated `uri` and associated `app_id`.
   */
  generateUri(body: CloudGenerateUriParams, options?: RequestOptions): APIPromise<CloudGenerateUriResponse> {
    return this._client.post('/cloud/generate_uri', { body, ...options });
  }
}

export type CloudDeleteAppsResponse = { [key: string]: unknown };

export type CloudGenerateUriResponse = { [key: string]: string };

export interface CloudDeleteAppsParams {
  /**
   * Name of the application to delete
   */
  app_name: string;
}

export interface CloudGenerateUriParams {
  /**
   * ID of the application
   */
  app_id: string;

  /**
   * Name of the application
   */
  name: string;

  /**
   * ID of the user who owns the app
   */
  user_id: string;

  /**
   * Number of days until the token expires
   */
  expiry_days?: number;
}

export declare namespace Cloud {
  export {
    type CloudDeleteAppsResponse as CloudDeleteAppsResponse,
    type CloudGenerateUriResponse as CloudGenerateUriResponse,
    type CloudDeleteAppsParams as CloudDeleteAppsParams,
    type CloudGenerateUriParams as CloudGenerateUriParams,
  };
}
