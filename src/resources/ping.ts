// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Ping extends APIResource {
  /**
   * Simple health check endpoint that returns 200 OK.
   */
  check(options?: RequestOptions): APIPromise<PingCheckResponse> {
    return this._client.get('/ping', options);
  }
}

/**
 * Response for health check endpoint
 */
export interface PingCheckResponse {
  message: string;

  status: string;
}

export declare namespace Ping {
  export { type PingCheckResponse as PingCheckResponse };
}
