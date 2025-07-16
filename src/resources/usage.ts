// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Usage extends APIResource {
  /**
   * Retrieve recent telemetry records for the user or application.
   *
   * Args: auth: Authentication context; admin users receive global records.
   * operation_type: Optional operation type to filter by. since: Only return records
   * newer than this timestamp. status: Optional status filter (e.g. `success` or
   * `error`).
   *
   * Returns: A list of usage entries sorted by timestamp, each represented as a
   * dictionary.
   */
  listRecent(
    query: UsageListRecentParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<UsageListRecentResponse> {
    return this._client.get('/usage/recent', { query, ...options });
  }

  /**
   * Get usage statistics for the authenticated user.
   *
   * Args: auth: Authentication context identifying the caller.
   *
   * Returns: A mapping of operation types to token usage counts.
   */
  retrieveStats(options?: RequestOptions): APIPromise<UsageRetrieveStatsResponse> {
    return this._client.get('/usage/stats', options);
  }
}

export type UsageListRecentResponse = Array<{ [key: string]: unknown }>;

export type UsageRetrieveStatsResponse = { [key: string]: number };

export interface UsageListRecentParams {
  operation_type?: string | null;

  since?: string | null;

  status?: string | null;
}

export declare namespace Usage {
  export {
    type UsageListRecentResponse as UsageListRecentResponse,
    type UsageRetrieveStatsResponse as UsageRetrieveStatsResponse,
    type UsageListRecentParams as UsageListRecentParams,
  };
}
