// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Logs extends APIResource {
  /**
   * Return recent logs for the authenticated user (scoped by user_id).
   */
  list(query: LogListParams | null | undefined = {}, options?: RequestOptions): APIPromise<LogListResponse> {
    return this._client.get('/logs/', { query, ...options });
  }
}

export type LogListResponse = Array<LogListResponse.LogListResponseItem>;

export namespace LogListResponse {
  /**
   * Public serialisable view of a UsageRecord.
   */
  export interface LogListResponseItem {
    duration_ms: number;

    operation_type: string;

    status: string;

    timestamp: string;

    tokens_used: number;

    user_id: string;

    app_id?: string | null;

    error?: string | null;

    metadata?: { [key: string]: unknown } | null;
  }
}

export interface LogListParams {
  limit?: number;

  op_type?: string | null;

  since?: string | null;

  status?: string | null;
}

export declare namespace Logs {
  export { type LogListResponse as LogListResponse, type LogListParams as LogListParams };
}
