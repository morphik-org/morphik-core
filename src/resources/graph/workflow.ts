// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Workflow extends APIResource {
  /**
   * Check the status of a graph build/update workflow.
   *
   * This endpoint polls the external graph API to check the status of an async
   * operation.
   *
   * Args: workflow_id: The workflow ID returned from build/update operations run_id:
   * Optional run ID for the specific workflow run auth: Authentication context
   *
   * Returns: Dict containing status ('running', 'completed', or 'failed') and
   * optional result
   *
   * @example
   * ```ts
   * const response = await client.graph.workflow.status(
   *   'workflow_id',
   * );
   * ```
   */
  status(
    workflowID: string,
    query: WorkflowStatusParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<WorkflowStatusResponse> {
    return this._client.get(path`/graph/workflow/${workflowID}/status`, { query, ...options });
  }
}

export type WorkflowStatusResponse = { [key: string]: unknown };

export interface WorkflowStatusParams {
  run_id?: string | null;
}

export declare namespace Workflow {
  export {
    type WorkflowStatusResponse as WorkflowStatusResponse,
    type WorkflowStatusParams as WorkflowStatusParams,
  };
}
