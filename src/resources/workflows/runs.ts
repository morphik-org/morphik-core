// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as WorkflowsAPI from './workflows';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Runs extends APIResource {
  /**
   * Get Workflow Run
   */
  retrieve(runID: string, options?: RequestOptions): APIPromise<WorkflowsAPI.WorkflowRun> {
    return this._client.get(path`/workflows/runs/${runID}`, options);
  }

  /**
   * Get all runs for a specific workflow.
   */
  list(workflowID: string, options?: RequestOptions): APIPromise<RunListResponse> {
    return this._client.get(path`/workflows/${workflowID}/runs`, options);
  }

  /**
   * Delete a workflow run.
   */
  delete(runID: string, options?: RequestOptions): APIPromise<RunDeleteResponse> {
    return this._client.delete(path`/workflows/runs/${runID}`, options);
  }
}

export type RunListResponse = Array<WorkflowsAPI.WorkflowRun>;

export type RunDeleteResponse = { [key: string]: string };

export declare namespace Runs {
  export { type RunListResponse as RunListResponse, type RunDeleteResponse as RunDeleteResponse };
}
