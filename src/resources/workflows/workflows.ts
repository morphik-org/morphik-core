// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as RunsAPI from './runs';
import { RunDeleteResponse, RunListResponse, Runs } from './runs';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Workflows extends APIResource {
  runs: RunsAPI.Runs = new RunsAPI.Runs(this._client);

  /**
   * Create a new workflow.
   */
  create(body: WorkflowCreateParams, options?: RequestOptions): APIPromise<Workflow> {
    return this._client.post('/workflows', { body, ...options });
  }

  /**
   * Get Workflow
   */
  retrieve(workflowID: string, options?: RequestOptions): APIPromise<Workflow> {
    return this._client.get(path`/workflows/${workflowID}`, options);
  }

  /**
   * Update Workflow
   */
  update(workflowID: string, params: WorkflowUpdateParams, options?: RequestOptions): APIPromise<Workflow> {
    const { body } = params;
    return this._client.put(path`/workflows/${workflowID}`, { body: body, ...options });
  }

  /**
   * List workflows visible to the caller.
   */
  list(options?: RequestOptions): APIPromise<WorkflowListResponse> {
    return this._client.get('/workflows', options);
  }

  /**
   * Delete Workflow
   */
  delete(workflowID: string, options?: RequestOptions): APIPromise<WorkflowDeleteResponse> {
    return this._client.delete(path`/workflows/${workflowID}`, options);
  }

  /**
   * Queue a workflow run and return immediately.
   *
   * The heavy execution is scheduled in a _BackgroundTask_ so the request finishes
   * quickly, enabling the UI to update the document row to "processing" without
   * waiting for LLM calls.
   */
  run(documentID: string, params: WorkflowRunParams, options?: RequestOptions): APIPromise<WorkflowRun> {
    const { workflow_id } = params;
    return this._client.post(path`/workflows/${workflow_id}/run/${documentID}`, options);
  }
}

/**
 * High-level definition of a multi-step document processing workflow.
 */
export interface Workflow {
  name: string;

  /**
   * Organization / developer ID owning this workflow
   */
  owner_id: string;

  /**
   * Ordered list of actions to execute
   */
  steps: Array<Workflow.Step>;

  id?: string;

  /**
   * App ID when created inside an application context
   */
  app_id?: string | null;

  description?: string | null;

  system_metadata?: { [key: string]: unknown };

  /**
   * End-user ID when created in a narrowed scope
   */
  user_id?: string | null;
}

export namespace Workflow {
  /**
   * A single step inside a _Workflow_ configuration.
   *
   * The concrete implementation is looked-up at runtime via the _action_id_ in the
   * Action Registry (see _core.workflows.registry_).
   */
  export interface Step {
    /**
     * Registry identifier of the action to execute
     */
    action_id: string;

    /**
     * JSON-serialisable parameters for the action
     */
    parameters?: { [key: string]: unknown };
  }
}

/**
 * Represents a concrete execution of a _Workflow_ against a document.
 */
export interface WorkflowRun {
  document_id: string;

  workflow_id: string;

  id?: string;

  app_id?: string | null;

  completed_at?: string | null;

  error?: string | null;

  final_output?: { [key: string]: unknown } | null;

  results_per_step?: Array<{ [key: string]: unknown }>;

  started_at?: string | null;

  /**
   * Allowed status values for _WorkflowRun_ objects.
   */
  status?: 'queued' | 'running' | 'completed' | 'failed';

  system_metadata?: { [key: string]: unknown };

  user_id?: string | null;
}

export type WorkflowListResponse = Array<Workflow>;

export type WorkflowDeleteResponse = { [key: string]: string };

export interface WorkflowCreateParams {
  name: string;

  /**
   * Organization / developer ID owning this workflow
   */
  owner_id: string;

  /**
   * Ordered list of actions to execute
   */
  steps: Array<WorkflowCreateParams.Step>;

  id?: string;

  /**
   * App ID when created inside an application context
   */
  app_id?: string | null;

  description?: string | null;

  system_metadata?: { [key: string]: unknown };

  /**
   * End-user ID when created in a narrowed scope
   */
  user_id?: string | null;
}

export namespace WorkflowCreateParams {
  /**
   * A single step inside a _Workflow_ configuration.
   *
   * The concrete implementation is looked-up at runtime via the _action_id_ in the
   * Action Registry (see _core.workflows.registry_).
   */
  export interface Step {
    /**
     * Registry identifier of the action to execute
     */
    action_id: string;

    /**
     * JSON-serialisable parameters for the action
     */
    parameters?: { [key: string]: unknown };
  }
}

export interface WorkflowUpdateParams {
  body: { [key: string]: unknown };
}

export interface WorkflowRunParams {
  workflow_id: string;
}

Workflows.Runs = Runs;

export declare namespace Workflows {
  export {
    type Workflow as Workflow,
    type WorkflowRun as WorkflowRun,
    type WorkflowListResponse as WorkflowListResponse,
    type WorkflowDeleteResponse as WorkflowDeleteResponse,
    type WorkflowCreateParams as WorkflowCreateParams,
    type WorkflowUpdateParams as WorkflowUpdateParams,
    type WorkflowRunParams as WorkflowRunParams,
  };

  export {
    Runs as Runs,
    type RunListResponse as RunListResponse,
    type RunDeleteResponse as RunDeleteResponse,
  };
}
