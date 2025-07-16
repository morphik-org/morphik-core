// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as WorkflowsWorkflowsAPI from '../workflows/workflows';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Workflows extends APIResource {
  /**
   * List all workflows associated with a folder.
   */
  list(folderID: string, options?: RequestOptions): APIPromise<WorkflowListResponse> {
    return this._client.get(path`/folders/${folderID}/workflows`, options);
  }

  /**
   * Associate a workflow with a folder for automatic execution on document
   * ingestion.
   */
  associate(
    workflowID: string,
    params: WorkflowAssociateParams,
    options?: RequestOptions,
  ): APIPromise<WorkflowAssociateResponse> {
    const { folder_id } = params;
    return this._client.post(path`/folders/${folder_id}/workflows/${workflowID}`, options);
  }

  /**
   * Remove a workflow association from a folder.
   */
  disassociate(
    workflowID: string,
    params: WorkflowDisassociateParams,
    options?: RequestOptions,
  ): APIPromise<WorkflowDisassociateResponse> {
    const { folder_id } = params;
    return this._client.delete(path`/folders/${folder_id}/workflows/${workflowID}`, options);
  }
}

export type WorkflowListResponse = Array<WorkflowsWorkflowsAPI.Workflow>;

export type WorkflowAssociateResponse = { [key: string]: boolean | string };

export type WorkflowDisassociateResponse = { [key: string]: boolean | string };

export interface WorkflowAssociateParams {
  folder_id: string;
}

export interface WorkflowDisassociateParams {
  folder_id: string;
}

export declare namespace Workflows {
  export {
    type WorkflowListResponse as WorkflowListResponse,
    type WorkflowAssociateResponse as WorkflowAssociateResponse,
    type WorkflowDisassociateResponse as WorkflowDisassociateResponse,
    type WorkflowAssociateParams as WorkflowAssociateParams,
    type WorkflowDisassociateParams as WorkflowDisassociateParams,
  };
}
