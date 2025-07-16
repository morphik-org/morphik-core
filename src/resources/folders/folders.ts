// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as DocumentsAPI from './documents';
import { DocumentAddParams, DocumentAddResponse, DocumentRemoveParams, Documents } from './documents';
import * as WorkflowsAPI from './workflows';
import {
  WorkflowAssociateParams,
  WorkflowAssociateResponse,
  WorkflowDisassociateParams,
  WorkflowDisassociateResponse,
  WorkflowListResponse,
  Workflows,
} from './workflows';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Folders extends APIResource {
  documents: DocumentsAPI.Documents = new DocumentsAPI.Documents(this._client);
  workflows: WorkflowsAPI.Workflows = new WorkflowsAPI.Workflows(this._client);

  /**
   * Create a new folder.
   *
   * Args: folder_create: Folder creation request containing name and optional
   * description auth: Authentication context
   *
   * Returns: Folder: Created folder
   */
  create(body: FolderCreateParams, options?: RequestOptions): APIPromise<Folder> {
    return this._client.post('/folders', { body, ...options });
  }

  /**
   * Get a folder by ID.
   *
   * Args: folder_id: ID of the folder auth: Authentication context
   *
   * Returns: Folder: Folder if found and accessible
   */
  retrieve(folderID: string, options?: RequestOptions): APIPromise<Folder> {
    return this._client.get(path`/folders/${folderID}`, options);
  }

  /**
   * List all folders the user has access to.
   *
   * Args: auth: Authentication context
   *
   * Returns: List[Folder]: List of folders
   */
  list(options?: RequestOptions): APIPromise<FolderListResponse> {
    return this._client.get('/folders', options);
  }

  /**
   * Delete a folder and all associated documents.
   *
   * Args: folder_name: Name of the folder to delete auth: Authentication context
   * (must have write access to the folder)
   *
   * Returns: Deletion status
   */
  delete(folderName: string, options?: RequestOptions): APIPromise<FolderDeleteResponse> {
    return this._client.delete(path`/folders/${folderName}`, options);
  }

  /**
   * Return compact folder list (id, name, doc_count, updated_at).
   */
  listSummaries(options?: RequestOptions): APIPromise<FolderListSummariesResponse> {
    return this._client.get('/folders/summary', options);
  }

  /**
   * Set extraction rules for a folder.
   *
   * Args: folder_id: ID of the folder to set rules for request: SetFolderRuleRequest
   * containing metadata extraction rules auth: Authentication context
   * apply_to_existing: Whether to apply rules to existing documents in the folder
   *
   * Returns: Success status with processing results
   */
  setRule(
    folderID: string,
    params: FolderSetRuleParams,
    options?: RequestOptions,
  ): APIPromise<FolderSetRuleResponse> {
    const { apply_to_existing, ...body } = params;
    return this._client.post(path`/folders/${folderID}/set_rule`, {
      query: { apply_to_existing },
      body,
      ...options,
    });
  }
}

/**
 * Represents a folder that contains documents
 */
export interface Folder {
  name: string;

  id?: string;

  app_id?: string | null;

  description?: string | null;

  document_ids?: Array<string> | null;

  end_user_id?: string | null;

  rules?: Array<{ [key: string]: unknown }>;

  system_metadata?: { [key: string]: unknown };

  /**
   * List of workflow IDs to run on document ingestion
   */
  workflow_ids?: Array<string>;
}

export type FolderListResponse = Array<Folder>;

/**
 * Response for folder deletion endpoint
 */
export interface FolderDeleteResponse {
  message: string;

  status: string;
}

export type FolderListSummariesResponse = Array<FolderListSummariesResponse.FolderListSummariesResponseItem>;

export namespace FolderListSummariesResponse {
  export interface FolderListSummariesResponseItem {
    id: string;

    name: string;

    description?: string | null;

    doc_count?: number;

    updated_at?: string | null;
  }
}

/**
 * Response for folder rule setting endpoint
 */
export interface FolderSetRuleResponse {
  message: string;

  status: string;
}

export interface FolderCreateParams {
  name: string;

  description?: string | null;
}

export interface FolderSetRuleParams {
  /**
   * Body param:
   */
  rules: Array<FolderSetRuleParams.Rule>;

  /**
   * Query param:
   */
  apply_to_existing?: boolean;
}

export namespace FolderSetRuleParams {
  /**
   * Request model for metadata extraction rule
   */
  export interface Rule {
    schema?: { [key: string]: unknown };

    type?: string;
  }
}

Folders.Documents = Documents;
Folders.Workflows = Workflows;

export declare namespace Folders {
  export {
    type Folder as Folder,
    type FolderListResponse as FolderListResponse,
    type FolderDeleteResponse as FolderDeleteResponse,
    type FolderListSummariesResponse as FolderListSummariesResponse,
    type FolderSetRuleResponse as FolderSetRuleResponse,
    type FolderCreateParams as FolderCreateParams,
    type FolderSetRuleParams as FolderSetRuleParams,
  };

  export {
    Documents as Documents,
    type DocumentAddResponse as DocumentAddResponse,
    type DocumentAddParams as DocumentAddParams,
    type DocumentRemoveParams as DocumentRemoveParams,
  };

  export {
    Workflows as Workflows,
    type WorkflowListResponse as WorkflowListResponse,
    type WorkflowAssociateResponse as WorkflowAssociateResponse,
    type WorkflowDisassociateResponse as WorkflowDisassociateResponse,
    type WorkflowAssociateParams as WorkflowAssociateParams,
    type WorkflowDisassociateParams as WorkflowDisassociateParams,
  };
}
