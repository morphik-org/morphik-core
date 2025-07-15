// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Documents extends APIResource {
  /**
   * Add a document to a folder.
   *
   * Args: folder_id: ID of the folder document_id: ID of the document auth:
   * Authentication context
   *
   * Returns: Success status
   */
  add(documentID: string, params: DocumentAddParams, options?: RequestOptions): APIPromise<unknown> {
    const { folder_id } = params;
    return this._client.post(path`/folders/${folder_id}/documents/${documentID}`, options);
  }

  /**
   * Remove a document from a folder.
   *
   * Args: folder_id: ID of the folder document_id: ID of the document auth:
   * Authentication context
   *
   * Returns: Success status
   */
  remove(documentID: string, params: DocumentRemoveParams, options?: RequestOptions): APIPromise<unknown> {
    const { folder_id } = params;
    return this._client.delete(path`/folders/${folder_id}/documents/${documentID}`, options);
  }
}

export type DocumentAddResponse = unknown;

export type DocumentRemoveResponse = unknown;

export interface DocumentAddParams {
  folder_id: string;
}

export interface DocumentRemoveParams {
  folder_id: string;
}

export declare namespace Documents {
  export {
    type DocumentAddResponse as DocumentAddResponse,
    type DocumentRemoveResponse as DocumentRemoveResponse,
    type DocumentAddParams as DocumentAddParams,
    type DocumentRemoveParams as DocumentRemoveParams,
  };
}
