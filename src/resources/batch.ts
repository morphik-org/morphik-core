// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import * as IngestAPI from './ingest';
import * as RetrieveAPI from './retrieve';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';

export class Batch extends APIResource {
  /**
   * Retrieve specific chunks by their document ID and chunk number in a single batch
   * operation.
   *
   * Args: request: Dictionary containing: - sources: List of ChunkSource objects
   * (with document_id and chunk_number) - folder_name: Optional folder to scope the
   * operation to - end_user_id: Optional end-user ID to scope the operation to -
   * use_colpali: Whether to use ColPali-style embedding auth: Authentication context
   *
   * Returns: List[ChunkResult]: List of chunk results
   */
  retrieveChunks(
    params: BatchRetrieveChunksParams,
    options?: RequestOptions,
  ): APIPromise<BatchRetrieveChunksResponse> {
    const { body } = params;
    return this._client.post('/batch/chunks', { body: body, ...options });
  }

  /**
   * Retrieve multiple documents by their IDs in a single batch operation.
   *
   * Args: request: Dictionary containing: - document_ids: List of document IDs to
   * retrieve - folder_name: Optional folder to scope the operation to - end_user_id:
   * Optional end-user ID to scope the operation to auth: Authentication context
   *
   * Returns: List[Document]: List of documents matching the IDs
   */
  retrieveDocuments(
    params: BatchRetrieveDocumentsParams,
    options?: RequestOptions,
  ): APIPromise<BatchRetrieveDocumentsResponse> {
    const { body } = params;
    return this._client.post('/batch/documents', { body: body, ...options });
  }
}

export type BatchRetrieveChunksResponse = Array<RetrieveAPI.ChunkResult>;

export type BatchRetrieveDocumentsResponse = Array<IngestAPI.Document>;

export interface BatchRetrieveChunksParams {
  body: { [key: string]: unknown };
}

export interface BatchRetrieveDocumentsParams {
  body: { [key: string]: unknown };
}

export declare namespace Batch {
  export {
    type BatchRetrieveChunksResponse as BatchRetrieveChunksResponse,
    type BatchRetrieveDocumentsResponse as BatchRetrieveDocumentsResponse,
    type BatchRetrieveChunksParams as BatchRetrieveChunksParams,
    type BatchRetrieveDocumentsParams as BatchRetrieveDocumentsParams,
  };
}
