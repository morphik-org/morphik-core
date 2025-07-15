// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import * as IngestAPI from './ingest';
import { APIPromise } from '../core/api-promise';
import { type Uploadable } from '../core/uploads';
import { RequestOptions } from '../internal/request-options';
import { multipartFormRequestOptions } from '../internal/uploads';
import { path } from '../internal/utils/path';

export class Documents extends APIResource {
  /**
   * Retrieve a single document by its external identifier.
   *
   * Args: document_id: External ID of the document to fetch. auth: Authentication
   * context used to verify access rights.
   *
   * Returns: The :class:`Document` metadata if found.
   */
  retrieve(documentID: string, options?: RequestOptions): APIPromise<IngestAPI.Document> {
    return this._client.get(path`/documents/${documentID}`, options);
  }

  /**
   * List accessible documents.
   *
   * Args: auth: Authentication context skip: Number of documents to skip limit:
   * Maximum number of documents to return filters: Optional metadata filters
   * folder_name: Optional folder to scope the operation to end_user_id: Optional
   * end-user ID to scope the operation to
   *
   * Returns: List[Document]: List of accessible documents
   */
  list(
    params: DocumentListParams | null | undefined = undefined,
    options?: RequestOptions,
  ): APIPromise<DocumentListResponse> {
    const { end_user_id, folder_name, limit, skip, body } = params ?? {};
    return this._client.post('/documents', {
      query: { end_user_id, folder_name, limit, skip },
      body: body,
      ...options,
    });
  }

  /**
   * Delete a document and all associated data.
   *
   * This endpoint deletes a document and all its associated data, including:
   *
   * - Document metadata
   * - Document content in storage
   * - Document chunks and embeddings in vector store
   *
   * Args: document_id: ID of the document to delete auth: Authentication context
   * (must have write access to the document)
   *
   * Returns: Deletion status
   */
  delete(documentID: string, options?: RequestOptions): APIPromise<unknown> {
    return this._client.delete(path`/documents/${documentID}`, options);
  }

  /**
   * Get the processing status of a document.
   *
   * Args: document_id: ID of the document to check auth: Authentication context
   *
   * Returns: Dict containing status information for the document
   */
  getStatus(documentID: string, options?: RequestOptions): APIPromise<DocumentGetStatusResponse> {
    return this._client.get(path`/documents/${documentID}/status`, options);
  }

  /**
   * Get document by filename.
   *
   * Args: filename: Filename of the document to retrieve auth: Authentication
   * context folder_name: Optional folder to scope the operation to end_user_id:
   * Optional end-user ID to scope the operation to
   *
   * Returns: Document: Document metadata if found and accessible
   */
  retrieveByFilename(
    filename: string,
    query: DocumentRetrieveByFilenameParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<IngestAPI.Document> {
    return this._client.get(path`/documents/filename/${filename}`, { query, ...options });
  }

  /**
   * Update a document with content from a file using the specified strategy.
   *
   * Args: document_id: ID of the document to update file: File to add to the
   * document metadata: JSON string of metadata to merge with existing metadata
   * rules: JSON string of rules to apply to the content update_strategy: Strategy
   * for updating the document (default: 'add') use_colpali: Whether to use
   * multi-vector embedding auth: Authentication context
   *
   * Returns: Document: Updated document metadata
   */
  updateFile(
    documentID: string,
    body: DocumentUpdateFileParams,
    options?: RequestOptions,
  ): APIPromise<IngestAPI.Document> {
    return this._client.post(
      path`/documents/${documentID}/update_file`,
      multipartFormRequestOptions({ body, ...options }, this._client),
    );
  }

  /**
   * Update only a document's metadata.
   *
   * Args: document_id: ID of the document to update metadata: New metadata to merge
   * with existing metadata auth: Authentication context
   *
   * Returns: Document: Updated document metadata
   */
  updateMetadata(
    documentID: string,
    params: DocumentUpdateMetadataParams,
    options?: RequestOptions,
  ): APIPromise<IngestAPI.Document> {
    const { body } = params;
    return this._client.post(path`/documents/${documentID}/update_metadata`, { body: body, ...options });
  }

  /**
   * Update a document with new text content using the specified strategy.
   *
   * Args: document_id: ID of the document to update request: Text content and
   * metadata for the update update_strategy: Strategy for updating the document
   * (default: 'add') auth: Authentication context
   *
   * Returns: Document: Updated document metadata
   */
  updateText(
    documentID: string,
    params: DocumentUpdateTextParams,
    options?: RequestOptions,
  ): APIPromise<IngestAPI.Document> {
    const { update_strategy, ...body } = params;
    return this._client.post(path`/documents/${documentID}/update_text`, {
      query: { update_strategy },
      body,
      ...options,
    });
  }
}

export type DocumentListResponse = Array<IngestAPI.Document>;

export type DocumentDeleteResponse = unknown;

export type DocumentGetStatusResponse = { [key: string]: unknown };

export interface DocumentListParams {
  /**
   * Query param:
   */
  end_user_id?: string | null;

  /**
   * Query param:
   */
  folder_name?: string | Array<string> | null;

  /**
   * Query param:
   */
  limit?: number;

  /**
   * Query param:
   */
  skip?: number;

  /**
   * Body param:
   */
  body?: { [key: string]: unknown } | null;
}

export interface DocumentRetrieveByFilenameParams {
  end_user_id?: string | null;
}

export interface DocumentUpdateFileParams {
  file: Uploadable;

  metadata?: string;

  rules?: string;

  update_strategy?: string;

  use_colpali?: boolean | null;
}

export interface DocumentUpdateMetadataParams {
  body: { [key: string]: unknown };
}

export interface DocumentUpdateTextParams {
  /**
   * Body param:
   */
  content: string;

  /**
   * Query param:
   */
  update_strategy?: string;

  /**
   * Body param: Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  /**
   * Body param:
   */
  filename?: string | null;

  /**
   * Body param: Optional folder scope for the operation
   */
  folder_name?: string | null;

  /**
   * Body param:
   */
  metadata?: { [key: string]: unknown };

  /**
   * Body param:
   */
  rules?: Array<{ [key: string]: unknown }>;

  /**
   * Body param:
   */
  use_colpali?: boolean | null;
}

export declare namespace Documents {
  export {
    type DocumentListResponse as DocumentListResponse,
    type DocumentDeleteResponse as DocumentDeleteResponse,
    type DocumentGetStatusResponse as DocumentGetStatusResponse,
    type DocumentListParams as DocumentListParams,
    type DocumentRetrieveByFilenameParams as DocumentRetrieveByFilenameParams,
    type DocumentUpdateFileParams as DocumentUpdateFileParams,
    type DocumentUpdateMetadataParams as DocumentUpdateMetadataParams,
    type DocumentUpdateTextParams as DocumentUpdateTextParams,
  };
}
