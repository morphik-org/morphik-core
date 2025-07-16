// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as IngestAPI from '../ingest';
import * as ChatAPI from './chat';
import { Chat, ChatCompleteChatParams, ChatCompleteChatResponse, ChatRetrieveHistoryResponse } from './chat';
import { APIPromise } from '../../core/api-promise';
import { type Uploadable } from '../../core/uploads';
import { RequestOptions } from '../../internal/request-options';
import { multipartFormRequestOptions } from '../../internal/uploads';
import { path } from '../../internal/utils/path';

export class Documents extends APIResource {
  chat: ChatAPI.Chat = new ChatAPI.Chat(this._client);

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
   * Args: request: Request body containing filters and pagination auth:
   * Authentication context folder_name: Optional folder to scope the operation to
   * end_user_id: Optional end-user ID to scope the operation to
   *
   * Returns: List[Document]: List of accessible documents
   */
  list(params: DocumentListParams, options?: RequestOptions): APIPromise<DocumentListResponse> {
    const { end_user_id, folder_name, ...body } = params;
    return this._client.post('/documents/', { query: { end_user_id, folder_name }, body, ...options });
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
  delete(documentID: string, options?: RequestOptions): APIPromise<DocumentDeleteResponse> {
    return this._client.delete(path`/documents/${documentID}`, options);
  }

  /**
   * Download the actual file content for a document. This endpoint is used for local
   * storage when file:// URLs cannot be accessed by browsers.
   *
   * Args: document_id: External ID of the document auth: Authentication context
   *
   * Returns: StreamingResponse with the file content
   */
  downloadFile(documentID: string, options?: RequestOptions): APIPromise<unknown> {
    return this._client.get(path`/documents/${documentID}/file`, options);
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
  getByFilename(
    filename: string,
    query: DocumentGetByFilenameParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<IngestAPI.Document> {
    return this._client.get(path`/documents/filename/${filename}`, { query, ...options });
  }

  /**
   * Get a download URL for a specific document.
   *
   * Args: document_id: External ID of the document auth: Authentication context
   * expires_in: URL expiration time in seconds (default: 1 hour)
   *
   * Returns: Dictionary containing the download URL and metadata
   */
  getDownloadURL(
    documentID: string,
    query: DocumentGetDownloadURLParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<DocumentGetDownloadURLResponse> {
    return this._client.get(path`/documents/${documentID}/download_url`, { query, ...options });
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
   * Args: document_id: ID of the document to update metadata_updates: New metadata
   * to merge with existing metadata auth: Authentication context
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

/**
 * Response for document deletion endpoint
 */
export interface DocumentDeleteResponse {
  message: string;

  status: string;
}

export type DocumentListResponse = Array<IngestAPI.Document>;

export type DocumentDownloadFileResponse = unknown;

/**
 * Response for document download URL endpoint
 */
export interface DocumentGetDownloadURLResponse {
  download_url: string;

  expires_in: number;
}

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
   * Body param: Metadata filters for documents
   */
  document_filters?: { [key: string]: unknown } | null;

  /**
   * Body param:
   */
  limit?: number;

  /**
   * Body param:
   */
  skip?: number;
}

export interface DocumentGetByFilenameParams {
  end_user_id?: string | null;

  folder_name?: string | Array<string> | null;
}

export interface DocumentGetDownloadURLParams {
  /**
   * URL expiration time in seconds
   */
  expires_in?: number;
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

Documents.Chat = Chat;

export declare namespace Documents {
  export {
    type DocumentDeleteResponse as DocumentDeleteResponse,
    type DocumentListResponse as DocumentListResponse,
    type DocumentDownloadFileResponse as DocumentDownloadFileResponse,
    type DocumentGetDownloadURLResponse as DocumentGetDownloadURLResponse,
    type DocumentGetStatusResponse as DocumentGetStatusResponse,
    type DocumentListParams as DocumentListParams,
    type DocumentGetByFilenameParams as DocumentGetByFilenameParams,
    type DocumentGetDownloadURLParams as DocumentGetDownloadURLParams,
    type DocumentUpdateFileParams as DocumentUpdateFileParams,
    type DocumentUpdateMetadataParams as DocumentUpdateMetadataParams,
    type DocumentUpdateTextParams as DocumentUpdateTextParams,
  };

  export {
    Chat as Chat,
    type ChatCompleteChatResponse as ChatCompleteChatResponse,
    type ChatRetrieveHistoryResponse as ChatRetrieveHistoryResponse,
    type ChatCompleteChatParams as ChatCompleteChatParams,
  };
}
