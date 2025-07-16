// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { type Uploadable } from '../core/uploads';
import { RequestOptions } from '../internal/request-options';
import { multipartFormRequestOptions } from '../internal/uploads';

export class Ingest extends APIResource {
  /**
   * Batch ingest **multiple files** (async).
   *
   * Each file is treated the same as :func:`ingest_file` but sharing the same
   * request avoids many round-trips. All heavy work is still delegated to the
   * background worker pool.
   *
   * Args: files: List of files to upload. metadata: Either a single JSON-string dict
   * or list of dicts matching the number of files. rules: Either a single rules list
   * or list-of-lists per file. use_colpali: Enable multi-vector embeddings.
   * folder_name: Optional folder scoping for **all** files. end_user_id: Optional
   * end-user scoping for **all** files. auth: Caller context with _write_
   * permission. redis: arq redis connection to enqueue jobs.
   *
   * Returns: BatchIngestResponse summarising created documents & errors.
   */
  batchIngestFiles(
    body: IngestBatchIngestFilesParams,
    options?: RequestOptions,
  ): APIPromise<IngestBatchIngestFilesResponse> {
    return this._client.post(
      '/ingest/files',
      multipartFormRequestOptions({ body, ...options }, this._client),
    );
  }

  /**
   * Ingest a **file** asynchronously.
   *
   * The file is uploaded to object storage, a _Document_ stub is persisted with
   * `status='processing'` and a background worker picks up the heavy parsing /
   * chunking work.
   *
   * Args: file: Uploaded file from multipart/form-data. metadata: JSON-string
   * representing user metadata. rules: JSON-string with extraction / NL rules list.
   * auth: Caller context – must include _write_ permission. use_colpali: Switch to
   * multi-vector embeddings. folder_name: Optionally scope doc to a folder.
   * end_user_id: Optionally scope doc to an end-user. redis: arq redis connection –
   * used to enqueue the job.
   *
   * Returns: Document stub with `status='processing'`.
   */
  ingestFile(body: IngestIngestFileParams, options?: RequestOptions): APIPromise<Document> {
    return this._client.post('/ingest/file', multipartFormRequestOptions({ body, ...options }, this._client));
  }

  /**
   * Ingest a **text** document.
   *
   * Args: request: IngestTextRequest payload containing: • content – raw text to
   * ingest. • filename – optional filename to help detect MIME-type. • metadata –
   * optional JSON metadata dict. • rules – optional list of extraction / NL rules. •
   * folder_name – optional folder scope. • end_user_id – optional end-user scope.
   * auth: Decoded JWT context (injected).
   *
   * Returns: Document metadata row representing the newly-ingested text.
   */
  ingestText(body: IngestIngestTextParams, options?: RequestOptions): APIPromise<Document> {
    return this._client.post('/ingest/text', { body, ...options });
  }
}

/**
 * Represents a document stored in the database documents collection
 */
export interface Document {
  content_type: string;

  additional_metadata?: { [key: string]: unknown };

  app_id?: string | null;

  chunk_ids?: Array<string>;

  end_user_id?: string | null;

  external_id?: string;

  filename?: string | null;

  folder_name?: string | null;

  metadata?: { [key: string]: unknown };

  storage_files?: Array<Document.StorageFile>;

  storage_info?: { [key: string]: unknown };

  system_metadata?: { [key: string]: unknown };
}

export namespace Document {
  /**
   * Information about a file stored in storage
   */
  export interface StorageFile {
    bucket: string;

    key: string;

    content_type?: string | null;

    filename?: string | null;

    timestamp?: string;

    version?: number;
  }
}

/**
 * Request model for ingesting text content
 */
export interface TextRequest {
  content: string;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  filename?: string | null;

  /**
   * Optional folder scope for the operation
   */
  folder_name?: string | null;

  metadata?: { [key: string]: unknown };

  rules?: Array<{ [key: string]: unknown }>;

  use_colpali?: boolean | null;
}

/**
 * Response model for batch ingestion
 */
export interface IngestBatchIngestFilesResponse {
  documents: Array<Document>;

  errors: Array<{ [key: string]: string }>;
}

export interface IngestBatchIngestFilesParams {
  files: Array<Uploadable>;

  end_user_id?: string | null;

  folder_name?: string | null;

  metadata?: string;

  rules?: string;

  use_colpali?: boolean | null;
}

export interface IngestIngestFileParams {
  file: Uploadable;

  end_user_id?: string | null;

  folder_name?: string | null;

  metadata?: string;

  rules?: string;

  use_colpali?: boolean | null;
}

export interface IngestIngestTextParams {
  content: string;

  /**
   * Optional end-user scope for the operation
   */
  end_user_id?: string | null;

  filename?: string | null;

  /**
   * Optional folder scope for the operation
   */
  folder_name?: string | null;

  metadata?: { [key: string]: unknown };

  rules?: Array<{ [key: string]: unknown }>;

  use_colpali?: boolean | null;
}

export declare namespace Ingest {
  export {
    type Document as Document,
    type TextRequest as TextRequest,
    type IngestBatchIngestFilesResponse as IngestBatchIngestFilesResponse,
    type IngestBatchIngestFilesParams as IngestBatchIngestFilesParams,
    type IngestIngestFileParams as IngestIngestFileParams,
    type IngestIngestTextParams as IngestIngestTextParams,
  };
}
