// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../../core/resource';
import * as AuthAPI from './auth';
import {
  Auth,
  AuthFinalizeManualAuthParams,
  AuthFinalizeManualAuthResponse,
  AuthGetInitiateAuthURLParams,
  AuthGetInitiateAuthURLResponse,
} from './auth';
import { APIPromise } from '../../../core/api-promise';
import { RequestOptions } from '../../../internal/request-options';
import { path } from '../../../internal/utils/path';

export class Connectors extends APIResource {
  auth: AuthAPI.Auth = new AuthAPI.Auth(this._client);

  /**
   * Disconnects the user from the specified connector by removing stored
   * credentials.
   */
  disconnect(connectorType: string, options?: RequestOptions): APIPromise<ConnectorDisconnectResponse> {
    return this._client.post(path`/ee/connectors/${connectorType}/disconnect`, options);
  }

  /**
   * Checks the current authentication status for the given connector type.
   */
  getAuthStatus(connectorType: string, options?: RequestOptions): APIPromise<ConnectorGetAuthStatusResponse> {
    return this._client.get(path`/ee/connectors/${connectorType}/auth_status`, options);
  }

  /**
   * Handles the OAuth 2.0 callback from the authentication provider. Validates
   * state, finalizes authentication, and stores credentials.
   */
  handleOAuthCallback(
    connectorType: string,
    query: ConnectorHandleOAuthCallbackParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<unknown> {
    return this._client.get(path`/ee/connectors/${connectorType}/oauth2callback`, { query, ...options });
  }

  /**
   * Downloads a file from the connector and ingests it into Morphik via
   * DocumentService.
   */
  ingestFile(
    connectorType: string,
    body: ConnectorIngestFileParams,
    options?: RequestOptions,
  ): APIPromise<ConnectorIngestFileResponse> {
    return this._client.post(path`/ee/connectors/${connectorType}/ingest`, { body, ...options });
  }

  /**
   * Lists files and folders from the specified connector.
   */
  listFiles(
    connectorType: string,
    query: ConnectorListFilesParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<ConnectorListFilesResponse> {
    return this._client.get(path`/ee/connectors/${connectorType}/files`, { query, ...options });
  }
}

export type ConnectorDisconnectResponse = { [key: string]: unknown };

export interface ConnectorGetAuthStatusResponse {
  is_authenticated: boolean;

  auth_url?: string | null;

  message?: string | null;
}

export type ConnectorHandleOAuthCallbackResponse = unknown;

export type ConnectorIngestFileResponse = { [key: string]: unknown };

export interface ConnectorListFilesResponse {
  files: Array<ConnectorListFilesResponse.File>;

  next_page_token?: string | null;
}

export namespace ConnectorListFilesResponse {
  export interface File {
    id: string;

    name: string;

    is_folder?: boolean;

    mime_type?: string | null;

    modified_date?: string | null;

    size?: number | null;
  }
}

export interface ConnectorHandleOAuthCallbackParams {
  code?: string | null;

  error?: string | null;

  error_description?: string | null;

  state?: string | null;
}

export interface ConnectorIngestFileParams {
  file_id: string;

  metadata?: { [key: string]: unknown } | null;

  morphik_end_user_id?: string | null;

  morphik_folder_name?: string | null;

  rules?: Array<{ [key: string]: unknown }> | null;
}

export interface ConnectorListFilesParams {
  page_size?: number;

  page_token?: string | null;

  path?: string | null;

  q_filter?: string | null;
}

Connectors.Auth = Auth;

export declare namespace Connectors {
  export {
    type ConnectorDisconnectResponse as ConnectorDisconnectResponse,
    type ConnectorGetAuthStatusResponse as ConnectorGetAuthStatusResponse,
    type ConnectorHandleOAuthCallbackResponse as ConnectorHandleOAuthCallbackResponse,
    type ConnectorIngestFileResponse as ConnectorIngestFileResponse,
    type ConnectorListFilesResponse as ConnectorListFilesResponse,
    type ConnectorHandleOAuthCallbackParams as ConnectorHandleOAuthCallbackParams,
    type ConnectorIngestFileParams as ConnectorIngestFileParams,
    type ConnectorListFilesParams as ConnectorListFilesParams,
  };

  export {
    Auth as Auth,
    type AuthFinalizeManualAuthResponse as AuthFinalizeManualAuthResponse,
    type AuthGetInitiateAuthURLResponse as AuthGetInitiateAuthURLResponse,
    type AuthFinalizeManualAuthParams as AuthFinalizeManualAuthParams,
    type AuthGetInitiateAuthURLParams as AuthGetInitiateAuthURLParams,
  };
}
