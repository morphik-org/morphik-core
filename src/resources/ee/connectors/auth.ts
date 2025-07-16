// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../../core/resource';
import { APIPromise } from '../../../core/api-promise';
import { RequestOptions } from '../../../internal/request-options';
import { path } from '../../../internal/utils/path';

export class Auth extends APIResource {
  /**
   * Finalize authentication using manual credentials.
   *
   * This endpoint is used for connectors that require manual credential input (like
   * Zotero) instead of OAuth flows.
   */
  finalizeManualAuth(
    connectorType: string,
    body: AuthFinalizeManualAuthParams,
    options?: RequestOptions,
  ): APIPromise<AuthFinalizeManualAuthResponse> {
    return this._client.post(path`/ee/connectors/${connectorType}/auth/finalize`, { body, ...options });
  }

  /**
   * Return the provider's _authorization_url_ for the given connector.
   *
   * The method mirrors the logic of the `/auth/initiate` endpoint but sends a JSON
   * payload instead of a redirect so that browsers can stay on the same origin until
   * they intentionally navigate away.
   *
   * For OAuth-based connectors, this returns authorization_url. For manual
   * credential connectors, this returns the credential form specification.
   */
  getInitiateAuthURL(
    connectorType: string,
    query: AuthGetInitiateAuthURLParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<AuthGetInitiateAuthURLResponse> {
    return this._client.get(path`/ee/connectors/${connectorType}/auth/initiate_url`, { query, ...options });
  }
}

export type AuthFinalizeManualAuthResponse = { [key: string]: unknown };

export type AuthGetInitiateAuthURLResponse =
  | AuthGetInitiateAuthURLResponse.ManualCredentialsAuthResponse
  | AuthGetInitiateAuthURLResponse.OAuthAuthResponse;

export namespace AuthGetInitiateAuthURLResponse {
  export interface ManualCredentialsAuthResponse {
    auth_type: string;

    required_fields: Array<ManualCredentialsAuthResponse.RequiredField>;

    instructions?: string | null;
  }

  export namespace ManualCredentialsAuthResponse {
    export interface RequiredField {
      description: string;

      label: string;

      name: string;

      required: boolean;

      type: string;

      options?: Array<RequiredField.Option> | null;
    }

    export namespace RequiredField {
      export interface Option {
        label: string;

        value: string;
      }
    }
  }

  export interface OAuthAuthResponse {
    authorization_url: string;
  }
}

export interface AuthFinalizeManualAuthParams {
  credentials: { [key: string]: unknown };
}

export interface AuthGetInitiateAuthURLParams {
  app_redirect_uri?: string | null;
}

export declare namespace Auth {
  export {
    type AuthFinalizeManualAuthResponse as AuthFinalizeManualAuthResponse,
    type AuthGetInitiateAuthURLResponse as AuthGetInitiateAuthURLResponse,
    type AuthFinalizeManualAuthParams as AuthFinalizeManualAuthParams,
    type AuthGetInitiateAuthURLParams as AuthGetInitiateAuthURLParams,
  };
}
