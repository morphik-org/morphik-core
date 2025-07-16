// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { buildHeaders } from '../internal/headers';
import { RequestOptions } from '../internal/request-options';

export class Local extends APIResource {
  /**
   * Generate a development URI for running Morphik locally.
   *
   * Args: name: Developer name to embed in the token payload. expiry_days: Number of
   * days the generated token should remain valid.
   *
   * Returns: A dictionary containing the `uri` that can be used to connect to the
   * local instance.
   */
  generateUri(
    body: LocalGenerateUriParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<LocalGenerateUriResponse> {
    return this._client.post('/local/generate_uri', {
      body,
      ...options,
      headers: buildHeaders([{ 'Content-Type': 'application/x-www-form-urlencoded' }, options?.headers]),
    });
  }
}

export type LocalGenerateUriResponse = { [key: string]: string };

export interface LocalGenerateUriParams {
  expiry_days?: number;

  name?: string;
}

export declare namespace Local {
  export {
    type LocalGenerateUriResponse as LocalGenerateUriResponse,
    type LocalGenerateUriParams as LocalGenerateUriParams,
  };
}
