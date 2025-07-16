// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import type { Morphik } from '../client';

export abstract class APIResource {
  protected _client: Morphik;

  constructor(client: Morphik) {
    this._client = client;
  }
}
