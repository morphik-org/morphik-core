// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import * as AppsAPI from './apps';
import { AppCreateParams, AppCreateResponse, AppDeleteParams, AppDeleteResponse, Apps } from './apps';
import * as ConnectorsAPI from './connectors/connectors';
import {
  ConnectorDisconnectResponse,
  ConnectorGetAuthStatusResponse,
  ConnectorHandleOAuthCallbackParams,
  ConnectorHandleOAuthCallbackResponse,
  ConnectorIngestFileParams,
  ConnectorIngestFileResponse,
  ConnectorListFilesParams,
  ConnectorListFilesResponse,
  Connectors,
} from './connectors/connectors';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';

export class Ee extends APIResource {
  apps: AppsAPI.Apps = new AppsAPI.Apps(this._client);
  connectors: ConnectorsAPI.Connectors = new ConnectorsAPI.Connectors(this._client);

  /**
   * Generate a cloud URI for _request.app_id_ owned by the calling user.
   *
   * The _user_id_ is derived from the bearer token. The caller can therefore not
   * create applications for _other_ users unless their token carries the `admin`
   * permission (mirroring the community behaviour).
   */
  createApp(body: EeCreateAppParams, options?: RequestOptions): APIPromise<EeCreateAppResponse> {
    return this._client.post('/ee/create_app', { body, ...options });
  }
}

export type EeCreateAppResponse = { [key: string]: string };

export interface EeCreateAppParams {
  /**
   * ID of the application
   */
  app_id: string;

  /**
   * Name of the application
   */
  name: string;

  /**
   * Token validity in days
   */
  expiry_days?: number;
}

Ee.Apps = Apps;
Ee.Connectors = Connectors;

export declare namespace Ee {
  export { type EeCreateAppResponse as EeCreateAppResponse, type EeCreateAppParams as EeCreateAppParams };

  export {
    Apps as Apps,
    type AppCreateResponse as AppCreateResponse,
    type AppDeleteResponse as AppDeleteResponse,
    type AppCreateParams as AppCreateParams,
    type AppDeleteParams as AppDeleteParams,
  };

  export {
    Connectors as Connectors,
    type ConnectorDisconnectResponse as ConnectorDisconnectResponse,
    type ConnectorGetAuthStatusResponse as ConnectorGetAuthStatusResponse,
    type ConnectorHandleOAuthCallbackResponse as ConnectorHandleOAuthCallbackResponse,
    type ConnectorIngestFileResponse as ConnectorIngestFileResponse,
    type ConnectorListFilesResponse as ConnectorListFilesResponse,
    type ConnectorHandleOAuthCallbackParams as ConnectorHandleOAuthCallbackParams,
    type ConnectorIngestFileParams as ConnectorIngestFileParams,
    type ConnectorListFilesParams as ConnectorListFilesParams,
  };
}
