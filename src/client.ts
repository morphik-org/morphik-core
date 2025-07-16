// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import type { RequestInit, RequestInfo, BodyInit } from './internal/builtin-types';
import type { HTTPMethod, PromiseOrValue, MergedRequestInit, FinalizedRequestInit } from './internal/types';
import { uuid4 } from './internal/utils/uuid';
import { validatePositiveInteger, isAbsoluteURL, safeJSON } from './internal/utils/values';
import { sleep } from './internal/utils/sleep';
export type { Logger, LogLevel } from './internal/utils/log';
import { castToError, isAbortError } from './internal/errors';
import type { APIResponseProps } from './internal/parse';
import { getPlatformHeaders } from './internal/detect-platform';
import * as Shims from './internal/shims';
import * as Opts from './internal/request-options';
import * as qs from './internal/qs';
import { VERSION } from './version';
import * as Errors from './core/error';
import * as Uploads from './core/uploads';
import * as API from './resources/index';
import { APIPromise } from './core/api-promise';
import { Agent, AgentExecuteQueryParams, AgentExecuteQueryResponse } from './resources/agent';
import { APIKeyCreateParams, APIKeyCreateResponse, APIKeyListResponse, APIKeys } from './resources/api-keys';
import {
  Batch,
  BatchRetrieveChunksParams,
  BatchRetrieveChunksResponse,
  BatchRetrieveDocumentsParams,
  BatchRetrieveDocumentsResponse,
} from './resources/batch';
import {
  Cache,
  CacheAddDocsParams,
  CacheAddDocsResponse,
  CacheCreateParams,
  CacheCreateResponse,
  CacheQueryParams,
  CacheRetrieveResponse,
  CacheUpdateResponse,
  CompletionResponse,
} from './resources/cache';
import {
  Chat,
  ChatListParams,
  ChatListResponse,
  ChatRetrieveHistoryResponse,
  ChatUpdateTitleParams,
  ChatUpdateTitleResponse,
} from './resources/chat';
import {
  Cloud,
  CloudDeleteAppParams,
  CloudDeleteAppResponse,
  CloudGenerateUriParams,
  CloudGenerateUriResponse,
} from './resources/cloud';
import {
  Document,
  Ingest,
  IngestBatchIngestFilesParams,
  IngestBatchIngestFilesResponse,
  IngestIngestFileParams,
  IngestIngestTextParams,
  TextRequest,
} from './resources/ingest';
import { Local, LocalGenerateUriParams, LocalGenerateUriResponse } from './resources/local';
import { LogListParams, LogListResponse, Logs } from './resources/logs';
import {
  ModelCreateParams,
  ModelDeleteResponse,
  ModelListAvailableResponse,
  ModelListCustomResponse,
  ModelListResponse,
  ModelResponse,
  Models,
} from './resources/models';
import { Ping, PingCheckResponse } from './resources/ping';
import { Query, QueryGenerateCompletionParams } from './resources/query';
import {
  Usage,
  UsageListRecentParams,
  UsageListRecentResponse,
  UsageRetrieveStatsResponse,
} from './resources/usage';
import {
  DocumentDeleteResponse,
  DocumentDownloadFileResponse,
  DocumentGetByFilenameParams,
  DocumentGetDownloadURLParams,
  DocumentGetDownloadURLResponse,
  DocumentGetStatusResponse,
  DocumentListParams,
  DocumentListResponse,
  DocumentUpdateFileParams,
  DocumentUpdateMetadataParams,
  DocumentUpdateTextParams,
  Documents,
} from './resources/documents/documents';
import { Ee, EeCreateAppParams, EeCreateAppResponse } from './resources/ee/ee';
import {
  Folder,
  FolderCreateParams,
  FolderDeleteResponse,
  FolderListResponse,
  FolderListSummariesResponse,
  FolderSetRuleParams,
  FolderSetRuleResponse,
  Folders,
} from './resources/folders/folders';
import {
  EntityExtractionPromptOverride,
  EntityResolutionPromptOverride,
  Graph,
  GraphCreateParams,
  GraphDeleteResponse,
  GraphListParams,
  GraphListResponse,
  GraphPromptOverrides,
  GraphResource,
  GraphRetrieveParams,
  GraphStatusParams,
  GraphStatusResponse,
  GraphUpdateParams,
  GraphVisualizationParams,
  GraphVisualizationResponse,
} from './resources/graph/graph';
import {
  ModelConfig,
  ModelConfigCreateParams,
  ModelConfigDeleteResponse,
  ModelConfigListResponse,
  ModelConfigResource,
  ModelConfigUpdateParams,
} from './resources/model-config/model-config';
import {
  Retrieve,
  RetrieveCreateDocsParams,
  RetrieveCreateDocsResponse,
} from './resources/retrieve/retrieve';
import {
  Workflow,
  WorkflowCreateParams,
  WorkflowDeleteResponse,
  WorkflowListResponse,
  WorkflowRun,
  WorkflowRunParams,
  WorkflowUpdateParams,
  Workflows,
} from './resources/workflows/workflows';
import { type Fetch } from './internal/builtin-types';
import { HeadersLike, NullableHeaders, buildHeaders } from './internal/headers';
import { FinalRequestOptions, RequestOptions } from './internal/request-options';
import { readEnv } from './internal/utils/env';
import {
  type LogLevel,
  type Logger,
  formatRequestDetails,
  loggerFor,
  parseLogLevel,
} from './internal/utils/log';
import { isEmptyObj } from './internal/utils/values';

export interface ClientOptions {
  /**
   * Defaults to process.env['MORPHIK_API_KEY'].
   */
  apiKey?: string | null | undefined;

  /**
   * Override the default base URL for the API, e.g., "https://api.example.com/v2/"
   *
   * Defaults to process.env['MORPHIK_BASE_URL'].
   */
  baseURL?: string | null | undefined;

  /**
   * The maximum amount of time (in milliseconds) that the client should wait for a response
   * from the server before timing out a single request.
   *
   * Note that request timeouts are retried by default, so in a worst-case scenario you may wait
   * much longer than this timeout before the promise succeeds or fails.
   *
   * @unit milliseconds
   */
  timeout?: number | undefined;
  /**
   * Additional `RequestInit` options to be passed to `fetch` calls.
   * Properties will be overridden by per-request `fetchOptions`.
   */
  fetchOptions?: MergedRequestInit | undefined;

  /**
   * Specify a custom `fetch` function implementation.
   *
   * If not provided, we expect that `fetch` is defined globally.
   */
  fetch?: Fetch | undefined;

  /**
   * The maximum number of times that the client will retry a request in case of a
   * temporary failure, like a network error or a 5XX error from the server.
   *
   * @default 2
   */
  maxRetries?: number | undefined;

  /**
   * Default headers to include with every request to the API.
   *
   * These can be removed in individual requests by explicitly setting the
   * header to `null` in request options.
   */
  defaultHeaders?: HeadersLike | undefined;

  /**
   * Default query parameters to include with every request to the API.
   *
   * These can be removed in individual requests by explicitly setting the
   * param to `undefined` in request options.
   */
  defaultQuery?: Record<string, string | undefined> | undefined;

  /**
   * Set the log level.
   *
   * Defaults to process.env['MORPHIK_LOG'] or 'warn' if it isn't set.
   */
  logLevel?: LogLevel | undefined;

  /**
   * Set the logger.
   *
   * Defaults to globalThis.console.
   */
  logger?: Logger | undefined;
}

/**
 * API Client for interfacing with the Morphik API.
 */
export class Morphik {
  apiKey: string | null;

  baseURL: string;
  maxRetries: number;
  timeout: number;
  logger: Logger | undefined;
  logLevel: LogLevel | undefined;
  fetchOptions: MergedRequestInit | undefined;

  private fetch: Fetch;
  #encoder: Opts.RequestEncoder;
  protected idempotencyHeader?: string;
  private _options: ClientOptions;

  /**
   * API Client for interfacing with the Morphik API.
   *
   * @param {string | null | undefined} [opts.apiKey=process.env['MORPHIK_API_KEY'] ?? null]
   * @param {string} [opts.baseURL=process.env['MORPHIK_BASE_URL'] ?? https://api.example.com] - Override the default base URL for the API.
   * @param {number} [opts.timeout=1 minute] - The maximum amount of time (in milliseconds) the client will wait for a response before timing out.
   * @param {MergedRequestInit} [opts.fetchOptions] - Additional `RequestInit` options to be passed to `fetch` calls.
   * @param {Fetch} [opts.fetch] - Specify a custom `fetch` function implementation.
   * @param {number} [opts.maxRetries=2] - The maximum number of times the client will retry a request.
   * @param {HeadersLike} opts.defaultHeaders - Default headers to include with every request to the API.
   * @param {Record<string, string | undefined>} opts.defaultQuery - Default query parameters to include with every request to the API.
   */
  constructor({
    baseURL = readEnv('MORPHIK_BASE_URL'),
    apiKey = readEnv('MORPHIK_API_KEY') ?? null,
    ...opts
  }: ClientOptions = {}) {
    const options: ClientOptions = {
      apiKey,
      ...opts,
      baseURL: baseURL || `https://api.example.com`,
    };

    this.baseURL = options.baseURL!;
    this.timeout = options.timeout ?? Morphik.DEFAULT_TIMEOUT /* 1 minute */;
    this.logger = options.logger ?? console;
    const defaultLogLevel = 'warn';
    // Set default logLevel early so that we can log a warning in parseLogLevel.
    this.logLevel = defaultLogLevel;
    this.logLevel =
      parseLogLevel(options.logLevel, 'ClientOptions.logLevel', this) ??
      parseLogLevel(readEnv('MORPHIK_LOG'), "process.env['MORPHIK_LOG']", this) ??
      defaultLogLevel;
    this.fetchOptions = options.fetchOptions;
    this.maxRetries = options.maxRetries ?? 2;
    this.fetch = options.fetch ?? Shims.getDefaultFetch();
    this.#encoder = Opts.FallbackEncoder;

    this._options = options;

    this.apiKey = apiKey;
  }

  /**
   * Create a new client instance re-using the same options given to the current client with optional overriding.
   */
  withOptions(options: Partial<ClientOptions>): this {
    const client = new (this.constructor as any as new (props: ClientOptions) => typeof this)({
      ...this._options,
      baseURL: this.baseURL,
      maxRetries: this.maxRetries,
      timeout: this.timeout,
      logger: this.logger,
      logLevel: this.logLevel,
      fetch: this.fetch,
      fetchOptions: this.fetchOptions,
      apiKey: this.apiKey,
      ...options,
    });
    return client;
  }

  /**
   * Check whether the base URL is set to its default.
   */
  #baseURLOverridden(): boolean {
    return this.baseURL !== 'https://api.example.com';
  }

  protected defaultQuery(): Record<string, string | undefined> | undefined {
    return this._options.defaultQuery;
  }

  protected validateHeaders({ values, nulls }: NullableHeaders) {
    if (this.apiKey && values.get('authorization')) {
      return;
    }
    if (nulls.has('authorization')) {
      return;
    }

    throw new Error(
      'Could not resolve authentication method. Expected the apiKey to be set. Or for the "Authorization" headers to be explicitly omitted',
    );
  }

  protected async authHeaders(opts: FinalRequestOptions): Promise<NullableHeaders | undefined> {
    if (this.apiKey == null) {
      return undefined;
    }
    return buildHeaders([{ Authorization: `Bearer ${this.apiKey}` }]);
  }

  protected stringifyQuery(query: Record<string, unknown>): string {
    return qs.stringify(query, { arrayFormat: 'comma' });
  }

  private getUserAgent(): string {
    return `${this.constructor.name}/JS ${VERSION}`;
  }

  protected defaultIdempotencyKey(): string {
    return `stainless-node-retry-${uuid4()}`;
  }

  protected makeStatusError(
    status: number,
    error: Object,
    message: string | undefined,
    headers: Headers,
  ): Errors.APIError {
    return Errors.APIError.generate(status, error, message, headers);
  }

  buildURL(
    path: string,
    query: Record<string, unknown> | null | undefined,
    defaultBaseURL?: string | undefined,
  ): string {
    const baseURL = (!this.#baseURLOverridden() && defaultBaseURL) || this.baseURL;
    const url =
      isAbsoluteURL(path) ?
        new URL(path)
      : new URL(baseURL + (baseURL.endsWith('/') && path.startsWith('/') ? path.slice(1) : path));

    const defaultQuery = this.defaultQuery();
    if (!isEmptyObj(defaultQuery)) {
      query = { ...defaultQuery, ...query };
    }

    if (typeof query === 'object' && query && !Array.isArray(query)) {
      url.search = this.stringifyQuery(query as Record<string, unknown>);
    }

    return url.toString();
  }

  /**
   * Used as a callback for mutating the given `FinalRequestOptions` object.
   */
  protected async prepareOptions(options: FinalRequestOptions): Promise<void> {}

  /**
   * Used as a callback for mutating the given `RequestInit` object.
   *
   * This is useful for cases where you want to add certain headers based off of
   * the request properties, e.g. `method` or `url`.
   */
  protected async prepareRequest(
    request: RequestInit,
    { url, options }: { url: string; options: FinalRequestOptions },
  ): Promise<void> {}

  get<Rsp>(path: string, opts?: PromiseOrValue<RequestOptions>): APIPromise<Rsp> {
    return this.methodRequest('get', path, opts);
  }

  post<Rsp>(path: string, opts?: PromiseOrValue<RequestOptions>): APIPromise<Rsp> {
    return this.methodRequest('post', path, opts);
  }

  patch<Rsp>(path: string, opts?: PromiseOrValue<RequestOptions>): APIPromise<Rsp> {
    return this.methodRequest('patch', path, opts);
  }

  put<Rsp>(path: string, opts?: PromiseOrValue<RequestOptions>): APIPromise<Rsp> {
    return this.methodRequest('put', path, opts);
  }

  delete<Rsp>(path: string, opts?: PromiseOrValue<RequestOptions>): APIPromise<Rsp> {
    return this.methodRequest('delete', path, opts);
  }

  private methodRequest<Rsp>(
    method: HTTPMethod,
    path: string,
    opts?: PromiseOrValue<RequestOptions>,
  ): APIPromise<Rsp> {
    return this.request(
      Promise.resolve(opts).then((opts) => {
        return { method, path, ...opts };
      }),
    );
  }

  request<Rsp>(
    options: PromiseOrValue<FinalRequestOptions>,
    remainingRetries: number | null = null,
  ): APIPromise<Rsp> {
    return new APIPromise(this, this.makeRequest(options, remainingRetries, undefined));
  }

  private async makeRequest(
    optionsInput: PromiseOrValue<FinalRequestOptions>,
    retriesRemaining: number | null,
    retryOfRequestLogID: string | undefined,
  ): Promise<APIResponseProps> {
    const options = await optionsInput;
    const maxRetries = options.maxRetries ?? this.maxRetries;
    if (retriesRemaining == null) {
      retriesRemaining = maxRetries;
    }

    await this.prepareOptions(options);

    const { req, url, timeout } = await this.buildRequest(options, {
      retryCount: maxRetries - retriesRemaining,
    });

    await this.prepareRequest(req, { url, options });

    /** Not an API request ID, just for correlating local log entries. */
    const requestLogID = 'log_' + ((Math.random() * (1 << 24)) | 0).toString(16).padStart(6, '0');
    const retryLogStr = retryOfRequestLogID === undefined ? '' : `, retryOf: ${retryOfRequestLogID}`;
    const startTime = Date.now();

    loggerFor(this).debug(
      `[${requestLogID}] sending request`,
      formatRequestDetails({
        retryOfRequestLogID,
        method: options.method,
        url,
        options,
        headers: req.headers,
      }),
    );

    if (options.signal?.aborted) {
      throw new Errors.APIUserAbortError();
    }

    const controller = new AbortController();
    const response = await this.fetchWithTimeout(url, req, timeout, controller).catch(castToError);
    const headersTime = Date.now();

    if (response instanceof Error) {
      const retryMessage = `retrying, ${retriesRemaining} attempts remaining`;
      if (options.signal?.aborted) {
        throw new Errors.APIUserAbortError();
      }
      // detect native connection timeout errors
      // deno throws "TypeError: error sending request for url (https://example/): client error (Connect): tcp connect error: Operation timed out (os error 60): Operation timed out (os error 60)"
      // undici throws "TypeError: fetch failed" with cause "ConnectTimeoutError: Connect Timeout Error (attempted address: example:443, timeout: 1ms)"
      // others do not provide enough information to distinguish timeouts from other connection errors
      const isTimeout =
        isAbortError(response) ||
        /timed? ?out/i.test(String(response) + ('cause' in response ? String(response.cause) : ''));
      if (retriesRemaining) {
        loggerFor(this).info(
          `[${requestLogID}] connection ${isTimeout ? 'timed out' : 'failed'} - ${retryMessage}`,
        );
        loggerFor(this).debug(
          `[${requestLogID}] connection ${isTimeout ? 'timed out' : 'failed'} (${retryMessage})`,
          formatRequestDetails({
            retryOfRequestLogID,
            url,
            durationMs: headersTime - startTime,
            message: response.message,
          }),
        );
        return this.retryRequest(options, retriesRemaining, retryOfRequestLogID ?? requestLogID);
      }
      loggerFor(this).info(
        `[${requestLogID}] connection ${isTimeout ? 'timed out' : 'failed'} - error; no more retries left`,
      );
      loggerFor(this).debug(
        `[${requestLogID}] connection ${isTimeout ? 'timed out' : 'failed'} (error; no more retries left)`,
        formatRequestDetails({
          retryOfRequestLogID,
          url,
          durationMs: headersTime - startTime,
          message: response.message,
        }),
      );
      if (isTimeout) {
        throw new Errors.APIConnectionTimeoutError();
      }
      throw new Errors.APIConnectionError({ cause: response });
    }

    const responseInfo = `[${requestLogID}${retryLogStr}] ${req.method} ${url} ${
      response.ok ? 'succeeded' : 'failed'
    } with status ${response.status} in ${headersTime - startTime}ms`;

    if (!response.ok) {
      const shouldRetry = await this.shouldRetry(response);
      if (retriesRemaining && shouldRetry) {
        const retryMessage = `retrying, ${retriesRemaining} attempts remaining`;

        // We don't need the body of this response.
        await Shims.CancelReadableStream(response.body);
        loggerFor(this).info(`${responseInfo} - ${retryMessage}`);
        loggerFor(this).debug(
          `[${requestLogID}] response error (${retryMessage})`,
          formatRequestDetails({
            retryOfRequestLogID,
            url: response.url,
            status: response.status,
            headers: response.headers,
            durationMs: headersTime - startTime,
          }),
        );
        return this.retryRequest(
          options,
          retriesRemaining,
          retryOfRequestLogID ?? requestLogID,
          response.headers,
        );
      }

      const retryMessage = shouldRetry ? `error; no more retries left` : `error; not retryable`;

      loggerFor(this).info(`${responseInfo} - ${retryMessage}`);

      const errText = await response.text().catch((err: any) => castToError(err).message);
      const errJSON = safeJSON(errText);
      const errMessage = errJSON ? undefined : errText;

      loggerFor(this).debug(
        `[${requestLogID}] response error (${retryMessage})`,
        formatRequestDetails({
          retryOfRequestLogID,
          url: response.url,
          status: response.status,
          headers: response.headers,
          message: errMessage,
          durationMs: Date.now() - startTime,
        }),
      );

      const err = this.makeStatusError(response.status, errJSON, errMessage, response.headers);
      throw err;
    }

    loggerFor(this).info(responseInfo);
    loggerFor(this).debug(
      `[${requestLogID}] response start`,
      formatRequestDetails({
        retryOfRequestLogID,
        url: response.url,
        status: response.status,
        headers: response.headers,
        durationMs: headersTime - startTime,
      }),
    );

    return { response, options, controller, requestLogID, retryOfRequestLogID, startTime };
  }

  async fetchWithTimeout(
    url: RequestInfo,
    init: RequestInit | undefined,
    ms: number,
    controller: AbortController,
  ): Promise<Response> {
    const { signal, method, ...options } = init || {};
    if (signal) signal.addEventListener('abort', () => controller.abort());

    const timeout = setTimeout(() => controller.abort(), ms);

    const isReadableBody =
      ((globalThis as any).ReadableStream && options.body instanceof (globalThis as any).ReadableStream) ||
      (typeof options.body === 'object' && options.body !== null && Symbol.asyncIterator in options.body);

    const fetchOptions: RequestInit = {
      signal: controller.signal as any,
      ...(isReadableBody ? { duplex: 'half' } : {}),
      method: 'GET',
      ...options,
    };
    if (method) {
      // Custom methods like 'patch' need to be uppercased
      // See https://github.com/nodejs/undici/issues/2294
      fetchOptions.method = method.toUpperCase();
    }

    try {
      // use undefined this binding; fetch errors if bound to something else in browser/cloudflare
      return await this.fetch.call(undefined, url, fetchOptions);
    } finally {
      clearTimeout(timeout);
    }
  }

  private async shouldRetry(response: Response): Promise<boolean> {
    // Note this is not a standard header.
    const shouldRetryHeader = response.headers.get('x-should-retry');

    // If the server explicitly says whether or not to retry, obey.
    if (shouldRetryHeader === 'true') return true;
    if (shouldRetryHeader === 'false') return false;

    // Retry on request timeouts.
    if (response.status === 408) return true;

    // Retry on lock timeouts.
    if (response.status === 409) return true;

    // Retry on rate limits.
    if (response.status === 429) return true;

    // Retry internal errors.
    if (response.status >= 500) return true;

    return false;
  }

  private async retryRequest(
    options: FinalRequestOptions,
    retriesRemaining: number,
    requestLogID: string,
    responseHeaders?: Headers | undefined,
  ): Promise<APIResponseProps> {
    let timeoutMillis: number | undefined;

    // Note the `retry-after-ms` header may not be standard, but is a good idea and we'd like proactive support for it.
    const retryAfterMillisHeader = responseHeaders?.get('retry-after-ms');
    if (retryAfterMillisHeader) {
      const timeoutMs = parseFloat(retryAfterMillisHeader);
      if (!Number.isNaN(timeoutMs)) {
        timeoutMillis = timeoutMs;
      }
    }

    // About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
    const retryAfterHeader = responseHeaders?.get('retry-after');
    if (retryAfterHeader && !timeoutMillis) {
      const timeoutSeconds = parseFloat(retryAfterHeader);
      if (!Number.isNaN(timeoutSeconds)) {
        timeoutMillis = timeoutSeconds * 1000;
      } else {
        timeoutMillis = Date.parse(retryAfterHeader) - Date.now();
      }
    }

    // If the API asks us to wait a certain amount of time (and it's a reasonable amount),
    // just do what it says, but otherwise calculate a default
    if (!(timeoutMillis && 0 <= timeoutMillis && timeoutMillis < 60 * 1000)) {
      const maxRetries = options.maxRetries ?? this.maxRetries;
      timeoutMillis = this.calculateDefaultRetryTimeoutMillis(retriesRemaining, maxRetries);
    }
    await sleep(timeoutMillis);

    return this.makeRequest(options, retriesRemaining - 1, requestLogID);
  }

  private calculateDefaultRetryTimeoutMillis(retriesRemaining: number, maxRetries: number): number {
    const initialRetryDelay = 0.5;
    const maxRetryDelay = 8.0;

    const numRetries = maxRetries - retriesRemaining;

    // Apply exponential backoff, but not more than the max.
    const sleepSeconds = Math.min(initialRetryDelay * Math.pow(2, numRetries), maxRetryDelay);

    // Apply some jitter, take up to at most 25 percent of the retry time.
    const jitter = 1 - Math.random() * 0.25;

    return sleepSeconds * jitter * 1000;
  }

  async buildRequest(
    inputOptions: FinalRequestOptions,
    { retryCount = 0 }: { retryCount?: number } = {},
  ): Promise<{ req: FinalizedRequestInit; url: string; timeout: number }> {
    const options = { ...inputOptions };
    const { method, path, query, defaultBaseURL } = options;

    const url = this.buildURL(path!, query as Record<string, unknown>, defaultBaseURL);
    if ('timeout' in options) validatePositiveInteger('timeout', options.timeout);
    options.timeout = options.timeout ?? this.timeout;
    const { bodyHeaders, body } = this.buildBody({ options });
    const reqHeaders = await this.buildHeaders({ options: inputOptions, method, bodyHeaders, retryCount });

    const req: FinalizedRequestInit = {
      method,
      headers: reqHeaders,
      ...(options.signal && { signal: options.signal }),
      ...((globalThis as any).ReadableStream &&
        body instanceof (globalThis as any).ReadableStream && { duplex: 'half' }),
      ...(body && { body }),
      ...((this.fetchOptions as any) ?? {}),
      ...((options.fetchOptions as any) ?? {}),
    };

    return { req, url, timeout: options.timeout };
  }

  private async buildHeaders({
    options,
    method,
    bodyHeaders,
    retryCount,
  }: {
    options: FinalRequestOptions;
    method: HTTPMethod;
    bodyHeaders: HeadersLike;
    retryCount: number;
  }): Promise<Headers> {
    let idempotencyHeaders: HeadersLike = {};
    if (this.idempotencyHeader && method !== 'get') {
      if (!options.idempotencyKey) options.idempotencyKey = this.defaultIdempotencyKey();
      idempotencyHeaders[this.idempotencyHeader] = options.idempotencyKey;
    }

    const headers = buildHeaders([
      idempotencyHeaders,
      {
        Accept: 'application/json',
        'User-Agent': this.getUserAgent(),
        'X-Stainless-Retry-Count': String(retryCount),
        ...(options.timeout ? { 'X-Stainless-Timeout': String(Math.trunc(options.timeout / 1000)) } : {}),
        ...getPlatformHeaders(),
      },
      await this.authHeaders(options),
      this._options.defaultHeaders,
      bodyHeaders,
      options.headers,
    ]);

    this.validateHeaders(headers);

    return headers.values;
  }

  private buildBody({ options: { body, headers: rawHeaders } }: { options: FinalRequestOptions }): {
    bodyHeaders: HeadersLike;
    body: BodyInit | undefined;
  } {
    if (!body) {
      return { bodyHeaders: undefined, body: undefined };
    }
    const headers = buildHeaders([rawHeaders]);
    if (
      // Pass raw type verbatim
      ArrayBuffer.isView(body) ||
      body instanceof ArrayBuffer ||
      body instanceof DataView ||
      (typeof body === 'string' &&
        // Preserve legacy string encoding behavior for now
        headers.values.has('content-type')) ||
      // `Blob` is superset of `File`
      body instanceof Blob ||
      // `FormData` -> `multipart/form-data`
      body instanceof FormData ||
      // `URLSearchParams` -> `application/x-www-form-urlencoded`
      body instanceof URLSearchParams ||
      // Send chunked stream (each chunk has own `length`)
      ((globalThis as any).ReadableStream && body instanceof (globalThis as any).ReadableStream)
    ) {
      return { bodyHeaders: undefined, body: body as BodyInit };
    } else if (
      typeof body === 'object' &&
      (Symbol.asyncIterator in body ||
        (Symbol.iterator in body && 'next' in body && typeof body.next === 'function'))
    ) {
      return { bodyHeaders: undefined, body: Shims.ReadableStreamFrom(body as AsyncIterable<Uint8Array>) };
    } else {
      return this.#encoder({ body, headers });
    }
  }

  static Morphik = this;
  static DEFAULT_TIMEOUT = 60000; // 1 minute

  static MorphikError = Errors.MorphikError;
  static APIError = Errors.APIError;
  static APIConnectionError = Errors.APIConnectionError;
  static APIConnectionTimeoutError = Errors.APIConnectionTimeoutError;
  static APIUserAbortError = Errors.APIUserAbortError;
  static NotFoundError = Errors.NotFoundError;
  static ConflictError = Errors.ConflictError;
  static RateLimitError = Errors.RateLimitError;
  static BadRequestError = Errors.BadRequestError;
  static AuthenticationError = Errors.AuthenticationError;
  static InternalServerError = Errors.InternalServerError;
  static PermissionDeniedError = Errors.PermissionDeniedError;
  static UnprocessableEntityError = Errors.UnprocessableEntityError;

  static toFile = Uploads.toFile;

  ping: API.Ping = new API.Ping(this);
  models: API.Models = new API.Models(this);
  ingest: API.Ingest = new API.Ingest(this);
  folders: API.Folders = new API.Folders(this);
  workflows: API.Workflows = new API.Workflows(this);
  modelConfig: API.ModelConfigResource = new API.ModelConfigResource(this);
  apiKeys: API.APIKeys = new API.APIKeys(this);
  logs: API.Logs = new API.Logs(this);
  cache: API.Cache = new API.Cache(this);
  graph: API.GraphResource = new API.GraphResource(this);
  ee: API.Ee = new API.Ee(this);
  retrieve: API.Retrieve = new API.Retrieve(this);
  batch: API.Batch = new API.Batch(this);
  query: API.Query = new API.Query(this);
  agent: API.Agent = new API.Agent(this);
  usage: API.Usage = new API.Usage(this);
  local: API.Local = new API.Local(this);
  cloud: API.Cloud = new API.Cloud(this);
  documents: API.Documents = new API.Documents(this);
  chat: API.Chat = new API.Chat(this);
}
Morphik.Ping = Ping;
Morphik.Models = Models;
Morphik.Ingest = Ingest;
Morphik.Folders = Folders;
Morphik.Workflows = Workflows;
Morphik.ModelConfigResource = ModelConfigResource;
Morphik.APIKeys = APIKeys;
Morphik.Logs = Logs;
Morphik.Cache = Cache;
Morphik.GraphResource = GraphResource;
Morphik.Ee = Ee;
Morphik.Retrieve = Retrieve;
Morphik.Batch = Batch;
Morphik.Query = Query;
Morphik.Agent = Agent;
Morphik.Usage = Usage;
Morphik.Local = Local;
Morphik.Cloud = Cloud;
Morphik.Documents = Documents;
Morphik.Chat = Chat;
export declare namespace Morphik {
  export type RequestOptions = Opts.RequestOptions;

  export { Ping as Ping, type PingCheckResponse as PingCheckResponse };

  export {
    Models as Models,
    type ModelResponse as ModelResponse,
    type ModelListResponse as ModelListResponse,
    type ModelDeleteResponse as ModelDeleteResponse,
    type ModelListAvailableResponse as ModelListAvailableResponse,
    type ModelListCustomResponse as ModelListCustomResponse,
    type ModelCreateParams as ModelCreateParams,
  };

  export {
    Ingest as Ingest,
    type Document as Document,
    type TextRequest as TextRequest,
    type IngestBatchIngestFilesResponse as IngestBatchIngestFilesResponse,
    type IngestBatchIngestFilesParams as IngestBatchIngestFilesParams,
    type IngestIngestFileParams as IngestIngestFileParams,
    type IngestIngestTextParams as IngestIngestTextParams,
  };

  export {
    Folders as Folders,
    type Folder as Folder,
    type FolderListResponse as FolderListResponse,
    type FolderDeleteResponse as FolderDeleteResponse,
    type FolderListSummariesResponse as FolderListSummariesResponse,
    type FolderSetRuleResponse as FolderSetRuleResponse,
    type FolderCreateParams as FolderCreateParams,
    type FolderSetRuleParams as FolderSetRuleParams,
  };

  export {
    Workflows as Workflows,
    type Workflow as Workflow,
    type WorkflowRun as WorkflowRun,
    type WorkflowListResponse as WorkflowListResponse,
    type WorkflowDeleteResponse as WorkflowDeleteResponse,
    type WorkflowCreateParams as WorkflowCreateParams,
    type WorkflowUpdateParams as WorkflowUpdateParams,
    type WorkflowRunParams as WorkflowRunParams,
  };

  export {
    ModelConfigResource as ModelConfigResource,
    type ModelConfig as ModelConfig,
    type ModelConfigListResponse as ModelConfigListResponse,
    type ModelConfigDeleteResponse as ModelConfigDeleteResponse,
    type ModelConfigCreateParams as ModelConfigCreateParams,
    type ModelConfigUpdateParams as ModelConfigUpdateParams,
  };

  export {
    APIKeys as APIKeys,
    type APIKeyCreateResponse as APIKeyCreateResponse,
    type APIKeyListResponse as APIKeyListResponse,
    type APIKeyCreateParams as APIKeyCreateParams,
  };

  export { Logs as Logs, type LogListResponse as LogListResponse, type LogListParams as LogListParams };

  export {
    Cache as Cache,
    type CompletionResponse as CompletionResponse,
    type CacheCreateResponse as CacheCreateResponse,
    type CacheRetrieveResponse as CacheRetrieveResponse,
    type CacheUpdateResponse as CacheUpdateResponse,
    type CacheAddDocsResponse as CacheAddDocsResponse,
    type CacheCreateParams as CacheCreateParams,
    type CacheAddDocsParams as CacheAddDocsParams,
    type CacheQueryParams as CacheQueryParams,
  };

  export {
    GraphResource as GraphResource,
    type EntityExtractionPromptOverride as EntityExtractionPromptOverride,
    type EntityResolutionPromptOverride as EntityResolutionPromptOverride,
    type Graph as Graph,
    type GraphPromptOverrides as GraphPromptOverrides,
    type GraphListResponse as GraphListResponse,
    type GraphDeleteResponse as GraphDeleteResponse,
    type GraphStatusResponse as GraphStatusResponse,
    type GraphVisualizationResponse as GraphVisualizationResponse,
    type GraphCreateParams as GraphCreateParams,
    type GraphRetrieveParams as GraphRetrieveParams,
    type GraphUpdateParams as GraphUpdateParams,
    type GraphListParams as GraphListParams,
    type GraphStatusParams as GraphStatusParams,
    type GraphVisualizationParams as GraphVisualizationParams,
  };

  export {
    Ee as Ee,
    type EeCreateAppResponse as EeCreateAppResponse,
    type EeCreateAppParams as EeCreateAppParams,
  };

  export {
    Retrieve as Retrieve,
    type RetrieveCreateDocsResponse as RetrieveCreateDocsResponse,
    type RetrieveCreateDocsParams as RetrieveCreateDocsParams,
  };

  export {
    Batch as Batch,
    type BatchRetrieveChunksResponse as BatchRetrieveChunksResponse,
    type BatchRetrieveDocumentsResponse as BatchRetrieveDocumentsResponse,
    type BatchRetrieveChunksParams as BatchRetrieveChunksParams,
    type BatchRetrieveDocumentsParams as BatchRetrieveDocumentsParams,
  };

  export { Query as Query, type QueryGenerateCompletionParams as QueryGenerateCompletionParams };

  export {
    Agent as Agent,
    type AgentExecuteQueryResponse as AgentExecuteQueryResponse,
    type AgentExecuteQueryParams as AgentExecuteQueryParams,
  };

  export {
    Usage as Usage,
    type UsageListRecentResponse as UsageListRecentResponse,
    type UsageRetrieveStatsResponse as UsageRetrieveStatsResponse,
    type UsageListRecentParams as UsageListRecentParams,
  };

  export {
    Local as Local,
    type LocalGenerateUriResponse as LocalGenerateUriResponse,
    type LocalGenerateUriParams as LocalGenerateUriParams,
  };

  export {
    Cloud as Cloud,
    type CloudDeleteAppResponse as CloudDeleteAppResponse,
    type CloudGenerateUriResponse as CloudGenerateUriResponse,
    type CloudDeleteAppParams as CloudDeleteAppParams,
    type CloudGenerateUriParams as CloudGenerateUriParams,
  };

  export {
    Documents as Documents,
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
    type ChatListResponse as ChatListResponse,
    type ChatRetrieveHistoryResponse as ChatRetrieveHistoryResponse,
    type ChatUpdateTitleResponse as ChatUpdateTitleResponse,
    type ChatListParams as ChatListParams,
    type ChatUpdateTitleParams as ChatUpdateTitleParams,
  };
}
