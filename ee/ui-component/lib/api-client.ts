/**
 * API client utilities that respect connection type (local vs cloud)
 */
import { ConnectionInfo, createAuthHeaders } from "./connection-utils";

export interface ApiClientConfig {
  connectionInfo: ConnectionInfo | null;
  timeout?: number;
}

export class ApiClient {
  private connectionInfo: ConnectionInfo | null;
  private timeout: number;

  constructor(config: ApiClientConfig) {
    this.connectionInfo = config.connectionInfo;
    this.timeout = config.timeout || 30000;
  }

  /**
   * Make a fetch request with proper authentication based on connection type
   */
  async fetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
    if (!this.connectionInfo) {
      throw new Error("No connection configured");
    }

    const url = `${this.connectionInfo.apiBaseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

    // Build headers properly handling different HeadersInit types
    const authHeaders = createAuthHeaders(this.connectionInfo);
    const headers = new Headers(options.headers);

    // Add auth headers
    Object.entries(authHeaders).forEach(([key, value]) => {
      headers.set(key, value);
    });

    // Only add Content-Type if not already set and not FormData
    if (!headers.has("Content-Type") && !(options.body instanceof FormData)) {
      headers.set("Content-Type", "application/json");
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === "AbortError") {
        throw new Error(`Request timeout after ${this.timeout}ms`);
      }

      throw error;
    }
  }

  /**
   * Make a JSON API request
   */
  async request<T = unknown>(method: string, endpoint: string, data?: unknown, options: RequestInit = {}): Promise<T> {
    const response = await this.fetch(endpoint, {
      ...options,
      method,
      body: data ? JSON.stringify(data) : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "Unknown error");
      throw new Error(`API Error (${response.status}): ${errorText}`);
    }

    return response.json();
  }

  /**
   * Convenience methods
   */
  get<T = unknown>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>("GET", endpoint, undefined, options);
  }

  post<T = unknown>(endpoint: string, data?: unknown, options?: RequestInit): Promise<T> {
    return this.request<T>("POST", endpoint, data, options);
  }

  put<T = unknown>(endpoint: string, data?: unknown, options?: RequestInit): Promise<T> {
    return this.request<T>("PUT", endpoint, data, options);
  }

  delete<T = unknown>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>("DELETE", endpoint, undefined, options);
  }

  /**
   * Update connection info
   */
  updateConnection(connectionInfo: ConnectionInfo | null) {
    this.connectionInfo = connectionInfo;
  }
}
