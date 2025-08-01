/**
 * Connection utilities for handling different URI types (local vs cloud)
 */

export interface ConnectionInfo {
  type: "local" | "cloud";
  apiBaseUrl: string;
  authToken: string | null;
  originalUri: string;
  isSecure: boolean; // Whether using HTTPS
}

// Storage key for connection URI
export const CONNECTION_URI_STORAGE_KEY = "morphik-connection-uri";

/**
 * Determine if a URI is for local development
 */
export function isLocalUri(uri: string): boolean {
  // Check for localhost, 127.0.0.1, or local domain
  const localPatterns = ["localhost", "127.0.0.1", "0.0.0.0", ".local", "host.docker.internal"];

  const uriLower = uri.toLowerCase();
  return localPatterns.some(pattern => uriLower.includes(pattern));
}

/**
 * Parse any URI format into connection info
 */
export function parseConnectionUri(uri: string): ConnectionInfo {
  if (!uri || uri.trim() === "") {
    // Default local connection
    return {
      type: "local",
      apiBaseUrl: "http://localhost:8000",
      authToken: null,
      originalUri: uri,
      isSecure: false,
    };
  }

  // Clean up malformed URIs that have both https:// and morphik://
  let cleanUri = uri;
  if (uri.includes("://morphik://")) {
    // Remove the leading protocol if someone entered https://morphik://...
    cleanUri = uri.replace(/^https?:\/\//, "");
    console.warn("Cleaning malformed URI:", uri, "→", cleanUri);
  }

  // Check if it's already a morphik:// URI
  if (cleanUri.startsWith("morphik://")) {
    // Parse morphik://appname:token@host or morphik://appname:token@protocol:host format
    const match = cleanUri.match(/^morphik:\/\/([^:]+):([^@]+)@(.+)/);
    if (match) {
      const [, , token, hostPart] = match;

      // Check if protocol is explicitly specified (e.g., http:example.com or https:example.com)
      let apiBaseUrl: string;
      let isSecure: boolean;

      if (hostPart.startsWith("https:")) {
        apiBaseUrl = `https://${hostPart.substring(6)}`;
        isSecure = true;
      } else if (hostPart.startsWith("http:")) {
        apiBaseUrl = `http://${hostPart.substring(5)}`;
        isSecure = false;
      } else {
        // No protocol specified - default to HTTP (not HTTPS)
        apiBaseUrl = `http://${hostPart}`;
        isSecure = false;
      }

      const isLocal = isLocalUri(hostPart);

      return {
        type: isLocal ? "local" : "cloud",
        apiBaseUrl,
        authToken: token,
        originalUri: uri,
        isSecure,
      };
    }
  }

  // Check if it's a plain HTTP/HTTPS URL
  if (uri.startsWith("http://") || uri.startsWith("https://")) {
    const isLocal = isLocalUri(uri);
    const isSecure = uri.startsWith("https://");
    return {
      type: isLocal ? "local" : "cloud",
      apiBaseUrl: uri,
      authToken: null,
      originalUri: uri,
      isSecure,
    };
  }

  // Plain host:port format - default to HTTP
  const host = uri.includes("://") ? uri : `http://${uri}`;
  const isLocal = isLocalUri(host);

  return {
    type: isLocal ? "local" : "cloud",
    apiBaseUrl: host,
    authToken: null,
    originalUri: uri,
    isSecure: false,
  };
}

/**
 * Generate a Morphik URI from connection info
 *
 * Examples:
 * - generateMorphikUri('app', 'token', 'api.example.com') → 'morphik://app:token@api.example.com'
 * - generateMorphikUri('app', 'token', 'http://api.example.com') → 'morphik://app:token@http:api.example.com'
 */
export function generateMorphikUri(
  appName: string,
  token: string,
  host: string,
  forceProtocol?: "http" | "https"
): string {
  // If host includes protocol, preserve it in the morphik URI
  if (host.startsWith("http://")) {
    const cleanHost = host.replace(/^http:\/\//, "");
    return `morphik://${appName}:${token}@http:${cleanHost}`;
  } else if (host.startsWith("https://")) {
    const cleanHost = host.replace(/^https:\/\//, "");
    return `morphik://${appName}:${token}@https:${cleanHost}`;
  } else if (forceProtocol) {
    // If protocol is forced, include it
    return `morphik://${appName}:${token}@${forceProtocol}:${host}`;
  } else {
    // No protocol specified - default format
    return `morphik://${appName}:${token}@${host}`;
  }
}

/**
 * Create authorization headers based on connection type
 */
export function createAuthHeaders(connection: ConnectionInfo, contentType?: string): HeadersInit {
  const headers: HeadersInit = {};

  // Add auth header if token exists (regardless of connection type)
  // This supports both cloud connections and local dev servers with auth
  if (connection.authToken) {
    headers["Authorization"] = `Bearer ${connection.authToken}`;
  }

  if (contentType) {
    headers["Content-Type"] = contentType;
  }

  return headers;
}

/**
 * Clear stored connection URI if it's a local connection
 */
export function clearLocalConnectionFromStorage(): void {
  if (typeof window === "undefined") return;

  try {
    const stored = window.localStorage.getItem(CONNECTION_URI_STORAGE_KEY);
    if (stored && isLocalUri(stored)) {
      console.log("Clearing local connection from storage:", stored);
      window.localStorage.removeItem(CONNECTION_URI_STORAGE_KEY);
    }
  } catch {
    // Ignore errors
  }
}

/**
 * Check if we should auto-clear local connections
 * Can be called on app startup or when switching between pages
 */
export function shouldClearLocalConnection(): boolean {
  if (typeof window === "undefined") return false;

  try {
    const stored = window.localStorage.getItem(CONNECTION_URI_STORAGE_KEY);
    return stored ? isLocalUri(stored) : false;
  } catch {
    return false;
  }
}
