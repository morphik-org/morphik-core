import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines class names using clsx and tailwind-merge
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Extracts an auth token from a Morphik URI
 *
 * @param uri - The Morphik connection URI (format: morphik://appname:token@host)
 * @returns The extracted token or null if invalid
 */
export function extractTokenFromUri(uri: string | undefined): string | null {
  if (!uri) return null;

  try {
    console.log("Attempting to extract token from URI:", uri);
    // The URI format is: morphik://appname:token@host
    // We need to extract just the token part (after the colon and before the @)

    // First get everything before the @ symbol
    const beforeAt = uri.split("@")[0];
    if (!beforeAt) return null;

    // Now extract the token after the colon
    const parts = beforeAt.split("://");
    if (parts.length < 2) return null;

    // Check if there's a colon in the second part, which separates app name from token
    const authPart = parts[1];
    if (authPart.includes(":")) {
      // Format is appname:token
      const fullToken = authPart.split(":")[1];
      console.log("Extracted token:", fullToken ? `${fullToken.substring(0, 5)}...` : "null");
      return fullToken;
    } else {
      // Old format with just token (no appname:)
      console.log("Extracted token (old format):", authPart ? `${authPart.substring(0, 5)}...` : "null");
      return authPart;
    }
  } catch (err) {
    console.error("Error extracting token from URI:", err);
    return null;
  }
}

/**
 * Extracts the host from a Morphik URI and creates an API base URL
 *
 * @param uri - The Morphik connection URI (format: morphik://appname:token@host)
 * @param defaultUrl - The default API URL to use if URI is invalid
 * @returns The API base URL derived from the URI host
 */
export function getApiBaseUrlFromUri(uri: string | undefined, defaultUrl: string = "http://localhost:8000"): string {
  // If URI is empty or undefined, connect to the default URL
  if (!uri || uri.trim() === "") {
    return defaultUrl;
  }

  try {
    // Expected format: morphik://{token}@{host}
    const match = uri.match(/^morphik:\/\/[^@]+@(.+)/);
    if (!match || !match[1]) return defaultUrl; // Default if invalid format

    // Get the host part
    let host = match[1];

    // If it's local, localhost or 127.0.0.1, ensure http:// protocol and add port if needed
    if (host.includes("local") || host.includes("127.0.0.1")) {
      if (!host.includes("://")) {
        host = `http://${host}`;
      }
      // Add default port 8000 if no port specified
      if (!host.includes(":") || host.endsWith(":")) {
        host = `${host.replace(/:$/, "")}:8000`;
      }
    } else {
      // For other hosts, ensure https:// protocol
      if (!host.includes("://")) {
        host = `https://${host}`;
      }
    }

    console.log("Extracted API base URL:", host);
    return host;
  } catch (err) {
    console.error("Error extracting host from URI:", err);
    return defaultUrl; // Default on error
  }
}

/**
 * Creates authorization headers for API requests
 *
 * @param token - The auth token
 * @param contentType - Optional content type header
 * @returns Headers object with authorization
 */
export function createAuthHeaders(token: string | null, contentType?: string): HeadersInit {
  const headers: HeadersInit = {};

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  if (contentType) {
    headers["Content-Type"] = contentType;
  }

  return headers;
}

/**
 * Generate a UUID v4 string
 * This is a simple implementation for client-side use
 */
export function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0,
      v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Converts a simple HTTP URL to Morphik URI format for local development
 * If already in Morphik format, returns as-is
 *
 * @param input - HTTP URL or Morphik URI
 * @returns Morphik URI format or the original input if already valid
 */
export function normalizeToMorphikUri(input: string): string {
  if (!input || input.trim() === "") {
    return "";
  }

  // If already in morphik:// format, return as-is
  if (input.startsWith("morphik://")) {
    return input;
  }

  // If it's a simple HTTP/HTTPS URL, convert to morphik format
  if (input.startsWith("http://") || input.startsWith("https://")) {
    // For local development, create a simple morphik URI
    // Format: morphik://local@host
    const url = new URL(input);
    return `morphik://local@${url.host}`;
  }

  // If it's just a host:port, assume http and convert
  if (input.includes(":") && !input.includes("://")) {
    return `morphik://local@${input}`;
  }

  // Default case - assume it's a host and add localhost prefix
  return `morphik://local@${input}`;
}

/**
 * Format date to relative time (e.g. "2 hours ago")
 */
export function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return "just now";
  }

  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return `${diffInMinutes} minute${diffInMinutes > 1 ? "s" : ""} ago`;
  }

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return `${diffInHours} hour${diffInHours > 1 ? "s" : ""} ago`;
  }

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 30) {
    return `${diffInDays} day${diffInDays > 1 ? "s" : ""} ago`;
  }

  const diffInMonths = Math.floor(diffInDays / 30);
  if (diffInMonths < 12) {
    return `${diffInMonths} month${diffInMonths > 1 ? "s" : ""} ago`;
  }

  const diffInYears = Math.floor(diffInMonths / 12);
  return `${diffInYears} year${diffInYears > 1 ? "s" : ""} ago`;
}
