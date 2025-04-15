import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
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
    console.log('Attempting to extract token from URI:', uri);
    // The URI format is: morphik://appname:token@host
    // We need to extract just the token part (after the colon and before the @)
    
    // First get everything before the @ symbol
    const beforeAt = uri.split('@')[0];
    if (!beforeAt) return null;
    
    // Now extract the token after the colon
    const parts = beforeAt.split('://');
    if (parts.length < 2) return null;
    
    // Check if there's a colon in the second part, which separates app name from token
    const authPart = parts[1];
    if (authPart.includes(':')) {
      // Format is appname:token
      const fullToken = authPart.split(':')[1];
      console.log('Extracted token:', fullToken ? `${fullToken.substring(0, 5)}...` : 'null');
      return fullToken;
    } else {
      // Old format with just token (no appname:)
      console.log('Extracted token (old format):', authPart ? `${authPart.substring(0, 5)}...` : 'null');
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
export function getApiBaseUrlFromUri(uri: string | undefined, defaultUrl: string = 'http://localhost:8000'): string {
  // If URI is empty or undefined, connect to the default URL
  if (!uri || uri.trim() === '') {
    return defaultUrl;
  }
  
  try {
    // Expected format: morphik://{token}@{host}
    const match = uri.match(/^morphik:\/\/[^@]+@(.+)/);
    if (!match || !match[1]) return defaultUrl; // Default if invalid format
    
    // Get the host part
    let host = match[1];
    
    // If it's local, localhost or 127.0.0.1, ensure http:// protocol and add port if needed
    if (host.includes('local') || host.includes('127.0.0.1')) {
      if (!host.includes('://')) {
        host = `http://${host}`;
      }
      // Add default port 8000 if no port specified
      if (!host.includes(':') || host.endsWith(':')) {
        host = `${host.replace(/:$/, '')}:8000`;
      }
    } else {
      // For other hosts, ensure https:// protocol
      if (!host.includes('://')) {
        host = `https://${host}`;
      }
    }
    
    console.log('Extracted API base URL:', host);
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
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  if (contentType) {
    headers['Content-Type'] = contentType;
  }
  
  return headers;
}
