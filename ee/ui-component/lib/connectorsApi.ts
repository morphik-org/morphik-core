// Defines the expected shape of the authentication status from the backend
export interface ConnectorAuthStatus {
  is_authenticated: boolean;
  message?: string;
  auth_url?: string;
}

// Fetches the authentication status for a given connector type
export async function getConnectorAuthStatus(apiBaseUrl: string, connectorType: string): Promise<ConnectorAuthStatus> {
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/auth_status`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `Failed to fetch auth status for ${connectorType}: ${response.status} ${response.statusText} - ${errorData.detail || ""}`
    );
  }
  return response.json();
}

// Initiates the authentication process by redirecting the user
// The backend will handle the actual redirect to the OAuth provider
export function initiateConnectorAuth(apiBaseUrl: string, connectorType: string, appRedirectUri: string): void {
  // The backend /auth/initiate endpoint itself performs a redirect.
  // So, navigating to it will trigger the OAuth flow.
  // We add the app_redirect_uri for the backend to use after successful callback.
  const initiateUrl = new URL(`${apiBaseUrl}/ee/connectors/${connectorType}/auth/initiate`);
  initiateUrl.searchParams.append("app_redirect_uri", appRedirectUri);
  window.location.href = initiateUrl.toString();
}

// Disconnects a connector for the current user
export async function disconnectConnector(apiBaseUrl: string, connectorType: string): Promise<void> {
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/disconnect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `Failed to disconnect ${connectorType}: ${response.status} ${response.statusText} - ${errorData.detail || ""}`
    );
  }
  // Optionally, you can return response.json() if the backend sends a meaningful body
}

// Interface for ConnectorFile to be used in the frontend
// This should match the Pydantic model in base_connector.py
export interface ConnectorFile {
  id: string;
  name: string;
  is_folder: boolean;
  mime_type?: string;
  size?: number; // in bytes
  modified_date?: string; // ISO 8601 string
}

export async function listConnectorFiles(
  apiBaseUrl: string,
  connectorType: string,
  path: string | null,
  pageToken?: string,
  q_filter?: string,
  pageSize: number = 20
): Promise<{ files: ConnectorFile[]; next_page_token?: string }> {
  const params = new URLSearchParams();
  if (path && path !== "root") {
    // Google Drive uses 'root' for the main directory, often not needed as explicit param if backend defaults to root
    params.append("path", path);
  }
  if (pageToken) {
    params.append("page_token", pageToken);
  }
  if (q_filter) {
    params.append("q_filter", q_filter);
  }
  params.append("page_size", pageSize.toString());

  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/files?${params.toString()}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      // TODO: Add authorization headers if required by your API setup
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Failed to list files from connector." }));
    throw new Error(errorData.detail || "Failed to list files from connector.");
  }
  return response.json();
}

// Define the structure for ingestion options
interface IngestionOptions {
  metadata?: Record<string, unknown>;
  rules?: unknown[]; // Consider a more specific type for rules if available
  morphikFolderName?: string | null;
  morphikEndUserId?: string | null;
}

export async function ingestConnectorFile(
  apiBaseUrl: string,
  connectorType: string,
  fileId: string,
  // Replace displayName with ingestionOptions
  options?: IngestionOptions
): Promise<Record<string, unknown>> {
  // Define a more specific return type based on your API response for ingestion
  const body = {
    file_id: fileId,
    // Spread options to include metadata and rules if provided
    // The backend IngestFromConnectorRequest will need to be updated to accept these
    ...(options?.metadata && { metadata: options.metadata }),
    ...(options?.rules && { rules: options.rules }),
    morphik_folder_name: options?.morphikFolderName,
    morphik_end_user_id: options?.morphikEndUserId,
  };
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // TODO: Add authorization headers if required
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Failed to ingest file via connector." }));
    throw new Error(errorData.detail || "Failed to ingest file via connector.");
  }
  return response.json(); // Contains morphik_document_id and status_path
}
