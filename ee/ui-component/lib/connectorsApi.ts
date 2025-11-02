// Defines the expected shape of the authentication status from the backend
export interface ConnectorAuthStatus {
  is_authenticated: boolean;
  message?: string;
  auth_url?: string;
}

// Interface for manual credential fields
export interface CredentialField {
  name: string;
  label: string;
  description: string;
  type: "text" | "password" | "select";
  required: boolean;
  options?: Array<{ value: string; label: string }>;
}

// Interface for manual credentials auth response
export interface ManualAuthResponse {
  auth_type: "manual_credentials";
  required_fields: CredentialField[];
  instructions?: string;
}

// Interface for OAuth auth response
export interface OAuthAuthResponse {
  authorization_url: string;
}

// Union type for auth responses
export type AuthResponse = ManualAuthResponse | OAuthAuthResponse;

// Fetches the authentication status for a given connector type
export async function getConnectorAuthStatus(
  apiBaseUrl: string,
  connectorType: string,
  authToken: string | null
): Promise<ConnectorAuthStatus> {
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/auth_status`, {
    headers: {
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    },
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `Failed to fetch auth status for ${connectorType}: ${response.status} ${response.statusText} - ${errorData.detail || ""}`
    );
  }
  return response.json();
}

// Updated function to handle both OAuth and manual credentials
export async function initiateConnectorAuth(
  apiBaseUrl: string,
  connectorType: string,
  appRedirectUri: string,
  authToken: string | null
): Promise<AuthResponse> {
  // Use the *initiate_url* helper which returns a JSON payload containing either
  // the provider's authorization_url (OAuth) or credential form specification (manual).

  const helperUrl = new URL(`${apiBaseUrl}/ee/connectors/${connectorType}/auth/initiate_url`);
  helperUrl.searchParams.append("app_redirect_uri", appRedirectUri);

  try {
    const resp = await fetch(helperUrl.toString(), {
      method: "GET",
      headers: {
        ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      },
      credentials: "include",
    });

    if (!resp.ok) {
      let detail = "";
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const errBody = await resp.json();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        detail = (errBody as any)?.detail || "";
      } catch {
        /* ignore */
      }
      throw new Error(`Failed to initiate auth flow: ${resp.status} ${resp.statusText} ${detail}`);
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const data = await resp.json();

    // Check if this is a manual credentials flow or OAuth
    if (data.auth_type === "manual_credentials") {
      return data as ManualAuthResponse;
    } else {
      // OAuth flow - redirect to authorization URL
      if (!data.authorization_url) {
        throw new Error("Backend did not return authorization_url");
      }
      window.location.href = data.authorization_url;
      return data as OAuthAuthResponse;
    }
  } catch (err) {
    console.error("Error initiating connector auth:", err);
    throw err;
  }
}

// New function to submit manual credentials
export async function submitManualCredentials(
  apiBaseUrl: string,
  connectorType: string,
  credentials: Record<string, string>,
  authToken: string | null
): Promise<{ status: string; message: string }> {
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/auth/finalize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    },
    body: JSON.stringify({ credentials }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `Failed to submit credentials for ${connectorType}: ${response.status} ${response.statusText} - ${errorData.detail || ""}`
    );
  }

  return response.json();
}

// Disconnects a connector for the current user
export async function disconnectConnector(
  apiBaseUrl: string,
  connectorType: string,
  authToken: string | null
): Promise<void> {
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/disconnect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
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
  authToken: string | null,
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
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
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
  morphikFolderName?: string;
}

export async function ingestConnectorFile(
  apiBaseUrl: string,
  connectorType: string,
  authToken: string | null,
  fileId: string,
  options: IngestionOptions = {}
): Promise<Record<string, unknown>> {
  const response = await fetch(`${apiBaseUrl}/ee/connectors/${connectorType}/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    },
    body: JSON.stringify({
      file_id: fileId,
      ...(options.metadata && { metadata: options.metadata }),
      ...(options.morphikFolderName && { morphik_folder_name: options.morphikFolderName }),
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Failed to ingest file via connector." }));
    throw new Error(errorData.detail || "Failed to ingest file via connector.");
  }
  return response.json(); // Contains morphik_document_id and status_path
}
