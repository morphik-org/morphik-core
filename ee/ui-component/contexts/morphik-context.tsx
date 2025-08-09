"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { parseConnectionUri, ConnectionInfo, isLocalUri } from "@/lib/connection-utils";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const CONNECTION_URI_STORAGE_KEY = "morphik-connection-uri";

interface MorphikContextType {
  connectionUri: string | null;
  connectionInfo: ConnectionInfo | null;
  authToken: string | null;
  apiBaseUrl: string;
  isLocal: boolean;
  isReadOnlyUri: boolean;
  updateConnectionUri: (uri: string) => void;
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
  onBackClick?: () => void;
}

const MorphikContext = createContext<MorphikContextType | undefined>(undefined);

// Helper function to safely access localStorage with migration
function getStoredConnectionUri(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = window.localStorage.getItem(CONNECTION_URI_STORAGE_KEY);

    // Clean up malformed URIs
    if (stored && stored.includes("://morphik://")) {
      // Remove the leading protocol from malformed URIs like https://morphik://...
      const cleaned = stored.replace(/^https?:\/\//, "");
      console.log("Cleaning malformed stored URI:", stored, "→", cleaned);
      window.localStorage.setItem(CONNECTION_URI_STORAGE_KEY, cleaned);
      return cleaned;
    }

    // Migration: If stored URI is in old morphik:// format for localhost, convert it
    if (stored && stored.startsWith("morphik://local@")) {
      // Extract the host part and return as plain host:port
      const match = stored.match(/^morphik:\/\/local@(.+)$/);
      if (match && match[1]) {
        const migratedUri = match[1];
        console.log("Migrating old URI format:", stored, "→", migratedUri);
        // Update storage with migrated value
        window.localStorage.setItem(CONNECTION_URI_STORAGE_KEY, migratedUri);
        return migratedUri;
      }
    }

    return stored;
  } catch {
    return null;
  }
}

function setStoredConnectionUri(uri: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (uri) {
      window.localStorage.setItem(CONNECTION_URI_STORAGE_KEY, uri);
    } else {
      window.localStorage.removeItem(CONNECTION_URI_STORAGE_KEY);
    }
  } catch {
    // Ignore localStorage errors
  }
}

export function MorphikProvider({
  children,
  initialConnectionUri = null,
  isReadOnlyUri = false,
  connectionUri: externalConnectionUri,
  onBackClick,
  userProfile,
  onLogout,
  onProfileNavigate,
  onUpgradeClick,
}: {
  children: React.ReactNode;
  initialConnectionUri?: string | null;
  isReadOnlyUri?: boolean;
  connectionUri?: string | null;
  onBackClick?: () => void;
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
}) {
  const [connectionUri, setConnectionUri] = useState<string | null>(() => {
    const storedUri = getStoredConnectionUri();

    // Clear stored URI if it's a local connection (on app restart)
    if (storedUri && isLocalUri(storedUri)) {
      console.log("Clearing stored local connection URI on app restart:", storedUri);
      setStoredConnectionUri(null);
      // Use initial value or external prop for local connections
      return externalConnectionUri || initialConnectionUri;
    }

    // Priority: external prop > stored value > initial value
    return externalConnectionUri || storedUri || initialConnectionUri;
  });

  const connectionInfo = React.useMemo(() => {
    if (!connectionUri) return null;
    return parseConnectionUri(connectionUri);
  }, [connectionUri]);

  const authToken = connectionInfo?.authToken || null;

  // Ensure apiBaseUrl is always a valid HTTP(S) URL
  const apiBaseUrl = React.useMemo(() => {
    if (!connectionInfo?.apiBaseUrl) {
      return DEFAULT_API_BASE_URL;
    }

    let url = connectionInfo.apiBaseUrl.trim();

    // Safety check: ensure it's a proper HTTP(S) URL
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      console.error("[MorphikContext] Invalid apiBaseUrl:", url);
      return DEFAULT_API_BASE_URL;
    }

    // Force HTTPS for morphik.ai domains to avoid CORS preflight redirect failures
    try {
      const parsed = new URL(url);
      if (parsed.protocol === "http:" && parsed.hostname.endsWith("morphik.ai")) {
        parsed.protocol = "https:";
        url = parsed.toString().replace(/\/$/, "");
      }
    } catch {
      // ignore URL parse errors; fallback to original url
    }

    return url;
  }, [connectionInfo]);

  const isLocal = connectionInfo?.type === "local" || !connectionInfo;

  // Effect to persist connectionUri changes to localStorage
  useEffect(() => {
    // Only store non-local URIs persistently
    if (connectionUri && connectionInfo && connectionInfo.type === "cloud") {
      setStoredConnectionUri(connectionUri);
    } else if (connectionUri && connectionInfo && connectionInfo.type === "local") {
      // For local connections, we can optionally store temporarily
      // but it will be cleared on next app restart
      console.log("Local connection - will be cleared on restart:", connectionUri);
      setStoredConnectionUri(connectionUri);
    } else {
      // Clear storage if no URI
      setStoredConnectionUri(null);
    }
  }, [connectionUri, connectionInfo]);

  const updateConnectionUri = (uri: string) => {
    if (!isReadOnlyUri) {
      console.log("[MorphikContext] updateConnectionUri:", uri);
      setConnectionUri(uri);
    }
  };

  // Debug log when values change
  React.useEffect(() => {
    console.log("[MorphikContext] Current values:", {
      connectionUri,
      connectionInfo,
      apiBaseUrl,
      authToken,
      isLocal,
    });
  }, [connectionUri, connectionInfo, apiBaseUrl, authToken, isLocal]);

  return (
    <MorphikContext.Provider
      value={{
        connectionUri,
        connectionInfo,
        authToken,
        apiBaseUrl,
        isLocal,
        isReadOnlyUri,
        updateConnectionUri,
        userProfile,
        onLogout,
        onProfileNavigate,
        onUpgradeClick,
        onBackClick,
      }}
    >
      {children}
    </MorphikContext.Provider>
  );
}

export function useMorphik() {
  const context = useContext(MorphikContext);
  if (context === undefined) {
    throw new Error("useMorphik must be used within a MorphikProvider");
  }
  return context;
}
