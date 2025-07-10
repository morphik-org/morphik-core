"use client";

import React, { createContext, useContext, useState } from "react";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "@/lib/utils";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

interface MorphikContextType {
  connectionUri: string | null;
  authToken: string | null;
  apiBaseUrl: string;
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
  onBackClick?: () => void;
}

const MorphikContext = createContext<MorphikContextType | undefined>(undefined);

export function MorphikProvider({
  children,
  initialConnectionUri = null,
  isReadOnlyUri = false,
  connectionUri: externalConnectionUri,
  onBackClick,
  userProfile,
  onLogout,
  onProfileNavigate,
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
}) {
  const [connectionUri, setConnectionUri] = useState<string | null>(externalConnectionUri || initialConnectionUri);

  const authToken = connectionUri ? extractTokenFromUri(connectionUri) : null;
  const apiBaseUrl = connectionUri ? getApiBaseUrlFromUri(connectionUri) : DEFAULT_API_BASE_URL;

  const updateConnectionUri = (uri: string) => {
    if (!isReadOnlyUri) {
      setConnectionUri(uri);
    }
  };

  return (
    <MorphikContext.Provider
      value={{
        connectionUri,
        authToken,
        apiBaseUrl,
        isReadOnlyUri,
        updateConnectionUri,
        userProfile,
        onLogout,
        onProfileNavigate,
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
