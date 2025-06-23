"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Sidebar } from "@/components/ui/sidebar";
import DocumentsSection from "@/components/documents/DocumentsSection";
import SearchSection from "@/components/search/SearchSection";
import ChatSection from "@/components/chat/ChatSection";
import GraphSection from "@/components/GraphSection";
import WorkflowSection from "@/components/workflows/WorkflowSection";
import { ConnectorList } from "@/components/connectors/ConnectorList";
import { PDFViewer } from "@/components/pdf/PDFViewer";
import { PDFAPIService } from "@/components/pdf/PDFAPIService";
import { SettingsSection } from "@/components/settings/SettingsSection";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "@/lib/utils";
import { MorphikUIProps } from "./types";
import { cn } from "@/lib/utils";
import { setupLogging } from "@/lib/log";

// Default API base URL
const DEFAULT_API_BASE_URL = "http://localhost:8000";

// Disable excessive logging unless debug is enabled
setupLogging();

const MorphikUI: React.FC<MorphikUIProps> = ({
  connectionUri,
  apiBaseUrl = DEFAULT_API_BASE_URL,
  isReadOnlyUri = false, // Default to editable URI
  onUriChange,
  onBackClick,
  initialFolder = null,
  initialSection = "documents",
  onDocumentUpload,
  onDocumentDelete,
  onDocumentClick,
  onFolderCreate,
  onFolderClick,
  onSearchSubmit,
  onChatSubmit,
  onGraphClick,
  onGraphCreate,
  onGraphUpdate,
}) => {
  // State to manage connectionUri internally if needed
  const [currentUri, setCurrentUri] = useState(connectionUri);

  // Update internal state when prop changes
  useEffect(() => {
    setCurrentUri(connectionUri);
  }, [connectionUri]);

  // Valid section types, now matching the updated MorphikUIProps
  type SectionType = "documents" | "search" | "chat" | "graphs" | "workflows" | "connections" | "pdf" | "settings";

  useEffect(() => {
    // Ensure initialSection from props is a valid SectionType before setting
    setActiveSection(initialSection as SectionType);
  }, [initialSection]);

  // Handle URI changes from sidebar
  const handleUriChange = (newUri: string) => {
    console.log("MorphikUI: URI changed to:", newUri);
    setCurrentUri(newUri);
    onUriChange?.(newUri);
  };

  const [activeSection, setActiveSection] = useState<SectionType>(initialSection as SectionType);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [pdfViewerDocumentId, setPdfViewerDocumentId] = useState<string | undefined>(undefined);

  // Extract auth token and API URL from connection URI if provided
  const authToken = currentUri ? extractTokenFromUri(currentUri) : null;

  // Derive API base URL from the URI if provided
  const effectiveApiBaseUrl = getApiBaseUrlFromUri(currentUri ?? undefined, apiBaseUrl);

  // Generate session and user information for PDF viewer scoping
  const pdfSessionInfo = useMemo(() => {
    // Generate a unique session ID for this UI instance
    const sessionId = `ui-session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Try to extract user ID from auth token or use a fallback
    let userId = "anonymous";
    if (authToken) {
      try {
        // Try to decode JWT token to get user info (basic decode, not verification)
        const payload = JSON.parse(atob(authToken.split(".")[1]));
        userId = payload.sub || payload.user_id || payload.id || "authenticated";
      } catch (error) {
        console.error("MorphikUI: Error parsing auth token:", error);
        // If token parsing fails, use a generic authenticated user ID
        userId = "authenticated";
      }
    }

    return { sessionId, userId };
  }, [authToken]); // Regenerate if auth token changes

  // Log the effective API URL for debugging
  useEffect(() => {
    console.log("MorphikUI: Using API URL:", effectiveApiBaseUrl);
    console.log("MorphikUI: Auth token present:", !!authToken);
    console.log("MorphikUI: PDF session info:", pdfSessionInfo);
  }, [effectiveApiBaseUrl, authToken, pdfSessionInfo]);

  // Wrapper for section change to match expected type
  const handleSectionChange = (section: string) => {
    if (["documents", "search", "chat", "graphs", "workflows", "connections", "pdf", "settings"].includes(section)) {
      // Added "workflows"
      setActiveSection(section as SectionType); // Use SectionType here
    }
  };

  // Handle navigation to PDF viewer with specific document
  const handleViewInPDFViewer = (documentId: string) => {
    setPdfViewerDocumentId(documentId);
    setActiveSection("pdf");
  };

  // Clear PDF viewer document ID when switching away from PDF section
  useEffect(() => {
    if (activeSection !== "pdf") {
      setPdfViewerDocumentId(undefined);
    }
  }, [activeSection]);

  return (
    <PDFAPIService sessionId={pdfSessionInfo.sessionId} userId={pdfSessionInfo.userId}>
      <div className={cn("flex h-full w-full overflow-hidden")}>
        <Sidebar
          connectionUri={currentUri ?? undefined}
          isReadOnlyUri={isReadOnlyUri}
          onUriChange={handleUriChange}
          activeSection={activeSection}
          onSectionChange={handleSectionChange}
          isCollapsed={isSidebarCollapsed}
          setIsCollapsed={setIsSidebarCollapsed}
          onBackClick={onBackClick}
        />

        <main className="flex flex-1 flex-col overflow-hidden">
          {/* Render active section based on state */}
          {activeSection === "documents" && (
            <DocumentsSection
              key={`docs-${effectiveApiBaseUrl}-${initialFolder}`}
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
              initialFolder={initialFolder ?? undefined}
              onDocumentUpload={onDocumentUpload}
              onDocumentDelete={onDocumentDelete}
              onDocumentClick={onDocumentClick}
              onFolderCreate={onFolderCreate}
              onFolderClick={onFolderClick}
              onRefresh={undefined}
              onViewInPDFViewer={handleViewInPDFViewer}
            />
          )}
          {activeSection === "search" && (
            <SearchSection
              key={`search-${effectiveApiBaseUrl}`}
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
              onSearchSubmit={onSearchSubmit}
            />
          )}
          {activeSection === "chat" && (
            <ChatSection
              key={`chat-${effectiveApiBaseUrl}`}
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
              onChatSubmit={onChatSubmit}
            />
          )}
          {activeSection === "graphs" && (
            <GraphSection
              key={`graphs-${effectiveApiBaseUrl}`}
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
              onSelectGraph={onGraphClick}
              onGraphCreate={onGraphCreate}
              onGraphUpdate={onGraphUpdate}
            />
          )}
          {activeSection === "workflows" && (
            <WorkflowSection
              key={`workflows-${effectiveApiBaseUrl}`}
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
            />
          )}
          {activeSection === "connections" && (
            <div className="h-full overflow-auto p-4 md:p-6">
              {/* Wrapper div for consistent padding and full height */}
              <ConnectorList apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} />
            </div>
          )}
          {activeSection === "pdf" && (
            <PDFViewer
              key={`pdf-${effectiveApiBaseUrl}-${pdfViewerDocumentId}`}
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
              initialDocumentId={pdfViewerDocumentId}
            />
          )}
          {activeSection === "settings" && (
            <SettingsSection authToken={authToken} onBackClick={() => setActiveSection("chat")} />
          )}
        </main>
      </div>
    </PDFAPIService>
  );
};

export default MorphikUI;
