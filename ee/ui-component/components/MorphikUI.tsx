"use client";

import React, { useState, useEffect } from "react";
import { Sidebar } from "@/components/ui/sidebar";
import DocumentsSection from "@/components/documents/DocumentsSection";
import SearchSection from "@/components/search/SearchSection";
import ChatSection from "@/components/chat/ChatSection";
import AgentChatSection from "@/components/chat/AgentChatSection";
import GraphSection from "@/components/GraphSection";
import { AlertSystem } from "@/components/ui/alert-system";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "@/lib/utils";
import { MorphikUIProps } from "./types";
import { cn } from "@/lib/utils";

// Default API base URL
const DEFAULT_API_BASE_URL = "http://localhost:8000";

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
  onAgentSubmit,
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

  useEffect(() => {
    setActiveSection(initialSection);
  }, [initialSection]);

  // Handle URI changes from sidebar
  const handleUriChange = (newUri: string) => {
    console.log("MorphikUI: URI changed to:", newUri);
    setCurrentUri(newUri);
    onUriChange?.(newUri);
  };

  const [activeSection, setActiveSection] = useState(initialSection);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Extract auth token and API URL from connection URI if provided
  const authToken = currentUri ? extractTokenFromUri(currentUri) : null;

  // Derive API base URL from the URI if provided
  const effectiveApiBaseUrl = getApiBaseUrlFromUri(currentUri ?? undefined, apiBaseUrl);

  // Log the effective API URL for debugging
  useEffect(() => {
    console.log("MorphikUI: Using API URL:", effectiveApiBaseUrl);
    console.log("MorphikUI: Auth token present:", !!authToken);
  }, [effectiveApiBaseUrl, authToken]);

  // Wrapper for section change to match expected type
  const handleSectionChange = (section: string) => {
    if (["documents", "search", "chat", "graphs", "agent"].includes(section)) {
      setActiveSection(section as "documents" | "search" | "chat" | "graphs" | "agent");
    }
  };

  return (
    <div className={cn("flex h-screen w-full overflow-hidden")}>
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
            setSidebarCollapsed={setIsSidebarCollapsed}
            onDocumentUpload={onDocumentUpload}
            onDocumentDelete={onDocumentDelete}
            onDocumentClick={onDocumentClick}
            onFolderCreate={onFolderCreate}
            onFolderClick={onFolderClick}
            onRefresh={undefined}
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
        {activeSection === "agent" && (
          <AgentChatSection
            key={`agent-${effectiveApiBaseUrl}`}
            apiBaseUrl={effectiveApiBaseUrl}
            authToken={authToken}
            onAgentSubmit={onAgentSubmit}
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
      </main>

      {/* Global alert system - integrated directly in the component */}
      <AlertSystem position="bottom-right" />
    </div>
  );
};

export default MorphikUI;
