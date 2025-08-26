"use client";

import React, { useState, useCallback, useEffect } from "react";
import { MorphikUIProps } from "./types";
import DocumentsWithHeader from "@/components/documents/DocumentsWithHeader";
import DocumentsPageV2 from "@/components/documents-v2/DocumentsPageV2";
import SearchSection from "@/components/search/SearchSection";
import ChatSection from "@/components/chat/ChatSection";
import GraphSection from "@/components/GraphSection";
import WorkflowSection from "@/components/workflows/WorkflowSection";
import LogsSection from "@/components/logs/LogsSection";
import { ConnectorList } from "@/components/connectors/ConnectorList";
import { PDFViewer } from "@/components/pdf/PDFViewer";
import { SettingsSection } from "@/components/settings/SettingsSection";
import { Toaster } from "@/components/ui/sonner";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "@/lib/utils";
import { PDFAPIService } from "@/components/pdf/PDFAPIService";
import { MorphikSidebarRemote } from "@/components/sidebar-stateful";
import { useChatContext } from "@/components/chat/chat-context";
import { DynamicSiteHeader } from "@/components/dynamic-site-header";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar-components";
import { MorphikProvider } from "@/contexts/morphik-context";
import { HeaderProvider } from "@/contexts/header-context";
import { AlertSystem } from "@/components/ui/alert-system";
import { ThemeProvider } from "@/components/theme-provider";
import { useRouter, usePathname } from "next/navigation";
import { ChatProvider } from "@/components/chat/chat-context";

/**
 * MorphikUI Component
 *
 * Full dashboard component with sidebar navigation and header.
 * This includes the complete UI chrome for a standalone experience.
 */
const MorphikUI: React.FC<MorphikUIProps> = props => {
  const {
    connectionUri: initialConnectionUri,
    apiBaseUrl = "http://localhost:8000",
    initialSection = "documents",
    initialFolder = null,
    onBackClick,
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
    userProfile,
    onLogout,
    onProfileNavigate,
    onUpgradeClick,
    logoLight = "/morphikblack.png",
    logoDark = "/morphikwhite.png",
    breadcrumbItems,
  } = props;

  const [currentSection, setCurrentSection] = useState(initialSection);
  const [currentFolder, setCurrentFolder] = useState<string | null>(initialFolder);
  const [showChatView, setShowChatView] = useState(false);
  const [showSettingsView, setShowSettingsView] = useState(false);
  const connectionUri = initialConnectionUri;

  const router = useRouter();
  const pathname = usePathname() || "/";

  // Handle chat view changes
  const handleChatViewChange = useCallback(
    (show: boolean) => {
      setShowChatView(show);
      // If hiding chat view while on chat section, go back to documents
      if (!show && currentSection === "chat") {
        setCurrentSection("documents");
        // Also update the URL to reflect the section change
        const segments = pathname.split("/").filter(Boolean);
        if (segments.length > 0) {
          const appId = segments[0];
          router.push(`/${appId}/documents`);
        }
      }
    },
    [currentSection, pathname, router]
  );

  const authToken = connectionUri ? extractTokenFromUri(connectionUri) : null;
  const effectiveApiBaseUrl = getApiBaseUrlFromUri(connectionUri ?? undefined, apiBaseUrl);

  // For PDF viewer session info
  const sessionId = `ui-session-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  const userId = authToken ? "authenticated" : "anonymous";

  // sync prop changes from layout routing
  useEffect(() => {
    setCurrentSection(initialSection);
  }, [initialSection]);

  // Sync overlays with section (ensures leaving chat/settings hides their side panels)
  useEffect(() => {
    setShowChatView(currentSection === "chat");
    setShowSettingsView(currentSection === "settings");
  }, [currentSection]);

  const handleSectionChange = useCallback(
    (section: string) => {
      // Keep overlays consistent with section
      setShowChatView(section === "chat");
      setShowSettingsView(section === "settings");
      setCurrentSection(section as typeof initialSection);

      // --- update browser URL so Cloud mirrors standalone behaviour ----
      const segments = pathname.split("/").filter(Boolean);
      if (segments.length > 0) {
        const appId = segments[0];
        const newPath = `/${appId}/${section === "documents" ? "documents" : section}`;
        router.push(newPath);
      }
    },
    [router, pathname]
  );

  const handleFolderChange = useCallback(
    (folderName: string | null) => {
      setCurrentFolder(folderName);
      onFolderClick?.(folderName);
    },
    [onFolderClick]
  );

  const renderSection = () => {
    switch (currentSection) {
      case "documents":
        return (
          <DocumentsWithHeader
            apiBaseUrl={effectiveApiBaseUrl}
            authToken={authToken}
            initialFolder={currentFolder}
            onDocumentUpload={onDocumentUpload}
            onDocumentDelete={onDocumentDelete}
            onDocumentClick={onDocumentClick}
            onFolderCreate={onFolderCreate}
            onFolderClick={handleFolderChange}
          />
        );
      case "documents-v2":
        return (
          <DocumentsPageV2
            apiBaseUrl={effectiveApiBaseUrl}
            authToken={authToken}
            initialFolder={currentFolder}
            onDocumentUpload={onDocumentUpload}
            onDocumentDelete={onDocumentDelete}
            onDocumentClick={onDocumentClick}
            onFolderCreate={onFolderCreate}
            onFolderClick={handleFolderChange}
            onViewInPDFViewer={documentId => {
              // For now, just log - we'll implement this later
              console.log("Open PDF viewer for document:", documentId);
            }}
          />
        );
      case "search":
        return <SearchSection apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} onSearchSubmit={onSearchSubmit} />;
      case "chat":
        return <ChatSection apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} onChatSubmit={onChatSubmit} />;
      case "graphs":
        return (
          <GraphSection
            apiBaseUrl={effectiveApiBaseUrl}
            authToken={authToken}
            onSelectGraph={onGraphClick}
            onGraphCreate={onGraphCreate}
            onGraphUpdate={onGraphUpdate}
          />
        );
      case "workflows":
        return <WorkflowSection apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} />;
      case "connections":
        return (
          <div className="h-full overflow-auto p-4 md:p-6">
            <ConnectorList apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} />
          </div>
        );
      case "pdf":
        return <PDFViewer apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} />;
      case "settings":
        return <SettingsSection authToken={authToken} onBackClick={onBackClick} />;
      case "logs":
        return <LogsSection apiBaseUrl={effectiveApiBaseUrl} authToken={authToken} />;
      default:
        return (
          <div className="flex h-full items-center justify-center">
            <p className="text-muted-foreground">Unknown section: {initialSection}</p>
          </div>
        );
    }
  };

  // Local wrapper to bridge ChatContext into the Sidebar props
  const SidebarWithChatContext: React.FC<{
    logoLight?: string;
    logoDark?: string;
    showChatView: boolean;
    onChatViewChange: (show: boolean) => void;
    showSettingsView: boolean;
    onSettingsViewChange: (show: boolean) => void;
    currentSection: string;
    onSectionChange: (section: string) => void;
    userProfile?: typeof userProfile;
    onLogout?: typeof onLogout;
    onProfileNavigate?: typeof onProfileNavigate;
    onUpgradeClick?: typeof onUpgradeClick;
  }> = ({
    logoLight,
    logoDark,
    showChatView,
    onChatViewChange,
    showSettingsView,
    onSettingsViewChange,
    currentSection,
    onSectionChange,
    userProfile,
    onLogout,
    onProfileNavigate,
    onUpgradeClick,
  }) => {
    const { activeChatId, setActiveChatId, activeSettingsTab, setActiveSettingsTab } = useChatContext();

    return (
      <MorphikSidebarRemote
        currentSection={currentSection}
        onSectionChange={onSectionChange}
        userProfile={userProfile}
        onLogout={onLogout}
        onProfileNavigate={onProfileNavigate}
        onUpgradeClick={onUpgradeClick}
        logoLight={logoLight}
        logoDark={logoDark}
        showChatView={showChatView}
        onChatViewChange={onChatViewChange}
        activeChatId={activeChatId}
        onChatSelect={setActiveChatId}
        showSettingsView={showSettingsView}
        onSettingsViewChange={onSettingsViewChange}
        activeSettingsTab={activeSettingsTab}
        onSettingsTabChange={setActiveSettingsTab}
      />
    );
  };

  const contentInner = (
    <PDFAPIService sessionId={sessionId} userId={userId}>
      <div className="min-h-screen bg-sidebar">
        <Toaster position="top-right" richColors />
        <MorphikProvider
          connectionUri={connectionUri}
          onBackClick={onBackClick}
          userProfile={userProfile}
          onLogout={onLogout}
          onProfileNavigate={onProfileNavigate}
          onUpgradeClick={onUpgradeClick}
        >
          <HeaderProvider>
            <ChatProvider>
              <SidebarProvider
                style={
                  {
                    "--sidebar-width": "calc(var(--spacing) * 72)",
                    "--header-height": "calc(var(--spacing) * 12)",
                  } as React.CSSProperties
                }
              >
                <SidebarWithChatContext
                  currentSection={currentSection}
                  onSectionChange={handleSectionChange}
                  userProfile={userProfile}
                  onLogout={onLogout}
                  onProfileNavigate={onProfileNavigate}
                  onUpgradeClick={onUpgradeClick}
                  logoLight={logoLight}
                  logoDark={logoDark}
                  showChatView={showChatView}
                  onChatViewChange={handleChatViewChange}
                  showSettingsView={showSettingsView}
                  onSettingsViewChange={setShowSettingsView}
                />
                <SidebarInset>
                  <DynamicSiteHeader userProfile={userProfile} customBreadcrumbs={breadcrumbItems} />
                  <div className="flex flex-1 flex-col p-4 md:p-6">{renderSection()}</div>
                </SidebarInset>
              </SidebarProvider>
            </ChatProvider>
          </HeaderProvider>
        </MorphikProvider>
      </div>
      <AlertSystem position="bottom-right" />
    </PDFAPIService>
  );

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      {contentInner}
    </ThemeProvider>
  );
};

export default MorphikUI;
