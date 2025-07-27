"use client";

import React, { useState, useCallback, useEffect } from "react";
import { MorphikUIProps } from "./types";
import DocumentsWithHeader from "@/components/documents/DocumentsWithHeader";
import SearchSection from "@/components/search/SearchSection";
import ChatSection from "@/components/chat/ChatSection";
import GraphSection from "@/components/GraphSection";
import WorkflowSection from "@/components/workflows/WorkflowSection";
import LogsSection from "@/components/logs/LogsSection";
import { ConnectorList } from "@/components/connectors/ConnectorList";
import { PDFViewer } from "@/components/pdf/PDFViewer";
import { SettingsSection } from "@/components/settings/SettingsSection";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "@/lib/utils";
import { PDFAPIService } from "@/components/pdf/PDFAPIService";
import { MorphikSidebarStateful } from "@/components/morphik-sidebar-stateful";
import { DynamicSiteHeader } from "@/components/dynamic-site-header";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar-new";
import { MorphikProvider } from "@/contexts/morphik-context";
import { HeaderProvider } from "@/contexts/header-context";
import { AlertSystem } from "@/components/ui/alert-system";
import { ThemeProvider } from "@/components/theme-provider";
import { useRouter, usePathname } from "next/navigation";

/**
 * MorphikUI Component
 *
 * Full dashboard component with sidebar navigation and header.
 * This includes the complete UI chrome for a standalone experience.
 */
const MorphikUI: React.FC<MorphikUIProps> = props => {
  const {
    connectionUri,
    apiBaseUrl = typeof window !== 'undefined' 
      ? `${window.location.protocol}//${window.location.hostname}:8000`
      : "http://localhost:8000",
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
  } = props;

  const [currentSection, setCurrentSection] = useState(initialSection);
  const [currentFolder, setCurrentFolder] = useState<string | null>(initialFolder);

  const authToken = connectionUri ? extractTokenFromUri(connectionUri) : null;
  const effectiveApiBaseUrl = getApiBaseUrlFromUri(connectionUri ?? undefined, apiBaseUrl);

  // For PDF viewer session info
  const sessionId = `ui-session-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  const userId = authToken ? "authenticated" : "anonymous";

  // Local breadcrumbs managed here when section is not documents
  const [localBreadcrumbs, setLocalBreadcrumbs] = useState<
    { label: string; href?: string; onClick?: (e: React.MouseEvent) => void }[] | undefined
  >();

  // update breadcrumbs whenever section changes (initial and subsequent)
  useEffect(() => {
    if (currentSection === "documents") {
      setLocalBreadcrumbs(undefined);
      return;
    }

    const prettyLabel =
      currentSection === "graphs"
        ? "Knowledge Graphs"
        : currentSection.charAt(0).toUpperCase() + currentSection.slice(1);

    setLocalBreadcrumbs([
      {
        label: "Home",
        onClick: () => setCurrentSection("documents" as typeof initialSection),
      },
      { label: prettyLabel },
    ]);
  }, [currentSection]);

  // sync prop changes from layout routing
  useEffect(() => {
    setCurrentSection(initialSection);
  }, [initialSection]);

  const router = useRouter();
  const pathname = usePathname() || "/";

  const handleSectionChange = useCallback(
    (section: string) => {
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

  const contentInner = (
    <PDFAPIService sessionId={sessionId} userId={userId}>
      <div className="min-h-screen bg-sidebar">
        <MorphikProvider
          connectionUri={connectionUri}
          onBackClick={onBackClick}
          userProfile={userProfile}
          onLogout={onLogout}
          onProfileNavigate={onProfileNavigate}
          onUpgradeClick={onUpgradeClick}
        >
          <HeaderProvider>
            <SidebarProvider
              style={
                {
                  "--sidebar-width": "calc(var(--spacing) * 72)",
                  "--header-height": "calc(var(--spacing) * 12)",
                } as React.CSSProperties
              }
            >
              <MorphikSidebarStateful
                variant="inset"
                currentSection={currentSection}
                onSectionChange={handleSectionChange}
                userProfile={userProfile}
                onLogout={onLogout}
                onProfileNavigate={onProfileNavigate}
                onUpgradeClick={onUpgradeClick}
                logoLight={logoLight}
                logoDark={logoDark}
              />
              <SidebarInset>
                <DynamicSiteHeader userProfile={userProfile} customBreadcrumbs={localBreadcrumbs} />
                <div className="flex flex-1 flex-col p-4 md:p-6">{renderSection()}</div>
              </SidebarInset>
            </SidebarProvider>
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
