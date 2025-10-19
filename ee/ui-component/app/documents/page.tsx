"use client";

export const dynamic = "force-dynamic";

import { useEffect, useState, Suspense } from "react";
import DocumentsSection from "@/components/documents/DocumentsSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useRouter, useSearchParams } from "next/navigation";
import { useHeader } from "@/contexts/header-context";
import { Button } from "@/components/ui/button";
import { Trash2, Upload, RefreshCw, PlusCircle, ChevronsDown, ChevronsUp } from "lucide-react";

function DocumentsContent() {
  const { apiBaseUrl, authToken } = useMorphik();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setCustomBreadcrumbs, setRightContent } = useHeader();

  const folderParam = searchParams?.get("folder") || null;
  const [currentFolder, setCurrentFolder] = useState<string | null>(folderParam);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [allFoldersExpanded, setAllFoldersExpanded] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [showUploadDialog, setShowUploadDialog] = useState(false);

  // Sync folder state with URL param changes
  useEffect(() => {
    setCurrentFolder(folderParam);
  }, [folderParam]);

  // Update header breadcrumbs and controls when folder changes
  useEffect(() => {
    const breadcrumbs = [
      { label: "Home", href: "/" },
      {
        label: "Documents",
        ...(currentFolder
          ? {
              href: "/documents",
            }
          : {}),
      },
      ...(currentFolder
        ? [
            {
              label: currentFolder === "all" ? "All Documents" : currentFolder,
            },
          ]
        : []),
    ];

    setCustomBreadcrumbs(breadcrumbs);

    // Set right content based on current view
    const rightContent = currentFolder ? (
      // Folder view controls
      <>
        {selectedDocuments.length > 0 && (
          <Button
            variant="outline"
            size="icon"
            onClick={() => {
              const event = new CustomEvent("deleteMultipleDocuments");
              window.dispatchEvent(event);
            }}
            className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
            title={`Delete ${selectedDocuments.length} selected document${selectedDocuments.length > 1 ? "s" : ""}`}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}

        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            // Trigger refresh event
            window.location.reload();
          }}
          title="Refresh documents"
        >
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>

        <Button
          variant="default"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openUploadDialog");
            window.dispatchEvent(event);
          }}
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </>
    ) : (
      // Root level controls
      <>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openNewFolderDialog");
            window.dispatchEvent(event);
          }}
        >
          <PlusCircle className="mr-2 h-4 w-4" />
          New Folder
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("toggleExpandAllFolders");
            window.dispatchEvent(event);
          }}
          className="flex items-center gap-1.5"
          title="Expand or collapse all folders"
        >
          {allFoldersExpanded ? <ChevronsUp className="h-4 w-4" /> : <ChevronsDown className="h-4 w-4" />}
          <span>{allFoldersExpanded ? "Collapse All" : "Expand All"}</span>
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            // Trigger refresh event
            window.location.reload();
          }}
          title="Refresh documents"
        >
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>
        <Button
          variant="default"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openUploadDialog");
            window.dispatchEvent(event);
          }}
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </>
    );

    setRightContent(rightContent);

    return () => {
      setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [currentFolder, router, selectedDocuments, allFoldersExpanded, setCustomBreadcrumbs, setRightContent]);

  // Listen for events from DocumentsSection
  useEffect(() => {
    const handleSelectionChange = (event: CustomEvent<{ selectedDocuments?: string[] }>) => {
      setSelectedDocuments(event.detail?.selectedDocuments || []);
    };

    const handleOpenNewFolderDialog = () => {
      setShowNewFolderDialog(true);
    };

    const handleToggleExpandAllFolders = () => {
      setAllFoldersExpanded(prev => !prev);
    };

    const handleOpenUploadDialog = () => {
      setShowUploadDialog(true);
    };

    window.addEventListener("documentsSelectionChanged", handleSelectionChange as EventListener);
    window.addEventListener("openNewFolderDialog", handleOpenNewFolderDialog);
    window.addEventListener("toggleExpandAllFolders", handleToggleExpandAllFolders);
    window.addEventListener("openUploadDialog", handleOpenUploadDialog);

    return () => {
      window.removeEventListener("documentsSelectionChanged", handleSelectionChange as EventListener);
      window.removeEventListener("openNewFolderDialog", handleOpenNewFolderDialog);
      window.removeEventListener("toggleExpandAllFolders", handleToggleExpandAllFolders);
      window.removeEventListener("openUploadDialog", handleOpenUploadDialog);
    };
  }, []);

  // Handle folder navigation
  const handleFolderClick = (folderName: string | null) => {
    setCurrentFolder(folderName);
    if (folderName) {
      router.push(`/documents?folder=${encodeURIComponent(folderName)}`);
    } else {
      router.push("/documents");
    }
  };

  return (
    <DocumentsSection
      apiBaseUrl={apiBaseUrl}
      authToken={authToken}
      initialFolder={folderParam}
      onDocumentUpload={undefined}
      onDocumentDelete={undefined}
      onDocumentClick={undefined}
      onFolderCreate={undefined}
      onFolderClick={handleFolderClick}
      onRefresh={undefined}
      onViewInPDFViewer={(documentId: string) => {
        router.push(`/pdf?document=${documentId}`);
      }}
      allFoldersExpanded={allFoldersExpanded}
      showNewFolderDialog={showNewFolderDialog}
      setShowNewFolderDialog={setShowNewFolderDialog}
      showUploadDialog={showUploadDialog}
      setShowUploadDialog={setShowUploadDialog}
    />
  );
}

export default function DocumentsPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <DocumentsContent />
    </Suspense>
  );
}
