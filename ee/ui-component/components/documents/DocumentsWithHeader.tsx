"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { useHeader } from "@/contexts/header-context";
import { Button } from "@/components/ui/button";
import { Trash2, Upload, RefreshCw, PlusCircle, ChevronsDown, ChevronsUp } from "lucide-react";
import DocumentsSection from "./DocumentsSection";

interface DocumentsWithHeaderProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialFolder?: string | null;
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderClick?: (folderName: string | null) => void;
  onFolderCreate?: (folderName: string) => void;
  onRefresh?: () => void;
  onViewInPDFViewer?: (documentId: string) => void;
}

export default function DocumentsWithHeader(props: DocumentsWithHeaderProps) {
  const { setCustomBreadcrumbs, setRightContent } = useHeader();
  const [selectedFolder, setSelectedFolder] = useState<string | null>(props.initialFolder || null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [allFoldersExpanded, setAllFoldersExpanded] = useState(false);

  // Create a ref to access DocumentsSection methods
  const documentsSectionRef = useRef<{
    handleRefresh: () => void;
    handleDeleteMultipleDocuments: () => void;
    selectedDocuments: string[];
  } | null>(null);

  // Handle folder changes from DocumentsSection
  const handleFolderClick = useCallback(
    (folderName: string | null) => {
      setSelectedFolder(folderName);
      props.onFolderClick?.(folderName);
    },
    [props]
  );

  // Handle refresh
  const handleRefresh = useCallback(() => {
    if (documentsSectionRef.current?.handleRefresh) {
      documentsSectionRef.current.handleRefresh();
    }
    props.onRefresh?.();
  }, [props]);

  // Handle delete multiple
  const handleDeleteMultiple = useCallback(() => {
    if (documentsSectionRef.current?.handleDeleteMultipleDocuments) {
      documentsSectionRef.current.handleDeleteMultipleDocuments();
    }
  }, []);

  // Update header when folder changes
  useEffect(() => {
    // Set breadcrumbs - Removed - MorphikUI handles breadcrumbs centrally
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const breadcrumbs = selectedFolder
      ? [
          {
            label: "Documents",
            onClick: (e: React.MouseEvent) => {
              e.preventDefault();
              setSelectedFolder(null);
              handleFolderClick(null);
            },
          },
          { label: selectedFolder === "all" ? "All Documents" : selectedFolder },
        ]
      : [{ label: "Documents" }];

    // Disabled - MorphikUI handles breadcrumbs with organization context
    // setCustomBreadcrumbs(breadcrumbs);

    // Set right content based on current view
    const rightContent = selectedFolder ? (
      // Folder view controls
      <>
        {documentsSectionRef.current && documentsSectionRef.current.selectedDocuments.length > 0 && (
          <Button
            variant="outline"
            size="icon"
            onClick={handleDeleteMultiple}
            className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
            title={`Delete ${documentsSectionRef.current.selectedDocuments.length} selected document${
              documentsSectionRef.current.selectedDocuments.length > 1 ? "s" : ""
            }`}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}

        <Button variant="outline" size="sm" onClick={handleRefresh} title="Refresh documents">
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>

        <Button variant="default" size="sm" onClick={() => setShowUploadDialog(true)}>
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </>
    ) : (
      // Root level controls
      <>
        <Button variant="outline" size="sm" onClick={() => setShowNewFolderDialog(true)}>
          <PlusCircle className="mr-2 h-4 w-4" />
          New Folder
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAllFoldersExpanded(prev => !prev)}
          className="flex items-center gap-1.5"
          title="Expand or collapse all folders"
        >
          {allFoldersExpanded ? <ChevronsUp className="h-4 w-4" /> : <ChevronsDown className="h-4 w-4" />}
          <span>{allFoldersExpanded ? "Collapse All" : "Expand All"}</span>
        </Button>
        <Button variant="outline" size="sm" onClick={handleRefresh} title="Refresh documents">
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>
        <Button variant="default" size="sm" onClick={() => setShowUploadDialog(true)}>
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </>
    );

    setRightContent(rightContent);

    // Cleanup on unmount
    return () => {
      // Disabled - MorphikUI handles breadcrumbs with organization context
      // setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [
    selectedFolder,
    allFoldersExpanded,
    handleFolderClick,
    handleRefresh,
    handleDeleteMultiple,
    setCustomBreadcrumbs,
    setRightContent,
  ]);

  return (
    <DocumentsSection
      {...props}
      ref={documentsSectionRef}
      onFolderClick={handleFolderClick}
      showUploadDialog={showUploadDialog}
      setShowUploadDialog={setShowUploadDialog}
      showNewFolderDialog={showNewFolderDialog}
      setShowNewFolderDialog={setShowNewFolderDialog}
      allFoldersExpanded={allFoldersExpanded}
    />
  );
}
