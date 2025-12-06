"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { useHeader } from "../../contexts/header-context";
import { Button } from "@/components/ui/button";
import { Trash2, Upload, RefreshCw, PlusCircle } from "lucide-react";
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
  const { setRightContent } = useHeader();
  const [selectedFolder, setSelectedFolder] = useState<string | null>(props.initialFolder || null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);

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
    // Set right content based on current view
    const rightContent = selectedFolder ? (
      // Folder view controls
      <>
        <Button variant="outline" size="sm" onClick={() => setShowNewFolderDialog(true)}>
          <PlusCircle className="mr-2 h-4 w-4" />
          New Folder
        </Button>
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
      // Breadcrumbs are handled centrally by MorphikUI
      setRightContent(null);
    };
  }, [selectedFolder, handleRefresh, handleDeleteMultiple, setRightContent]);

  return (
    <DocumentsSection
      {...props}
      ref={documentsSectionRef}
      onFolderClick={handleFolderClick}
      showUploadDialog={showUploadDialog}
      setShowUploadDialog={setShowUploadDialog}
      showNewFolderDialog={showNewFolderDialog}
      setShowNewFolderDialog={setShowNewFolderDialog}
    />
  );
}
