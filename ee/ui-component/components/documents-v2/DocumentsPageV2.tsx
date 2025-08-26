"use client";

import React, { useEffect, useState } from "react";
import { DocumentsProviderV2 } from "./DocumentsProviderV2";
import { DocumentsHeaderV2 } from "./DocumentsHeaderV2";
import { DocumentsTableV2 } from "./DocumentsTableV2";
import { DocumentDetailModalV2 } from "./DocumentDetailModalV2";
import { UploadManagerV2 } from "./UploadManagerV2";
import { NewFolderDialogV2 } from "./NewFolderDialogV2";

export interface DocumentsPageV2Props {
  // Core props from API
  apiBaseUrl: string;
  authToken: string | null;

  // Optional callbacks
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderClick?: (folderName: string | null) => void;
  onFolderCreate?: (folderName: string) => void;
  onRefresh?: () => void;
  onViewInPDFViewer?: (documentId: string) => void;

  // Optional initial state
  initialFolder?: string | null;
}

export function DocumentsPageV2({
  apiBaseUrl,
  authToken,
  onDocumentUpload,
  onDocumentDelete,
  onDocumentClick,
  onFolderClick,
  onFolderCreate,
  onRefresh,
  onViewInPDFViewer,
  initialFolder,
}: DocumentsPageV2Props) {
  // Only render on client to avoid hydration issues with data fetching
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-muted-foreground">Loading documents...</div>
      </div>
    );
  }

  return (
    <DocumentsProviderV2
      apiBaseUrl={apiBaseUrl}
      authToken={authToken}
      initialFolder={initialFolder}
      onDocumentUpload={onDocumentUpload}
      onDocumentDelete={onDocumentDelete}
      onDocumentClick={onDocumentClick}
      onFolderClick={onFolderClick}
      onFolderCreate={onFolderCreate}
      onRefresh={onRefresh}
      onViewInPDFViewer={onViewInPDFViewer}
    >
      <div className="flex h-full flex-col">
        {/* Header with search and actions */}
        <DocumentsHeaderV2 />

        {/* Main content area */}
        <div className="flex-1 overflow-auto">
          <DocumentsTableV2 />
        </div>

        {/* Modal for document details */}
        <DocumentDetailModalV2 />

        {/* Upload manager handles drag-drop and progress */}
        <UploadManagerV2 />

        {/* New folder dialog */}
        <NewFolderDialogV2 />
      </div>
    </DocumentsProviderV2>
  );
}

export default DocumentsPageV2;
