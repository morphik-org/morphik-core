"use client";

import React from "react";
import DocumentsSection from "./DocumentsSection";
import { useDocumentsHeader } from "@/hooks/useDocumentsHeader";

interface DocumentsSectionWithHeaderProps {
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

export default function DocumentsSectionWithHeader(props: DocumentsSectionWithHeaderProps) {
  const [selectedFolder, setSelectedFolder] = React.useState<string | null>(null);
  const [selectedDocuments] = React.useState<string[]>([]);
  const [uploadDialogComponent] = React.useState<React.ReactNode>(null);

  // Use the header hook
  useDocumentsHeader({
    selectedFolder,
    onNavigateHome: () => setSelectedFolder(null),
    selectedDocuments,
    uploadDialogComponent,
  });

  return (
    <DocumentsSection
      {...props}
      onFolderClick={folderName => {
        setSelectedFolder(folderName);
        props.onFolderClick?.(folderName);
      }}
    />
  );
}
