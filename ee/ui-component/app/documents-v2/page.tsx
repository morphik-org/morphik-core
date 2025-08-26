"use client";

export const dynamic = "force-dynamic";

import DocumentsPageV2 from "@/components/documents-v2/DocumentsPageV2";
import { useMorphik } from "@/contexts/morphik-context";

export default function DocumentsV2Page() {
  const { apiBaseUrl, authToken } = useMorphik();

  return (
    <DocumentsPageV2
      apiBaseUrl={apiBaseUrl}
      authToken={authToken}
      onDocumentUpload={(fileName, fileSize) => {
        console.log("Document uploaded:", fileName, fileSize);
      }}
      onDocumentDelete={fileName => {
        console.log("Document deleted:", fileName);
      }}
      onDocumentClick={fileName => {
        console.log("Document clicked:", fileName);
      }}
      onFolderClick={folderName => {
        console.log("Folder clicked:", folderName);
      }}
      onFolderCreate={folderName => {
        console.log("Folder created:", folderName);
      }}
    />
  );
}
