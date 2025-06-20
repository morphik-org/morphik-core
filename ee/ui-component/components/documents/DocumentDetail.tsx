"use client";

import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Info, Calendar, Clock, Copy, Check } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Image from "next/image";
import DeleteConfirmationModal from "./DeleteConfirmationModal";

import { Document, FolderSummary } from "@/components/types";

interface DocumentDetailProps {
  selectedDocument: Document | null;
  handleDeleteDocument: (documentId: string) => Promise<void>;
  folders: FolderSummary[];
  apiBaseUrl: string;
  authToken: string | null;
  refreshDocuments: () => void;
  refreshFolders: () => void;
  loading: boolean;
  onClose: () => void;
  onViewInPDFViewer?: (documentId: string) => void; // Add navigation callback
}

const DocumentDetail: React.FC<DocumentDetailProps> = ({
  selectedDocument,
  handleDeleteDocument,
  folders,
  apiBaseUrl,
  authToken,
  refreshDocuments,
  refreshFolders,
  loading,
  onClose,
  onViewInPDFViewer,
}) => {
  const [isMovingToFolder, setIsMovingToFolder] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [copiedDocumentId, setCopiedDocumentId] = useState(false);

  if (!selectedDocument) {
    return (
      <div className="flex h-[calc(100vh-200px)] items-center justify-center rounded-lg border border-dashed p-8">
        <div className="text-center text-muted-foreground">
          <Info className="mx-auto mb-2 h-12 w-12" />
          <p>Select a document to view details</p>
        </div>
      </div>
    );
  }

  const currentFolder = selectedDocument.system_metadata?.folder_name as string | undefined;
  const status = selectedDocument.system_metadata?.status as string | undefined;
  const error = selectedDocument.system_metadata?.error as string | undefined;
  const createdAt = selectedDocument.system_metadata?.created_at as string | undefined;
  const updatedAt = selectedDocument.system_metadata?.updated_at as string | undefined;
  const version = selectedDocument.system_metadata?.version as number | undefined;

  // Format dates for display
  const formatDate = (dateString?: string) => {
    if (!dateString) return "N/A";
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  // Copy document ID to clipboard
  const copyDocumentId = async () => {
    try {
      await navigator.clipboard.writeText(selectedDocument.external_id);
      setCopiedDocumentId(true);
      setTimeout(() => setCopiedDocumentId(false), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy document ID:", err);
    }
  };

  // Get status badge variant
  const getStatusBadge = (status?: string) => {
    if (!status) return <Badge variant="outline">Unknown</Badge>;

    switch (status.toLowerCase()) {
      case "completed":
        return (
          <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100">
            Completed
          </Badge>
        );
      case "processing":
        return (
          <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100">
            Processing
          </Badge>
        );
      case "failed":
        return <Badge variant="destructive">Failed</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const handleDeleteConfirm = async () => {
    if (selectedDocument) {
      await handleDeleteDocument(selectedDocument.external_id);
      setShowDeleteModal(false);
    }
  };

  const handleMoveToFolder = async (folderName: string | null) => {
    if (isMovingToFolder || !selectedDocument) return;

    const documentId = selectedDocument.external_id;
    setIsMovingToFolder(true);

    try {
      // First, get the folder ID from the name if a name is provided
      if (folderName) {
        // Find the target folder by name
        const targetFolder = folders.find(folder => folder.name === folderName);
        if (targetFolder && targetFolder.id) {
          console.log(`Found folder with ID: ${targetFolder.id} for name: ${folderName}`);

          // Add to folder using folder ID
          await fetch(`${apiBaseUrl}/folders/${targetFolder.id}/documents/${documentId}`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
          });
        } else {
          console.error(`Could not find folder with name: ${folderName}`);
        }
      }

      // If there's a current folder and we're either moving to a new folder or removing from folder
      if (currentFolder) {
        // Find the current folder ID
        const currentFolderObj = folders.find(folder => folder.name === currentFolder);
        if (currentFolderObj && currentFolderObj.id) {
          // Remove from current folder using folder ID
          await fetch(`${apiBaseUrl}/folders/${currentFolderObj.id}/documents/${documentId}`, {
            method: "DELETE",
            headers: {
              "Content-Type": "application/json",
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
          });
        }
      }

      // Refresh folders first to get updated document_ids
      await refreshFolders();
      // Then refresh documents with the updated folder information
      await refreshDocuments();
    } catch (error) {
      console.error("Error updating folder:", error);
    } finally {
      setIsMovingToFolder(false);
    }
  };

  return (
    <div className="rounded-lg border">
      <div className="sticky top-0 flex items-center justify-between border-b bg-muted px-4 py-3">
        <h3 className="text-sm font-semibold">Document Details</h3>
        <Button variant="ghost" size="icon" onClick={onClose} className="rounded-full hover:bg-background/80">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
          <span className="sr-only">Close panel</span>
        </Button>
      </div>

      <ScrollArea className="h-[calc(100vh-200px)]">
        <div className="space-y-4 p-4">
          <div>
            <h3 className="mb-1 font-medium">Filename</h3>
            <p>{selectedDocument.filename || "N/A"}</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="mb-1 font-medium">Content Type</h3>
              <Badge variant="secondary">{selectedDocument.content_type}</Badge>
            </div>
            <div>
              <h3 className="mb-1 font-medium">Status</h3>
              {getStatusBadge(status)}
            </div>
          </div>

          {/* Error message for failed documents */}
          {status === "failed" && error && (
            <div className="rounded-lg bg-red-50 p-3 dark:bg-red-950/20">
              <h3 className="mb-1 font-medium text-red-800 dark:text-red-200">Error</h3>
              <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          {/* PDF Viewer Button - only show for PDF documents */}
          {selectedDocument.content_type === "application/pdf" && onViewInPDFViewer && (
            <div>
              <Button
                onClick={() => onViewInPDFViewer(selectedDocument.external_id)}
                className="w-full"
                disabled={loading}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="mr-2"
                >
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                View in PDF Viewer
              </Button>
            </div>
          )}

          <div>
            <h3 className="mb-1 font-medium">Folder</h3>
            <div className="flex items-center gap-2">
              <Image src="/icons/folder-icon.png" alt="Folder" width={16} height={16} />
              <Select
                value={currentFolder || "_none"}
                onValueChange={value => handleMoveToFolder(value === "_none" ? null : value)}
                disabled={isMovingToFolder}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Not in a folder" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="_none">Not in a folder</SelectItem>
                  {folders.map(folder => (
                    <SelectItem key={folder.name} value={folder.name}>
                      {folder.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="mb-1 flex items-center gap-1 font-medium">
                <Calendar className="h-4 w-4" />
                Created
              </h3>
              <p className="text-sm">{formatDate(createdAt)}</p>
            </div>
            <div>
              <h3 className="mb-1 flex items-center gap-1 font-medium">
                <Clock className="h-4 w-4" />
                Updated
              </h3>
              <p className="text-sm">{formatDate(updatedAt)}</p>
            </div>
          </div>

          <div>
            <h3 className="mb-1 font-medium">Document ID</h3>
            <button
              onClick={copyDocumentId}
              className="group flex items-center gap-2 font-mono text-xs text-muted-foreground transition-colors hover:text-foreground"
              title="Click to copy Document ID"
            >
              <span>{selectedDocument.external_id}</span>
              {copiedDocumentId ? (
                <Check className="h-3 w-3 text-green-500" />
              ) : (
                <Copy className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
              )}
            </button>
          </div>

          {version !== undefined && (
            <div>
              <h3 className="mb-1 font-medium">Version</h3>
              <p>{version}</p>
            </div>
          )}

          <Accordion type="single" collapsible>
            <AccordionItem value="metadata">
              <AccordionTrigger>Metadata</AccordionTrigger>
              <AccordionContent>
                <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-muted p-2 text-xs">
                  {JSON.stringify(selectedDocument.metadata, null, 2)}
                </pre>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="system-metadata">
              <AccordionTrigger>Text</AccordionTrigger>
              <AccordionContent>
                <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-muted p-2 text-xs">
                  {JSON.stringify(selectedDocument.system_metadata.content, null, 2)}
                </pre>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="mt-4 border-t pt-4">
            <Button
              variant="outline"
              size="sm"
              className="w-full border-red-500 text-red-500 hover:bg-red-100 dark:hover:bg-red-950"
              onClick={() => setShowDeleteModal(true)}
              disabled={loading}
            >
              Delete Document
            </Button>
          </div>
        </div>
      </ScrollArea>
      {selectedDocument && (
        <DeleteConfirmationModal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          onConfirm={handleDeleteConfirm}
          itemName={selectedDocument.filename || selectedDocument.external_id}
          loading={loading}
        />
      )}
    </div>
  );
};

export default DocumentDetail;
