"use client";

import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import {
  Info,
  Calendar,
  Clock,
  Copy,
  Check,
  Edit2,
  Save,
  X,
  Plus,
  Trash2,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { showAlert } from "@/components/ui/alert-system";
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
  onMetadataUpdate?: (documentId: string) => void; // Callback to refresh selected document
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
  onMetadataUpdate,
}) => {
  const [isMovingToFolder, setIsMovingToFolder] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [copiedDocumentId, setCopiedDocumentId] = useState(false);
  const [isEditingMetadata, setIsEditingMetadata] = useState(false);
  const [editedMetadata, setEditedMetadata] = useState<Record<string, unknown>>({});
  const [newMetadataKey, setNewMetadataKey] = useState("");
  const [newMetadataValue, setNewMetadataValue] = useState("");
  const [isSavingMetadata, setIsSavingMetadata] = useState(false);
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set());

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

  const currentFolder = selectedDocument.folder_name;
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

  // Helper function to toggle expanded state
  const toggleExpanded = (key: string) => {
    setExpandedKeys(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

  // Helper function to set value at a nested path
  const setNestedValue = (obj: Record<string, unknown>, path: string[], value: unknown): Record<string, unknown> => {
    const newObj = JSON.parse(JSON.stringify(obj)) as Record<string, unknown>; // Deep clone
    let current = newObj as Record<string, unknown>;

    for (let i = 0; i < path.length - 1; i++) {
      if (!current[path[i]]) {
        current[path[i]] = {};
      }
      current = current[path[i]] as Record<string, unknown>;
    }

    if (value === undefined) {
      delete current[path[path.length - 1]];
    } else {
      current[path[path.length - 1]] = value;
    }

    return newObj;
  };

  // Helper function to check if value is a primitive
  const isPrimitive = (value: unknown): boolean => {
    return (
      value === null ||
      value === undefined ||
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    );
  };

  // Helper function to parse JSON safely
  const parseJsonSafely = (value: string): unknown => {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  };

  // Initialize edited metadata when starting to edit
  const startEditingMetadata = () => {
    setEditedMetadata(selectedDocument.metadata || {});
    setIsEditingMetadata(true);
    // Expand all first-level objects by default
    const firstLevelObjects = Object.entries(selectedDocument.metadata || {})
      .filter(([, value]) => !isPrimitive(value))
      .map(([key]) => key);
    setExpandedKeys(new Set(firstLevelObjects));
  };

  // Cancel metadata editing
  const cancelEditingMetadata = () => {
    setEditedMetadata({});
    setNewMetadataKey("");
    setNewMetadataValue("");
    setIsEditingMetadata(false);
    setExpandedKeys(new Set());
  };

  // Update a metadata field at a given path
  const updateMetadataField = (path: string[], value: string) => {
    const parsedValue = parseJsonSafely(value);
    setEditedMetadata(prev => setNestedValue(prev as Record<string, unknown>, path, parsedValue));
  };

  // Delete a metadata field at a given path
  const deleteMetadataField = (path: string[]) => {
    setEditedMetadata(prev => setNestedValue(prev as Record<string, unknown>, path, undefined));
  };

  // Add a new field to an object at a given path
  const addFieldToObject = (parentPath: string[], key: string, value: string) => {
    const parsedValue = parseJsonSafely(value);
    const newPath = [...parentPath, key];
    setEditedMetadata(prev => setNestedValue(prev as Record<string, unknown>, newPath, parsedValue));
  };

  // Add new metadata field at root level
  const addMetadataField = () => {
    if (newMetadataKey.trim() && !editedMetadata.hasOwnProperty(newMetadataKey)) {
      const parsedValue = parseJsonSafely(newMetadataValue);
      setEditedMetadata(prev => ({
        ...prev,
        [newMetadataKey.trim()]: parsedValue,
      }));
      setNewMetadataKey("");
      setNewMetadataValue("");
    }
  };

  // Save metadata changes
  const saveMetadata = async () => {
    if (!selectedDocument) return;

    setIsSavingMetadata(true);
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${selectedDocument.external_id}/update_metadata`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify(editedMetadata),
      });

      if (!response.ok) {
        throw new Error(`Failed to update metadata: ${response.statusText}`);
      }

      showAlert("Metadata updated successfully", { type: "success", duration: 3000 });

      // Refresh documents to get the updated data
      await refreshDocuments();

      // Call the metadata update callback to refresh the selected document
      if (onMetadataUpdate) {
        onMetadataUpdate(selectedDocument.external_id);
      }

      // Exit edit mode
      setIsEditingMetadata(false);
      setEditedMetadata({});
    } catch (error) {
      console.error("Error updating metadata:", error);
      showAlert(`Failed to update metadata: ${error instanceof Error ? error.message : String(error)}`, {
        type: "error",
        duration: 3000,
      });
    } finally {
      setIsSavingMetadata(false);
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

  // Recursive component to render nested metadata
  const MetadataEditor = ({ data, path = [], depth = 0 }: { data: unknown; path?: string[]; depth?: number }) => {
    if (isPrimitive(data) || data === null || data === undefined) {
      return null;
    }

    const isArray = Array.isArray(data);
    const entries = isArray ? data.map((item, index) => [index.toString(), item]) : Object.entries(data);

    return (
      <div className={depth > 0 ? "ml-4 space-y-2" : "space-y-2"}>
        {entries.map(([key, value]) => {
          const currentPath = [...path, key];
          const pathKey = currentPath.join(".");
          const isExpanded = expandedKeys.has(pathKey);
          const isObject = !isPrimitive(value) && value !== null;

          return (
            <div key={key} className="space-y-1">
              <div className="flex items-center gap-2">
                {isObject && (
                  <Button variant="ghost" size="icon" onClick={() => toggleExpanded(pathKey)} className="h-6 w-6 p-0">
                    {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </Button>
                )}
                {!isObject && <div className="w-6" />}

                <Input value={key} disabled={isArray} className="h-8 w-[140px] flex-shrink-0 text-sm font-medium" />

                {isPrimitive(value) ? (
                  <Input
                    value={String(value ?? "")}
                    onChange={e => updateMetadataField(currentPath, e.target.value)}
                    className="h-8 flex-1 text-sm"
                    placeholder="Value"
                  />
                ) : (
                  <div className="flex-1 rounded border bg-muted/30 px-2 py-1 text-sm text-muted-foreground">
                    {Array.isArray(value) ? `Array[${value.length}]` : `Object{${Object.keys(value).length}}`}
                  </div>
                )}

                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => deleteMetadataField(currentPath)}
                  className="h-8 w-8 text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>

              {isObject && isExpanded && <MetadataEditor data={value} path={currentPath} depth={depth + 1} />}

              {isObject && isExpanded && !isArray && (
                <div className="ml-10 flex items-center gap-2 border-t pt-2">
                  <Input
                    placeholder="New key"
                    className="h-8 w-[140px] text-sm"
                    onKeyDown={e => {
                      if (e.key === "Enter" && e.currentTarget.value.trim()) {
                        const newKey = e.currentTarget.value.trim();
                        addFieldToObject(currentPath, newKey, "");
                        e.currentTarget.value = "";
                      }
                    }}
                  />
                  <span className="text-xs text-muted-foreground">Press Enter to add</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
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

          {/* Metadata Section */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">Metadata</h3>
              {!isEditingMetadata ? (
                <Button variant="ghost" size="sm" onClick={startEditingMetadata} className="h-8 px-2">
                  <Edit2 className="mr-1 h-4 w-4" />
                  Edit
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={cancelEditingMetadata}
                    disabled={isSavingMetadata}
                    className="h-8 px-2"
                  >
                    <X className="mr-1 h-4 w-4" />
                    Cancel
                  </Button>
                  <Button size="sm" onClick={saveMetadata} disabled={isSavingMetadata} className="h-8 px-2">
                    <Save className="mr-1 h-4 w-4" />
                    {isSavingMetadata ? "Saving..." : "Save"}
                  </Button>
                </div>
              )}
            </div>

            <div className="rounded-lg border bg-muted/30 p-3">
              {!isEditingMetadata ? (
                // View mode
                <div className="space-y-2">
                  {selectedDocument.metadata && Object.keys(selectedDocument.metadata).length > 0 ? (
                    Object.entries(selectedDocument.metadata).map(([key, value]) => (
                      <div key={key} className="flex items-start gap-2">
                        <span className="min-w-[120px] text-sm font-medium">{key}:</span>
                        <span className="break-all text-sm text-muted-foreground">
                          {typeof value === "object" ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No metadata available</p>
                  )}
                </div>
              ) : (
                // Edit mode
                <div className="space-y-3">
                  <MetadataEditor data={editedMetadata} />

                  {/* Add new field */}
                  <div className="flex items-center gap-2 border-t pt-3">
                    <Input
                      value={newMetadataKey}
                      onChange={e => setNewMetadataKey(e.target.value)}
                      placeholder="New key"
                      className="h-8 w-[140px] flex-shrink-0 text-sm"
                      onKeyPress={e => e.key === "Enter" && addMetadataField()}
                    />
                    <Input
                      value={newMetadataValue}
                      onChange={e => setNewMetadataValue(e.target.value)}
                      placeholder="New value (JSON supported)"
                      className="h-8 flex-1 text-sm"
                      onKeyPress={e => e.key === "Enter" && addMetadataField()}
                    />
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={addMetadataField}
                      disabled={!newMetadataKey.trim()}
                      className="h-8 w-8"
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>

          <Accordion type="single" collapsible>
            <AccordionItem value="system-metadata">
              <AccordionTrigger>Text Content</AccordionTrigger>
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
