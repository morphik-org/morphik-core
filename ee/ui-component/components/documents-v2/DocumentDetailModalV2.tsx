"use client";

import React, { useEffect, useState } from "react";
import { useDocumentsV2 } from "./DocumentsProviderV2";
import { Document, ProcessingProgress } from "../types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Label } from "../ui/label";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Progress } from "../ui/progress";
import { Badge } from "../ui/badge";
import { Separator } from "../ui/separator";
import { ScrollArea } from "../ui/scroll-area";
import { toast } from "sonner";
import {
  Download,
  File,
  FileText,
  FileImage,
  FileVideo,
  FileAudio,
  FileCode,
  FileArchive,
  Loader2,
  CheckCircle,
  AlertCircle,
  Copy,
  ExternalLink,
  Plus,
  X,
  Save,
  Edit2,
} from "lucide-react";

// Helper to get file icon based on content type
function getFileIcon(contentType?: string) {
  if (!contentType) return File;

  const type = contentType.toLowerCase();

  if (type.includes("image")) return FileImage;
  if (type.includes("video")) return FileVideo;
  if (type.includes("audio")) return FileAudio;
  if (type.includes("pdf")) return FileText;
  if (type.includes("zip") || type.includes("tar") || type.includes("rar")) return FileArchive;
  if (type.includes("json") || type.includes("xml") || type.includes("javascript") || type.includes("typescript"))
    return FileCode;
  if (type.includes("text")) return FileText;

  return File;
}


// Helper to copy text to clipboard
async function copyToClipboard(text: string, label: string = "Text") {
  try {
    await navigator.clipboard.writeText(text);
    toast.success(`${label} copied to clipboard`);
  } catch {
    toast.error("Failed to copy to clipboard");
  }
}

export function DocumentDetailModalV2() {
  const { selectedDocument, setSelectedDocument, fetchDocumentDetail, apiBaseUrl, authToken, onViewInPDFViewer } =
    useDocumentsV2();

  const [isOpen, setIsOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [detailDocument, setDetailDocument] = useState<Document | null>(null);
  const [isEditingMetadata, setIsEditingMetadata] = useState(false);
  const [editedMetadata, setEditedMetadata] = useState<Record<string, any>>({});
  const [newMetadataKey, setNewMetadataKey] = useState("");
  const [newMetadataValue, setNewMetadataValue] = useState("");
  const [isSavingMetadata, setIsSavingMetadata] = useState(false);

  // Listen for custom event to open modal
  useEffect(() => {
    const handleOpenDetail = (event: CustomEvent<Document>) => {
      if (event.detail) {
        setDetailDocument(event.detail);
        setIsOpen(true);
        // Fetch fresh details
        fetchFullDetails(event.detail.external_id);
      }
    };

    window.addEventListener("openDocumentDetail" as any, handleOpenDetail);
    return () => window.removeEventListener("openDocumentDetail" as any, handleOpenDetail);
  }, []);

  // Also open when selectedDocument changes
  useEffect(() => {
    if (selectedDocument) {
      setDetailDocument(selectedDocument);
      setEditedMetadata(selectedDocument.metadata || {});
      setIsOpen(true);
      fetchFullDetails(selectedDocument.external_id);
    }
  }, [selectedDocument]);

  // Fetch full document details
  const fetchFullDetails = async (documentId: string) => {
    try {
      const fullDoc = await fetchDocumentDetail(documentId);
      if (fullDoc) {
        setDetailDocument(fullDoc);
        setEditedMetadata(fullDoc.metadata || {});
      }
    } catch (error) {
      console.error("Failed to fetch document details:", error);
      toast.error("Failed to load document details");
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setSelectedDocument(null);
    setDetailDocument(null);
    setIsEditingMetadata(false);
    setEditedMetadata({});
    setNewMetadataKey("");
    setNewMetadataValue("");
  };

  const handleDownload = async () => {
    if (!detailDocument) return;

    setIsDownloading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${detailDocument.external_id}/download`, {
        method: "GET",
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = detailDocument.filename || detailDocument.external_id;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success(`Downloaded "${detailDocument.filename || detailDocument.external_id}"`);
      } else {
        toast.error("Failed to download document");
      }
    } catch (error) {
      console.error("Download error:", error);
      toast.error("Failed to download document");
    } finally {
      setIsDownloading(false);
    }
  };

  const handleViewInPDFViewer = () => {
    if (detailDocument && onViewInPDFViewer) {
      onViewInPDFViewer(detailDocument.external_id);
      handleClose();
    }
  };

  const handleSaveMetadata = async () => {
    if (!detailDocument) return;

    setIsSavingMetadata(true);
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${detailDocument.external_id}/update_metadata`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify(editedMetadata),
      });

      if (response.ok) {
        const updatedDoc = await response.json();
        setDetailDocument(updatedDoc);
        setIsEditingMetadata(false);
        toast.success("Metadata updated successfully");
      } else {
        toast.error("Failed to update metadata");
      }
    } catch (error) {
      console.error("Error updating metadata:", error);
      toast.error("Failed to update metadata");
    } finally {
      setIsSavingMetadata(false);
    }
  };

  const handleAddMetadataField = () => {
    if (!newMetadataKey.trim()) {
      toast.error("Please enter a key for the new metadata field");
      return;
    }

    try {
      // Try to parse as JSON, fallback to string
      let value: any = newMetadataValue;
      try {
        value = JSON.parse(newMetadataValue);
      } catch {
        // Keep as string
      }

      setEditedMetadata(prev => ({
        ...prev,
        [newMetadataKey]: value
      }));

      setNewMetadataKey("");
      setNewMetadataValue("");
      toast.success(`Added field: ${newMetadataKey}`);
    } catch (error) {
      toast.error("Failed to add metadata field");
    }
  };

  const handleRemoveMetadataField = (key: string) => {
    setEditedMetadata(prev => {
      const updated = { ...prev };
      delete updated[key];
      return updated;
    });
  };

  const handleMetadataValueChange = (key: string, value: string) => {
    try {
      // Try to parse as JSON, fallback to string
      let parsedValue: any = value;
      try {
        parsedValue = JSON.parse(value);
      } catch {
        // Keep as string
      }

      setEditedMetadata(prev => ({
        ...prev,
        [key]: parsedValue
      }));
    } catch {
      // Keep existing value on parse error
    }
  };

  if (!detailDocument) return null;

  const displayName = detailDocument.filename || detailDocument.external_id;
  const status = detailDocument.system_metadata?.status as string | undefined;
  const processingProgress = detailDocument.system_metadata?.processing_progress as ProcessingProgress | undefined;
  const FileIcon = getFileIcon(detailDocument.content_type);

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-3xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <FileIcon className="h-5 w-5 text-muted-foreground" />
            <span className="truncate">{displayName}</span>
            {status && (
              <Badge variant={status === "completed" ? "default" : status === "processing" ? "secondary" : "destructive"}>
                {status}
              </Badge>
            )}
          </DialogTitle>
          {detailDocument.folder_name && (
            <DialogDescription>Located in folder: {detailDocument.folder_name}</DialogDescription>
          )}
        </DialogHeader>

          <div className="space-y-4">
            {/* Processing Progress */}
            {status === "processing" && processingProgress && (
              <div className="space-y-2 rounded-lg border p-4 bg-muted/50">
                <Label>Processing Progress</Label>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{processingProgress.step_name}</span>
                    <span className="text-muted-foreground">
                      Step {processingProgress.current_step} of {processingProgress.total_steps}
                    </span>
                  </div>
                  <Progress value={processingProgress.percentage || 0} className="h-2" />
                </div>
              </div>
            )}

            {/* Basic Information */}
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Document ID</Label>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-mono truncate">{detailDocument.external_id}</p>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={() => copyToClipboard(detailDocument.external_id, "Document ID")}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Filename</Label>
                  <p className="text-sm">{detailDocument.filename || "N/A"}</p>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Content Type</Label>
                  <p className="text-sm">{detailDocument.content_type || "Unknown"}</p>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Status</Label>
                  <div className="flex items-center gap-2">
                    {status === "processing" && (
                      <>
                        <Loader2 className="h-3 w-3 animate-spin" />
                        <span className="text-sm">Processing</span>
                      </>
                    )}
                    {status === "completed" && (
                      <>
                        <CheckCircle className="h-3 w-3 text-green-600" />
                        <span className="text-sm">Ready</span>
                      </>
                    )}
                    {status === "failed" && (
                      <>
                        <AlertCircle className="h-3 w-3 text-destructive" />
                        <span className="text-sm">Failed</span>
                      </>
                    )}
                    {!status && (
                      <>
                        <CheckCircle className="h-3 w-3 text-muted-foreground" />
                        <span className="text-sm">Ready</span>
                      </>
                    )}
                  </div>
                </div>

                {detailDocument.folder_name && (
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Folder</Label>
                    <p className="text-sm">{detailDocument.folder_name}</p>
                  </div>
                )}

                {detailDocument.end_user_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">End User ID</Label>
                    <p className="text-sm font-mono">{detailDocument.end_user_id}</p>
                  </div>
                )}
              </div>
            </div>

            <Separator />

            {/* Metadata Section */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-medium">Metadata</h4>
                {!isEditingMetadata ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsEditingMetadata(true)}
                  >
                    <Edit2 className="mr-2 h-3 w-3" />
                    Edit
                  </Button>
                ) : (
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setIsEditingMetadata(false);
                        setEditedMetadata(detailDocument.metadata || {});
                      }}
                    >
                      Cancel
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleSaveMetadata}
                      disabled={isSavingMetadata}
                    >
                      {isSavingMetadata ? (
                        <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                      ) : (
                        <Save className="mr-2 h-3 w-3" />
                      )}
                      Save
                    </Button>
                  </div>
                )}
              </div>

              {isEditingMetadata ? (
                <div className="space-y-3">
                  {/* Existing metadata fields */}
                  {Object.entries(editedMetadata).map(([key, value]) => (
                    <div key={key} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs text-muted-foreground">{key}</Label>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => handleRemoveMetadataField(key)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                      <Textarea
                        className="text-sm font-mono min-h-[60px]"
                        value={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
                        onChange={(e) => handleMetadataValueChange(key, e.target.value)}
                      />
                    </div>
                  ))}

                  {/* Add new field */}
                  <div className="space-y-2 rounded-lg border p-3 bg-muted/30">
                    <Label className="text-xs">Add New Field</Label>
                    <div className="flex gap-2">
                      <Input
                        placeholder="Key"
                        value={newMetadataKey}
                        onChange={(e) => setNewMetadataKey(e.target.value)}
                        className="flex-1"
                      />
                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={handleAddMetadataField}
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                    <Textarea
                      placeholder="Value (JSON or string)"
                      value={newMetadataValue}
                      onChange={(e) => setNewMetadataValue(e.target.value)}
                      className="text-sm font-mono min-h-[80px]"
                    />
                  </div>

                  {Object.keys(editedMetadata).length === 0 && (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No metadata. Add a field above.
                    </p>
                  )}
                </div>
              ) : (
                <div>
                  {Object.keys(detailDocument.metadata || {}).length > 0 ? (
                    <div className="space-y-3">
                      {Object.entries(detailDocument.metadata).map(([key, value]) => (
                        <div key={key} className="space-y-1">
                          <Label className="text-xs text-muted-foreground">{key}</Label>
                          <pre className="text-sm bg-muted p-3 rounded-md overflow-x-auto max-h-32">
                            {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No metadata. Click Edit to add metadata fields.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </ScrollArea>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Close
          </Button>
          {detailDocument.content_type?.includes("pdf") && onViewInPDFViewer && (
            <Button variant="outline" onClick={handleViewInPDFViewer}>
              <ExternalLink className="mr-2 h-4 w-4" />
              Open in PDF Viewer
            </Button>
          )}
          {status !== "processing" && (
            <Button onClick={handleDownload} disabled={isDownloading}>
              {isDownloading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Downloading...
                </>
              ) : (
                <>
                  <Download className="mr-2 h-4 w-4" />
                  Download
                </>
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
