"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { useDocumentsV2 } from "./DocumentsProviderV2";
import { ProcessingProgress } from "../types";
import { Upload, X, FileIcon, CheckCircle, AlertCircle, Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent } from "../ui/card";
import { Progress } from "../ui/progress";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "../ui/dialog";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";
import { Checkbox } from "../ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { toast } from "sonner";

interface UploadItem {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "uploaded" | "processing" | "complete" | "error";
  error?: string;
  processingProgress?: ProcessingProgress;
}

export function UploadManagerV2() {
  const {
    apiBaseUrl,
    authToken,
    currentFolder,
    fetchDocuments,
    setDocuments,
    documents,
    onDocumentUpload,
    updateDocumentStatus,
  } = useDocumentsV2();

  const [isDragging, setIsDragging] = useState(false);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [textContent, setTextContent] = useState("");
  const [metadata, setMetadata] = useState("{}");
  const [rules, setRules] = useState("[]");
  const [useColpali, setUseColpali] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [uploadType, setUploadType] = useState<"file" | "text">("file");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Generate unique ID
  const generateId = () => {
    return `upload-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  };

  // Poll for processing status
  const pollProcessingDocuments = useCallback(async () => {
    const processingDocs = documents.filter(
      doc => doc.system_metadata?.status === "processing" && !doc.external_id.startsWith("temp-")
    );

    if (processingDocs.length === 0) {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    try {
      // Fetch updated status for each processing document
      const statusPromises = processingDocs.map(async doc => {
        const response = await fetch(`${apiBaseUrl}/documents/${doc.external_id}`, {
          method: "GET",
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });

        if (response.ok) {
          const updatedDoc = await response.json();
          return updatedDoc;
        }
        return null;
      });

      const updatedDocs = await Promise.all(statusPromises);

      // Update document statuses in the main list only
      updatedDocs.forEach(updatedDoc => {
        if (updatedDoc) {
          // Update document status in the main document list
          if (updatedDoc.system_metadata?.status !== "processing") {
            updateDocumentStatus(updatedDoc.external_id, updatedDoc.system_metadata?.status || "completed");
          } else {
            // Update with progress information
            updateDocumentStatus(updatedDoc.external_id, "processing");
            // The document row will show the progress from system_metadata.progress
          }
        }
      });
    } catch (error) {
      console.error("Error polling document status:", error);
    }
  }, [documents, apiBaseUrl, authToken, updateDocumentStatus]);

  // Start polling immediately after upload and when there are processing documents
  useEffect(() => {
    const hasProcessingDocs = documents.some(doc => doc.system_metadata?.status === "processing");

    if (hasProcessingDocs && !pollingIntervalRef.current) {
      // Poll every 2 seconds
      pollingIntervalRef.current = setInterval(() => {
        pollProcessingDocuments();
        fetchDocuments();
      }, 2000);
    } else if (!hasProcessingDocs && pollingIntervalRef.current) {
      // Stop polling when no processing docs
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [documents, pollProcessingDocuments]);

  // Unified file upload handler
  const handleFileUpload = useCallback(
    async (files: File[], metadataOverride?: string, rulesOverride?: string, useColpaliOverride?: boolean) => {
      if (files.length === 0) return;

      const metadataToUse = metadataOverride ?? metadata;
      const rulesToUse = rulesOverride ?? rules;
      const useColpaliToUse = useColpaliOverride ?? useColpali;

      // Create upload items with unique IDs
      const uploadItems: UploadItem[] = files.map(file => ({
        id: generateId(),
        file,
        progress: 0,
        status: "pending" as const,
      }));

      // Add to uploads state for progress tracking
      setUploads(prev => [...prev, ...uploadItems]);

      // Prepare FormData
      const formData = new FormData();
      files.forEach(file => {
        formData.append("files", file);
      });

      // Add metadata and options
      formData.append("metadata", metadataToUse);
      formData.append("rules", rulesToUse);
      formData.append("use_colpali", String(useColpaliToUse));
      formData.append("parallel", "true");

      // Add folder name if we're in a folder
      if (currentFolder && currentFolder !== "all") {
        formData.append("folder_name", currentFolder);
      }

      // Use XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();

      // Track upload progress
      xhr.upload.onprogress = event => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;

          setUploads(prev =>
            prev.map(item => {
              if (uploadItems.find(u => u.id === item.id)) {
                return { ...item, progress: percentComplete, status: "uploading" };
              }
              return item;
            })
          );
        }
      };

      // Handle completion
      xhr.onload = () => {
        if (xhr.status === 200) {
          // Mark as uploaded (not processing yet)
          setUploads(prev =>
            prev.map(item => {
              if (uploadItems.find(u => u.id === item.id)) {
                return { ...item, progress: 100, status: "uploaded" };
              }
              return item;
            })
          );

          // Parse metadata
          let parsedMetadata = {};
          try {
            parsedMetadata = JSON.parse(metadataToUse);
          } catch {}

          // Don't add temp documents - just rely on upload notification

          // Call callback for each uploaded file
          files.forEach(file => {
            if (onDocumentUpload) {
              onDocumentUpload(file.name, file.size);
            }
          });

          // Show success toast
          if (files.length === 1) {
            toast.success(`"${files[0].name}" uploaded successfully`);
          } else {
            toast.success(`${files.length} files uploaded successfully`);
          }

          // Immediately fetch documents and start polling
          fetchDocuments();

          // Start polling for processing status
          if (!pollingIntervalRef.current) {
            pollingIntervalRef.current = setInterval(() => {
              fetchDocuments();
            }, 2000);
          }

          // Clear uploaded items after 2 seconds (they're now visible in the main list)
          setTimeout(() => {
            setUploads(prev => prev.filter(item => !uploadItems.find(u => u.id === item.id)));
          }, 2000);
        } else {
          // Handle error
          const errorMessage = `Upload failed: ${xhr.statusText}`;

          setUploads(prev =>
            prev.map(item => {
              if (uploadItems.find(u => u.id === item.id)) {
                return { ...item, status: "error", error: errorMessage };
              }
              return item;
            })
          );

          toast.error(errorMessage);
        }
      };

      // Handle errors
      xhr.onerror = () => {
        const errorMessage = "Network error occurred during upload";

        setUploads(prev =>
          prev.map(item => {
            if (uploadItems.find(u => u.id === item.id)) {
              return { ...item, status: "error", error: errorMessage };
            }
            return item;
          })
        );

        toast.error(errorMessage);
      };

      // Send request
      xhr.open("POST", `${apiBaseUrl}/ingest/files`);
      if (authToken) {
        xhr.setRequestHeader("Authorization", `Bearer ${authToken}`);
      }
      xhr.send(formData);

      // Close dialog if open
      setShowUploadDialog(false);
      setSelectedFiles([]);
      resetForm();
    },
    [
      apiBaseUrl,
      authToken,
      currentFolder,
      fetchDocuments,
      onDocumentUpload,
      documents,
      setDocuments,
      updateDocumentStatus,
      metadata,
      rules,
      useColpali,
    ]
  );

  // Handle text upload
  const handleTextUpload = useCallback(async () => {
    if (!textContent.trim()) {
      toast.error("Please enter text content");
      return;
    }

    const fileName = `text-${Date.now()}.txt`;

    // Parse metadata
    let parsedMetadata = {};
    try {
      parsedMetadata = JSON.parse(metadata);
    } catch {
      toast.error("Invalid metadata JSON");
      return;
    }

    // Parse rules
    let parsedRules = [];
    try {
      parsedRules = JSON.parse(rules);
    } catch {
      toast.error("Invalid rules JSON");
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/ingest/text`, {
        method: "POST",
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: textContent,
          metadata: parsedMetadata,
          rules: parsedRules,
          folder_name: currentFolder && currentFolder !== "all" ? currentFolder : undefined,
          use_colpali: useColpali,
        }),
      });

      if (response.ok) {
        toast.success("Text uploaded successfully");

        // Don't add temp document - just fetch real documents
        // Immediately fetch documents and start polling
        fetchDocuments();

        // Start polling for processing status
        if (!pollingIntervalRef.current) {
          pollingIntervalRef.current = setInterval(() => {
            fetchDocuments();
          }, 2000);
        }

        setShowUploadDialog(false);
        resetForm();
      } else {
        toast.error("Failed to upload text");
      }
    } catch (error) {
      console.error("Error uploading text:", error);
      toast.error("Error uploading text");
    }
  }, [
    textContent,
    metadata,
    rules,
    useColpali,
    apiBaseUrl,
    authToken,
    currentFolder,
    documents,
    setDocuments,
    fetchDocuments,
  ]);

  // Reset form
  const resetForm = () => {
    setSelectedFiles([]);
    setTextContent("");
    setMetadata("{}");
    setRules("[]");
    setUseColpali(true);
    setShowAdvanced(false);
  };

  // Listen for upload dialog open event
  useEffect(() => {
    const handleOpenUpload = () => {
      setShowUploadDialog(true);
      resetForm();
    };

    window.addEventListener("openUploadDialog", handleOpenUpload);
    return () => window.removeEventListener("openUploadDialog", handleOpenUpload);
  }, []);

  // Global drag and drop handlers
  useEffect(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer?.items && e.dataTransfer.items.length > 0) {
        dragCounter.current++;
        setIsDragging(true);
      }
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current--;
      if (dragCounter.current === 0) {
        setIsDragging(false);
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setIsDragging(false);

      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        const files = Array.from(e.dataTransfer.files);
        handleFileUpload(files);
      }
    };

    document.addEventListener("dragenter", handleDragEnter);
    document.addEventListener("dragleave", handleDragLeave);
    document.addEventListener("dragover", handleDragOver);
    document.addEventListener("drop", handleDrop);

    return () => {
      document.removeEventListener("dragenter", handleDragEnter);
      document.removeEventListener("dragleave", handleDragLeave);
      document.removeEventListener("dragover", handleDragOver);
      document.removeEventListener("drop", handleDrop);
    };
  }, [handleFileUpload]);

  // Handle file selection in dialog
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles(files);
    }
  };

  // Handle upload button click in dialog
  const handleUploadClick = () => {
    if (uploadType === "file" && selectedFiles.length > 0) {
      handleFileUpload(selectedFiles);
    } else if (uploadType === "text") {
      handleTextUpload();
    }
  };

  // Remove an upload item
  const removeUpload = (id: string) => {
    setUploads(prev => prev.filter(item => item.id !== id));
  };

  // Get file icon based on type
  const getFileIcon = (_fileName: string) => {
    return <FileIcon className="h-4 w-4" />;
  };

  return (
    <>
      {/* Drag and drop overlay */}
      {isDragging && (
        <div className="pointer-events-none fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
          <div className="flex h-full items-center justify-center">
            <div className="rounded-lg border-2 border-dashed border-primary bg-background p-12">
              <Upload className="mx-auto h-16 w-16 animate-pulse text-primary" />
              <p className="mt-4 text-xl font-medium">Drop files to upload</p>
              {currentFolder && currentFolder !== "all" && (
                <p className="mt-2 text-sm text-muted-foreground">Files will be uploaded to "{currentFolder}" folder</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Upload dialog */}
      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Upload Documents</DialogTitle>
            <DialogDescription>
              Upload files or text to your document library.
              {currentFolder && currentFolder !== "all" && (
                <span className="mt-1 block">
                  Content will be uploaded to <strong>"{currentFolder}"</strong> folder.
                </span>
              )}
            </DialogDescription>
          </DialogHeader>

          <Tabs value={uploadType} onValueChange={v => setUploadType(v as "file" | "text")}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="file">File Upload</TabsTrigger>
              <TabsTrigger value="text">Text Upload</TabsTrigger>
            </TabsList>

            <TabsContent value="file" className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="file-upload">Select Files</Label>
                <Input
                  id="file-upload"
                  ref={fileInputRef}
                  type="file"
                  multiple
                  onChange={handleFileSelect}
                  className="cursor-pointer"
                />
              </div>

              {selectedFiles.length > 0 && (
                <div className="space-y-2">
                  <Label>Selected Files ({selectedFiles.length})</Label>
                  <div className="max-h-32 space-y-1 overflow-y-auto rounded border p-2">
                    {selectedFiles.map((file, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm">
                        {getFileIcon(file.name)}
                        <span className="flex-1 truncate">{file.name}</span>
                        <span className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} KB</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="text" className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="text-content">Text Content</Label>
                <Textarea
                  id="text-content"
                  value={textContent}
                  onChange={e => setTextContent(e.target.value)}
                  placeholder="Enter or paste your text here..."
                  className="min-h-32"
                />
              </div>
            </TabsContent>
          </Tabs>

          {/* Advanced options */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">Advanced Options</Label>
              <Button variant="ghost" size="sm" onClick={() => setShowAdvanced(!showAdvanced)}>
                {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </div>

            {showAdvanced && (
              <div className="space-y-4 rounded-lg border p-4">
                <div className="grid gap-2">
                  <Label htmlFor="metadata">Metadata (JSON)</Label>
                  <Textarea
                    id="metadata"
                    value={metadata}
                    onChange={e => setMetadata(e.target.value)}
                    placeholder='{"key": "value"}'
                    className="font-mono text-sm"
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="rules">Rules (JSON Array)</Label>
                  <Textarea
                    id="rules"
                    value={rules}
                    onChange={e => setRules(e.target.value)}
                    placeholder='["rule1", "rule2"]'
                    className="font-mono text-sm"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="use-colpali"
                    checked={useColpali}
                    onCheckedChange={checked => setUseColpali(checked === true)}
                  />
                  <Label
                    htmlFor="use-colpali"
                    className="cursor-pointer text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    Use Colpali (recommended for PDFs and images)
                  </Label>
                </div>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowUploadDialog(false);
                resetForm();
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleUploadClick}
              disabled={
                (uploadType === "file" && selectedFiles.length === 0) || (uploadType === "text" && !textContent.trim())
              }
            >
              Upload
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Upload progress cards */}
      {uploads.length > 0 && (
        <div className="fixed bottom-4 right-4 z-50 max-h-96 w-96 space-y-2 overflow-y-auto">
          {uploads.map(upload => (
            <Card key={upload.id} className="shadow-lg">
              <CardContent className="p-3">
                <div className="flex items-start gap-3">
                  {/* Status icon */}
                  <div className="mt-1">
                    {upload.status === "uploading" && <Loader2 className="h-4 w-4 animate-spin text-blue-500" />}
                    {upload.status === "uploaded" && <CheckCircle className="h-4 w-4 text-green-500" />}
                    {upload.status === "processing" && <Loader2 className="h-4 w-4 animate-spin text-orange-500" />}
                    {upload.status === "complete" && <CheckCircle className="h-4 w-4 text-green-600" />}
                    {upload.status === "error" && <AlertCircle className="h-4 w-4 text-destructive" />}
                    {upload.status === "pending" && <Upload className="h-4 w-4 text-muted-foreground" />}
                  </div>

                  {/* File info and progress */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <p className="truncate pr-2 text-sm font-medium">{upload.file.name}</p>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="-mr-1 h-6 w-6"
                        onClick={() => removeUpload(upload.id)}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>

                    <p className="mt-1 text-xs text-muted-foreground">
                      {upload.status === "uploading" && `Uploading... ${Math.round(upload.progress)}%`}
                      {upload.status === "uploaded" && "Uploaded successfully"}
                      {upload.status === "error" && (upload.error || "Upload failed")}
                      {upload.status === "pending" && "Waiting..."}
                    </p>

                    {/* Upload progress bar - only during upload */}
                    {upload.status === "uploading" && <Progress value={upload.progress} className="mt-2 h-2" />}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
