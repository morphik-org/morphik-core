"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Upload } from "lucide-react";
import { showAlert, removeAlert } from "@/components/ui/alert-system";
import DocumentList from "./DocumentList";
import DocumentDetail from "./DocumentDetail";
import FolderList from "./FolderList";
import { UploadDialog, useUploadDialog } from "./UploadDialog";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import DeleteConfirmationModal from "./DeleteConfirmationModal";

import { Document, Folder, FolderSummary } from "@/components/types";

// Custom hook for drag and drop functionality
function useDragAndDrop({ onDrop, disabled = false }: { onDrop: (files: File[]) => void; disabled?: boolean }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      if (disabled) return;
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    },
    [disabled]
  );

  const handleDragEnter = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      if (disabled) return;
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    },
    [disabled]
  );

  const handleDragLeave = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      if (disabled) return;
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
    },
    [disabled]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      if (disabled) return;
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        onDrop(files);
      }
    },
    [disabled, onDrop]
  );

  return {
    isDragging,
    dragHandlers: {
      onDragOver: handleDragOver,
      onDragEnter: handleDragEnter,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop,
    },
  };
}

interface DocumentsSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialFolder?: string | null;
  setSidebarCollapsed?: (collapsed: boolean) => void;

  // Callback props provided by parent
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderClick?: (folderName: string | null) => void;
  onFolderCreate?: (folderName: string) => void;
  onRefresh?: () => void;
  onViewInPDFViewer?: (documentId: string) => void; // Add PDF viewer navigation
}

// Debug render counter
let renderCount = 0;

const DocumentsSection: React.FC<DocumentsSectionProps> = ({
  apiBaseUrl,
  authToken,
  initialFolder = null,
  setSidebarCollapsed,
  // Destructure new props
  onDocumentUpload,
  onDocumentDelete,
  onDocumentClick,
  onFolderClick,
  onFolderCreate,
  onRefresh,
  onViewInPDFViewer,
}) => {
  // Increment render counter for debugging
  renderCount++;
  console.log(`DocumentsSection rendered: #${renderCount}`);
  // Ensure apiBaseUrl is correctly formatted, especially for localhost
  const effectiveApiUrl = React.useMemo(() => {
    console.log("DocumentsSection: Input apiBaseUrl:", apiBaseUrl);
    // Check if it's a localhost URL and ensure it has the right format
    if (apiBaseUrl.includes("localhost") || apiBaseUrl.includes("127.0.0.1")) {
      if (!apiBaseUrl.includes("http")) {
        return `http://${apiBaseUrl}`;
      }
    }
    return apiBaseUrl;
  }, [apiBaseUrl]);

  // State for documents and folders
  const [documents, setDocuments] = useState<Document[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(initialFolder);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [foldersLoading, setFoldersLoading] = useState(false);
  // Use ref to track if this is the initial mount
  const isInitialMount = useRef(true);
  // State for delete confirmation modal
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<string | null>(null); // For single delete: stores ID
  const [itemsToDeleteCount, setItemsToDeleteCount] = useState<number>(0); // For multiple delete: stores count

  // Upload dialog state from custom hook
  const uploadDialogState = useUploadDialog();
  // Extract only the state variables we actually use in this component
  const { showUploadDialog, setShowUploadDialog, metadata, rules, useColpali, resetUploadDialog } = uploadDialogState;

  // Initialize drag and drop
  const { isDragging, dragHandlers } = useDragAndDrop({
    onDrop: files => {
      // Only allow drag and drop when inside a folder
      if (selectedFolder && selectedFolder !== null) {
        handleBatchFileUpload(files, true);
      }
    },
    disabled: !selectedFolder || selectedFolder === null,
  });

  // Cache for folder details (document_ids) to avoid duplicate network calls
  const folderDetailsCache = useRef<Record<string, string[]>>({});

  // No need for a separate header function, use authToken directly

  // Fetch all documents, optionally filtered by folder
  const fetchDocuments = useCallback(
    async (source: string = "unknown") => {
      console.log(`fetchDocuments called from: ${source}, selectedFolder: ${selectedFolder}`);
      // Ensure API URL is valid before proceeding
      if (!effectiveApiUrl) {
        console.error("fetchDocuments: No valid API URL available.");
        setLoading(false);
        return;
      }

      // Immediately clear documents and set loading state if selectedFolder is null (folder grid view)
      if (selectedFolder === null) {
        console.log("fetchDocuments: No folder selected, clearing documents.");
        setDocuments([]);
        setLoading(false);
        return;
      }

      // Set loading state only for initial load or when explicitly changing folders
      if (documents.length === 0 || source === "folders loaded or selectedFolder changed") {
        setLoading(true);
      }

      try {
        let documentsToFetch: Document[] = [];

        if (selectedFolder === "all") {
          // Fetch all documents for the "all" view
          console.log("fetchDocuments: Fetching all documents");
          const response = await fetch(`${effectiveApiUrl}/documents`, {
            method: "POST", // Assuming POST is correct for fetching all
            headers: {
              "Content-Type": "application/json",
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
            body: JSON.stringify({}), // Empty body for all documents
          });
          if (!response.ok) {
            throw new Error(`Failed to fetch all documents: ${response.statusText}`);
          }
          documentsToFetch = await response.json();
          console.log(`fetchDocuments: Fetched ${documentsToFetch.length} total documents`);
        } else {
          // Fetch documents for a specific folder
          console.log(`fetchDocuments: Fetching documents for folder: ${selectedFolder}`);
          const targetFolder = folders.find(folder => folder.name === selectedFolder);

          if (!targetFolder) {
            console.log(`fetchDocuments: Folder ${selectedFolder} not found in summary list.`);
            documentsToFetch = [];
          } else {
            // Resolve the document_ids – first from cache, otherwise fetch detail
            let docIds = folderDetailsCache.current[targetFolder.id];

            if (!docIds) {
              console.log(`fetchDocuments: Fetching folder details for id=${targetFolder.id}`);
              const detailResp = await fetch(`${effectiveApiUrl}/folders/${targetFolder.id}`, {
                headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
              });
              if (!detailResp.ok) {
                throw new Error(`Failed to fetch folder detail: ${detailResp.statusText}`);
              }
              const detail: Folder = await detailResp.json();
              docIds = Array.isArray(detail.document_ids) ? detail.document_ids : [];
              // Cache for future use
              folderDetailsCache.current[targetFolder.id] = docIds;
            }

            if (docIds.length === 0) {
              console.log(`fetchDocuments: Folder ${selectedFolder} is empty.`);
              documentsToFetch = [];
            } else {
              console.log(`fetchDocuments: Fetching ${docIds.length} documents via batch API`);
              const response = await fetch(`${effectiveApiUrl}/batch/documents`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                },
                body: JSON.stringify({ document_ids: docIds }),
              });
              if (!response.ok) {
                throw new Error(`Failed to fetch batch documents: ${response.statusText}`);
              }
              documentsToFetch = await response.json();
              console.log(`fetchDocuments: Fetched details for ${documentsToFetch.length} documents`);
            }
          }
        }

        // Process fetched documents (add status if needed)
        const processedData = documentsToFetch.map((doc: Document) => {
          if (!doc.system_metadata) {
            doc.system_metadata = {};
          }
          if (!doc.system_metadata.status && doc.system_metadata.folder_name) {
            doc.system_metadata.status = "processing";
          }
          return doc;
        });

        // Update state
        setDocuments(processedData);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
        console.error(`Error in fetchDocuments (${source}): ${errorMsg}`);
        showAlert(errorMsg, {
          type: "error",
          title: "Error Fetching Documents",
          duration: 5000,
        });
        // Clear documents on error to avoid showing stale/incorrect data
        setDocuments([]);
      } finally {
        // Always ensure loading state is turned off
        setLoading(false);
      }
      // Dependencies: URL, auth, selected folder, and the folder list itself
    },
    [effectiveApiUrl, authToken, selectedFolder, folders, documents.length]
  );

  // Fetch all folders
  const fetchFolders = useCallback(async () => {
    console.log("fetchFolders called");
    setFoldersLoading(true);
    try {
      const response = await fetch(`${effectiveApiUrl}/folders/summary`, {
        method: "GET",
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch folders: ${response.statusText}`);
      }
      const data = (await response.json()) as FolderSummary[];
      console.log(`Fetched ${data.length} folders`);
      setFolders(data);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      console.error(`Folder fetch error: ${errorMsg}`);
      showAlert(errorMsg, {
        type: "error",
        title: "Error",
        duration: 5000,
      });
    } finally {
      setFoldersLoading(false);
    }
  }, [effectiveApiUrl, authToken]);

  // Fetch folders initially
  useEffect(() => {
    console.log("DocumentsSection: Initial folder fetch");
    fetchFolders();
  }, [fetchFolders]);

  // Fetch documents when folders are loaded or selectedFolder changes
  useEffect(() => {
    const effectSource = "folders loaded or selectedFolder changed";
    console.log(
      `Effect triggered: ${effectSource}, foldersLoading: ${foldersLoading}, folders count: ${folders.length}, selectedFolder: ${selectedFolder}`
    );

    // Guard against running when folders are still loading
    if (foldersLoading) {
      console.log(`Effect (${effectSource}): Folders still loading, skipping.`);
      return;
    }

    // Handle the case where there are no folders at all
    if (folders.length === 0 && selectedFolder === null) {
      console.log(`Effect (${effectSource}): No folders found, clearing documents and stopping loading.`);
      setDocuments([]);
      setLoading(false); // Ensure loading is off
      isInitialMount.current = false;
      return;
    }

    // Proceed if folders are loaded
    if (folders.length >= 0) {
      // Check >= 0 to handle empty folders array correctly
      // Avoid fetching documents on initial mount if selectedFolder is null
      // unless initialFolder was specified
      if (isInitialMount.current && selectedFolder === null && !initialFolder) {
        console.log(`Effect (${effectSource}): Initial mount with no folder selected, skipping document fetch`);
        isInitialMount.current = false;
        // Ensure loading is false if we skip fetching
        setLoading(false);
        return;
      }

      // If we reach here, we intend to fetch documents
      console.log(`Effect (${effectSource}): Preparing to fetch documents for folder: ${selectedFolder}`);

      // Wrap the async operation
      const fetchWrapper = async () => {
        // Explicitly set loading true *before* the async call within this effect's scope
        // Note: fetchDocuments might also set this, but we ensure it's set here.
        setLoading(true);
        try {
          await fetchDocuments(effectSource);
          // If fetchDocuments completes successfully, it will set loading = false in its finally block.
          // No need to set it here again in the try block.
          console.log(`Effect (${effectSource}): fetchDocuments call completed.`);
        } catch (error) {
          // Catch potential errors *from* the await fetchDocuments call itself, though
          // fetchDocuments has internal handling. This is an extra safeguard.
          console.error(`Effect (${effectSource}): Error occurred during fetchDocuments call:`, error);
          showAlert(`Error updating documents: ${error instanceof Error ? error.message : "Unknown error"}`, {
            type: "error",
          });
          // Ensure loading is turned off even if fetchDocuments had an issue before its finally.
          setLoading(false);
        } finally {
          // **User Request:** Explicitly set loading to false within the effect's finally block.
          // This acts as a safeguard, ensuring loading is false after the attempt,
          // regardless of fetchDocuments' internal state management.
          console.log(`Effect (${effectSource}): Finally block reached, ensuring loading is false.`);
          setLoading(false);
          isInitialMount.current = false; // Mark initial mount as complete here
        }
      };

      fetchWrapper();
    } else {
      console.log(`Effect (${effectSource}): Condition not met (folders length < 0 ?), should not happen.`);
      setLoading(false); // Fallback
    }
  }, [foldersLoading, folders, selectedFolder, fetchDocuments, initialFolder]); // Keep fetchDocuments dependency

  // ---------------------------------------------------------------------
  // Fine-grained polling – update status of documents that are processing
  // ---------------------------------------------------------------------
  useEffect(() => {
    // Identify docs still processing or recently failed (to capture error info)
    const processingDocs = documents.filter(
      doc =>
        doc.system_metadata?.status === "processing" ||
        (doc.system_metadata?.status === "failed" && !doc.system_metadata?.error)
    );

    // If none, skip polling
    if (processingDocs.length === 0) {
      return;
    }

    const intervalId = setInterval(async () => {
      try {
        // Fetch status for each processing document in parallel
        const updates = await Promise.all(
          processingDocs.map(async doc => {
            try {
              const resp = await fetch(`${effectiveApiUrl}/documents/${doc.external_id}/status`, {
                method: "GET",
                headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
              });
              if (!resp.ok) {
                throw new Error(resp.statusText);
              }
              const data = await resp.json();
              return {
                id: data.document_id as string,
                status: data.status as string,
                updatedAt: data.updated_at as string | undefined,
                error: data.error as string | undefined,
              };
            } catch (err) {
              console.error("Status poll error for", doc.external_id, err);
              return null;
            }
          })
        );

        // Update documents state with new statuses
        setDocuments(prevDocs =>
          prevDocs.map(d => {
            const upd = updates.find(u => u && u.id === d.external_id);
            if (upd && upd.status && (upd.status !== d.system_metadata?.status || upd.error)) {
              return {
                ...d,
                system_metadata: {
                  ...d.system_metadata,
                  status: upd.status,
                  updated_at: upd.updatedAt ?? d.system_metadata?.updated_at,
                  ...(upd.error && { error: upd.error }),
                },
              } as Document;
            }
            return d;
          })
        );
      } catch (err) {
        console.error("Error polling document statuses:", err);
      }
    }, 5000); // Poll every 5 seconds

    // Cleanup on unmount or when processingDocs changes
    return () => clearInterval(intervalId);
  }, [documents, effectiveApiUrl, authToken]);

  // Collapse sidebar when a folder is selected
  useEffect(() => {
    if (selectedFolder !== null && setSidebarCollapsed) {
      setSidebarCollapsed(true);
    } else if (setSidebarCollapsed) {
      setSidebarCollapsed(false);
    }
  }, [selectedFolder, setSidebarCollapsed]);

  // Fetch a specific document by ID
  const fetchDocument = async (documentId: string) => {
    try {
      const url = `${effectiveApiUrl}/documents/${documentId}`;
      console.log("DocumentsSection: Fetching document detail from:", url);

      // Use non-blocking fetch to avoid locking the UI
      fetch(url, {
        method: "GET",
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Failed to fetch document: ${response.statusText}`);
          }
          return response.json();
        })
        .then(data => {
          console.log(`Fetched document details for ID: ${documentId}`);

          // Ensure document has a valid status in system_metadata
          if (!data.system_metadata) {
            data.system_metadata = {};
          }

          // If status is missing and we have a newly uploaded document, it should be "processing"
          if (!data.system_metadata.status && data.system_metadata.folder_name) {
            data.system_metadata.status = "processing";
          }

          setSelectedDocument(data);
        })
        .catch(err => {
          const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
          console.error(`Error fetching document details: ${errorMsg}`);
          showAlert(`Error fetching document: ${errorMsg}`, {
            type: "error",
            duration: 5000,
          });
        });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      console.error(`Error in fetchDocument: ${errorMsg}`);
      showAlert(`Error: ${errorMsg}`, {
        type: "error",
        duration: 5000,
      });
    }
  };

  // Handle document click
  const handleDocumentClick = (document: Document) => {
    // Invoke callback prop before fetching
    const docName = document.filename || document.external_id; // Use filename, fallback to ID
    console.log(`handleDocumentClick: Calling onDocumentClick with '${docName}'`);
    onDocumentClick?.(docName);
    fetchDocument(document.external_id);
  };

  // Helper function for document deletion API call
  const deleteDocumentApi = async (documentId: string) => {
    const response = await fetch(`${effectiveApiUrl}/documents/${documentId}`, {
      method: "DELETE",
      headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
    });

    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`);
    }

    return response;
  };

  // Handle single document deletion
  const handleDeleteDocument = async (documentId: string) => {
    setItemToDelete(documentId);
    setItemsToDeleteCount(0); // Ensure this is 0 for single delete scenario
    setShowDeleteModal(true);
  };

  // Handle document download
  const handleDownloadDocument = async (documentId: string) => {
    try {
      // Get the download URL for this document
      const downloadUrlEndpoint = `${effectiveApiUrl}/documents/${documentId}/download_url`;
      console.log("Fetching download URL from:", downloadUrlEndpoint);

      const downloadUrlResponse = await fetch(downloadUrlEndpoint, {
        headers: {
          ...(authToken && { Authorization: `Bearer ${authToken}` }),
        },
      });

      if (!downloadUrlResponse.ok) {
        console.error("Download URL request failed:", downloadUrlResponse.status, downloadUrlResponse.statusText);
        throw new Error("Failed to get download URL");
      }

      const downloadData = await downloadUrlResponse.json();
      console.log("Download URL response:", downloadData);

      let downloadUrl = downloadData.download_url;

      // Check if it's a local file URL (file://) which browsers can't access
      if (downloadUrl.startsWith("file://")) {
        console.log("Detected file:// URL, switching to direct file endpoint");
        // Use our direct file endpoint instead for local storage
        downloadUrl = `${effectiveApiUrl}/documents/${documentId}/file`;
      }

      console.log("Final download URL:", downloadUrl);

      // Create a temporary link to trigger download
      const link = window.document.createElement("a");
      link.href = downloadUrl;

      // Get the document name for the download
      const docToDownload = documents.find(doc => doc.external_id === documentId);
      if (docToDownload?.filename) {
        link.download = docToDownload.filename;
      }

      window.document.body.appendChild(link);
      link.click();
      window.document.body.removeChild(link);

      console.log("Download initiated successfully");
    } catch (error) {
      console.error("Error downloading document:", error);
      showAlert("Error downloading document. Please try again.", {
        type: "error",
        duration: 3000,
      });
    }
  };

  const confirmDeleteSingleDocument = async () => {
    if (!itemToDelete) return;

    try {
      // Find document name before deleting (for callback)
      const docToDelete = documents.find(doc => doc.external_id === itemToDelete);
      const docName = docToDelete?.filename || itemToDelete; // Use filename, fallback to ID
      console.log(`confirmDeleteSingleDocument: Calling onDocumentDelete with '${docName}'`);
      onDocumentDelete?.(docName); // Invoke callback

      setLoading(true);
      setShowDeleteModal(false); // Close modal before starting deletion

      console.log("DocumentsSection: Deleting document:", itemToDelete);

      await deleteDocumentApi(itemToDelete);

      // Clear selected document if it was the one deleted
      if (selectedDocument?.external_id === itemToDelete) {
        setSelectedDocument(null);
      }

      // Refresh folders first, then documents
      await fetchFolders();
      await fetchDocuments(); // This will be triggered by folder fetch in useEffect

      // Show success message
      showAlert("Document deleted successfully", {
        type: "success",
        duration: 3000,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      showAlert(errorMsg, {
        type: "error",
        title: "Delete Failed",
        duration: 5000,
      });
      // Also remove the progress alert if there was an error
      removeAlert("delete-multiple-progress"); // Though not used for single, good to have
    } finally {
      setLoading(false);
      setItemToDelete(null);
    }
  };

  // Handle multiple document deletion
  const handleDeleteMultipleDocuments = async () => {
    if (selectedDocuments.length === 0) return;
    setItemsToDeleteCount(selectedDocuments.length);
    setItemToDelete(null); // Ensure this is null for multiple delete scenario
    setShowDeleteModal(true);
  };

  const confirmDeleteMultipleDocuments = async () => {
    if (selectedDocuments.length === 0) return;

    try {
      // Invoke callback for each selected document BEFORE deleting
      selectedDocuments.forEach(docId => {
        const docToDelete = documents.find(doc => doc.external_id === docId);
        const docName = docToDelete?.filename || docId; // Use filename, fallback to ID
        console.log(`confirmDeleteMultipleDocuments: Calling onDocumentDelete with '${docName}'`);
        onDocumentDelete?.(docName);
      });

      setLoading(true);
      setShowDeleteModal(false); // Close modal before starting deletion

      // Show initial alert for deletion progress
      const alertId = "delete-multiple-progress";
      showAlert(`Deleting ${selectedDocuments.length} documents...`, {
        type: "info",
        dismissible: false,
        id: alertId,
      });

      console.log("DocumentsSection: Deleting multiple documents:", selectedDocuments);

      // Perform deletions in parallel
      const results = await Promise.all(selectedDocuments.map(docId => deleteDocumentApi(docId)));

      // Check if any deletion failed
      const failedCount = results.filter(res => !res.ok).length;

      // Clear selected document if it was among deleted ones
      if (selectedDocument && selectedDocuments.includes(selectedDocument.external_id)) {
        setSelectedDocument(null);
      }

      // Clear selection
      setSelectedDocuments([]);

      // Refresh folders first, then documents
      await fetchFolders();
      await fetchDocuments(); // This will be triggered by folder fetch in useEffect

      // Remove progress alert
      removeAlert(alertId);

      // Show final result alert
      if (failedCount > 0) {
        showAlert(`Deleted ${selectedDocuments.length - failedCount} documents. ${failedCount} deletions failed.`, {
          type: "warning",
          duration: 4000,
        });
      } else {
        showAlert(`Successfully deleted ${selectedDocuments.length} documents`, {
          type: "success",
          duration: 3000,
        });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      showAlert(errorMsg, {
        type: "error",
        title: "Delete Failed",
        duration: 5000,
      });

      // Also remove the progress alert if there was an error
      removeAlert("delete-multiple-progress");
    } finally {
      setLoading(false);
      setSelectedDocuments([]); // Clear selection after attempting deletion
      setItemsToDeleteCount(0);
    }
  };

  // Handle checkbox change (wrapper function for use with shadcn checkbox)
  const handleCheckboxChange = (checked: boolean | "indeterminate", docId: string) => {
    setSelectedDocuments(prev => {
      if (checked === true && !prev.includes(docId)) {
        return [...prev, docId];
      } else if (checked === false && prev.includes(docId)) {
        return prev.filter(id => id !== docId);
      }
      return prev;
    });
  };

  // Helper function to get "indeterminate" state for select all checkbox
  const getSelectAllState = () => {
    if (selectedDocuments.length === 0) return false;
    if (selectedDocuments.length === documents.length) return true;
    return "indeterminate";
  };

  // Handle file upload
  const handleFileUpload = async (file: File | null) => {
    if (!file) {
      showAlert("Please select a file to upload", {
        type: "error",
        duration: 3000,
      });
      return;
    }

    // Close dialog and update upload count using alert system
    setShowUploadDialog(false);
    const uploadId = "upload-progress";
    showAlert(`Uploading 1 file...`, {
      type: "upload",
      dismissible: false,
      id: uploadId,
    });

    // Save file reference before we reset the form
    const fileToUploadRef = file;
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;

    // Reset form
    resetUploadDialog();

    try {
      const formData = new FormData();
      formData.append("file", fileToUploadRef);
      formData.append("metadata", metadataRef);
      formData.append("rules", rulesRef);
      formData.append("use_colpali", String(useColpaliRef));

      // If we're in a specific folder (not "all" documents), add the folder_name to form data
      if (selectedFolder && selectedFolder !== "all") {
        try {
          // Parse metadata to validate it's proper JSON, but don't modify it
          JSON.parse(metadataRef || "{}");

          // The API expects folder_name as a direct Form parameter
          // This will be used by document_service._ensure_folder_exists()
          formData.set("metadata", metadataRef);
          formData.append("folder_name", selectedFolder);

          // Log for debugging
          console.log(`Adding file to folder: ${selectedFolder} as form field`);
        } catch (e) {
          console.error("Error parsing metadata:", e);
          formData.set("metadata", metadataRef);
          formData.append("folder_name", selectedFolder);
        }
      }

      const url = `${effectiveApiUrl}/ingest/file`;

      // Non-blocking fetch
      fetch(url, {
        method: "POST",
        headers: {
          Authorization: authToken ? `Bearer ${authToken}` : "",
        },
        body: formData,
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Failed to upload: ${response.statusText}`);
          }
          return response.json();
        })
        .then(newDocument => {
          // Invoke callback on success
          console.log(
            `handleFileUpload: Calling onDocumentUpload with '${fileToUploadRef.name}', size: ${fileToUploadRef.size}`
          );
          onDocumentUpload?.(fileToUploadRef.name, fileToUploadRef.size);

          // Log processing status of uploaded document
          if (newDocument && newDocument.system_metadata && newDocument.system_metadata.status === "processing") {
            console.log(`Document ${newDocument.external_id} is in processing status`);
            // No longer need to track processing documents for polling
          }

          // Force a fresh refresh after upload
          const refreshAfterUpload = async () => {
            try {
              console.log("Performing fresh refresh after upload (file)");
              // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
              await fetchFolders();
            } catch (err) {
              console.error("Error refreshing after file upload:", err);
            }
          };

          // Execute the refresh
          refreshAfterUpload();

          // Show success message and remove upload progress
          showAlert(`File uploaded successfully!`, {
            type: "success",
            duration: 3000,
          });

          // Remove the upload alert
          removeAlert("upload-progress");
        })
        .catch(err => {
          const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
          const errorMsg = `Error uploading ${fileToUploadRef.name}: ${errorMessage}`;

          // Show error alert and remove upload progress
          showAlert(errorMsg, {
            type: "error",
            title: "Upload Failed",
            duration: 5000,
          });

          // Remove the upload alert
          removeAlert("upload-progress");
        });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
      const errorMsg = `Error uploading ${fileToUploadRef.name}: ${errorMessage}`;

      // Show error alert
      showAlert(errorMsg, {
        type: "error",
        title: "Upload Failed",
        duration: 5000,
      });

      // Remove the upload progress alert
      removeAlert("upload-progress");
    }
  };

  // Handle batch file upload
  const handleBatchFileUpload = async (files: File[], fromDragAndDrop: boolean = false) => {
    if (files.length === 0) {
      showAlert("Please select files to upload", {
        type: "error",
        duration: 3000,
      });
      return;
    }

    // Close dialog if it's open (but not if drag and drop)
    if (!fromDragAndDrop) {
      setShowUploadDialog(false);
    }

    const fileCount = files.length;
    const uploadId = "batch-upload-progress";
    showAlert(`Uploading ${fileCount} files...`, {
      type: "upload",
      dismissible: false,
      id: uploadId,
    });

    // Save form data locally
    const batchFilesRef = [...files];
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;

    // Only reset form if not from drag and drop
    if (!fromDragAndDrop) {
      resetUploadDialog();
    }

    try {
      const formData = new FormData();

      // Append each file to the formData with the same field name
      batchFilesRef.forEach(file => {
        formData.append("files", file);
      });

      // Add metadata to all cases
      formData.append("metadata", metadataRef);

      // If we're in a specific folder (not "all" documents), add the folder_name as a separate field
      if (selectedFolder && selectedFolder !== "all") {
        // The API expects folder_name directly, not ID
        formData.append("folder_name", selectedFolder);

        // Log for debugging
        console.log(`Adding batch files to folder: ${selectedFolder} as form field`);
      }

      formData.append("rules", rulesRef);
      formData.append("parallel", "true");
      formData.append("use_colpali", String(useColpaliRef));

      const url = `${effectiveApiUrl}/ingest/files`;

      // Non-blocking fetch
      fetch(url, {
        method: "POST",
        headers: {
          Authorization: authToken ? `Bearer ${authToken}` : "",
        },
        body: formData,
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Failed to upload: ${response.statusText}`);
          }
          return response.json();
        })
        .then(result => {
          // Invoke callback on success
          console.log(
            `handleBatchFileUpload: Calling onDocumentUpload with '${batchFilesRef[0].name}', size: ${batchFilesRef[0].size} (for first file in batch)`
          );
          onDocumentUpload?.(batchFilesRef[0].name, batchFilesRef[0].size);

          // Log processing status of uploaded documents
          if (result && result.document_ids && result.document_ids.length > 0) {
            console.log(`${result.document_ids.length} documents are in processing status`);
            // No need for polling, just wait for manual refresh
          }

          // Force a fresh refresh after upload
          const refreshAfterUpload = async () => {
            try {
              console.log("Performing fresh refresh after upload (batch)");
              // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
              await fetchFolders();
            } catch (err) {
              console.error("Error refreshing after batch upload:", err);
            }
          };

          // Execute the refresh
          refreshAfterUpload();

          // If there are errors, show them in the error alert
          if (result.errors && result.errors.length > 0) {
            const errorMsg = `${result.errors.length} of ${fileCount} files failed to upload`;

            showAlert(errorMsg, {
              type: "error",
              title: "Upload Partially Failed",
              duration: 5000,
            });
          } else {
            // Show success message
            showAlert(`${fileCount} files uploaded successfully!`, {
              type: "success",
              duration: 3000,
            });
          }

          // Remove the upload alert
          removeAlert("batch-upload-progress");
        })
        .catch(err => {
          const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
          const errorMsg = `Error uploading files: ${errorMessage}`;

          // Show error alert
          showAlert(errorMsg, {
            type: "error",
            title: "Upload Failed",
            duration: 5000,
          });

          // Remove the upload alert
          removeAlert("batch-upload-progress");
        });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
      const errorMsg = `Error uploading files: ${errorMessage}`;

      // Show error alert
      showAlert(errorMsg, {
        type: "error",
        title: "Upload Failed",
        duration: 5000,
      });

      // Remove the upload progress alert
      removeAlert("batch-upload-progress");
    }
  };

  // Handle text upload
  const handleTextUpload = async (text: string, meta: string, rulesText: string, useColpaliFlag: boolean) => {
    if (!text.trim()) {
      showAlert("Please enter text content", {
        type: "error",
        duration: 3000,
      });
      return;
    }

    // Close dialog and update upload count using alert system
    setShowUploadDialog(false);
    const uploadId = "text-upload-progress";
    showAlert(`Uploading text document...`, {
      type: "upload",
      dismissible: false,
      id: uploadId,
    });

    // Save content before resetting
    const textContentRef = text;
    let metadataObj = {};
    let folderToUse = null;

    try {
      metadataObj = JSON.parse(meta || "{}");

      // If we're in a specific folder (not "all" documents), set folder variable
      if (selectedFolder && selectedFolder !== "all") {
        // The API expects the folder name directly
        folderToUse = selectedFolder;
        // Log for debugging
        console.log(`Will add text document to folder: ${selectedFolder}`);
      }
    } catch (e) {
      console.error("Error parsing metadata JSON:", e);
    }

    const rulesRef = rulesText;
    const useColpaliRef = useColpaliFlag;

    // Reset form immediately
    resetUploadDialog();

    try {
      // Non-blocking fetch with explicit use_colpali parameter
      const url = `${effectiveApiUrl}/ingest/text`;

      fetch(url, {
        method: "POST",
        headers: {
          Authorization: authToken ? `Bearer ${authToken}` : "",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: textContentRef,
          metadata: metadataObj,
          rules: JSON.parse(rulesRef || "[]"),
          folder_name: folderToUse,
          use_colpali: useColpaliRef,
        }),
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Failed to upload: ${response.statusText}`);
          }
          return response.json();
        })
        .then(newDocument => {
          // Currently skipping callback for text uploads until an explicit event is defined
          console.log(`handleTextUpload: Text uploaded successfully (tracking skipped).`);

          // Log processing status of uploaded document
          if (newDocument && newDocument.system_metadata && newDocument.system_metadata.status === "processing") {
            console.log(`Document ${newDocument.external_id} is in processing status`);
            // No longer need to track processing documents for polling
          }

          // Force a fresh refresh after upload
          const refreshAfterUpload = async () => {
            try {
              console.log("Performing fresh refresh after upload (text)");
              // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
              await fetchFolders();
            } catch (err) {
              console.error("Error refreshing after text upload:", err);
            }
          };

          // Execute the refresh
          refreshAfterUpload();

          // Show success message
          showAlert(`Text document uploaded successfully!`, {
            type: "success",
            duration: 3000,
          });

          // Remove the upload alert
          removeAlert("text-upload-progress");
        })
        .catch(err => {
          const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
          const errorMsg = `Error uploading text: ${errorMessage}`;

          // Show error alert
          showAlert(errorMsg, {
            type: "error",
            title: "Upload Failed",
            duration: 5000,
          });

          // Remove the upload alert
          removeAlert("text-upload-progress");
        });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
      const errorMsg = `Error uploading text: ${errorMessage}`;

      // Show error alert
      showAlert(errorMsg, {
        type: "error",
        title: "Upload Failed",
        duration: 5000,
      });

      // Remove the upload progress alert
      removeAlert("text-upload-progress");
    }
  };

  // Function to trigger refresh
  const handleRefresh = () => {
    // Invoke callback
    onRefresh?.();

    setLoading(true);

    // Create a new function to perform a truly fresh fetch
    const performFreshFetch = async () => {
      try {
        // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
        await fetchFolders();

        // Show success message (consider moving this if fetchFolders doesn't guarantee documents are loaded)
        showAlert("Refresh initiated. Data will update shortly.", {
          type: "success",
          duration: 1500,
        });
      } catch (error) {
        console.error("Error during refresh fetchFolders:", error);
        showAlert(`Error refreshing: ${error instanceof Error ? error.message : "Unknown error"}`, {
          type: "error",
          duration: 3000,
        });
      } finally {
        // setLoading(false); // Loading will be handled by fetchDocuments triggered by useEffect
      }
    };

    // Execute the fresh fetch
    performFreshFetch();
  };

  // Wrapper for setSelectedFolder to include callback invocation
  const handleFolderSelect = useCallback(
    (folderName: string | null) => {
      console.log(`handleFolderSelect: Calling onFolderClick with '${folderName}'`);
      onFolderClick?.(folderName);
      setSelectedFolder(folderName);
    },
    [onFolderClick]
  ); // Add setSelectedFolder if its identity matters, but it usually doesn't

  return (
    <div
      className={cn("relative flex h-full flex-1 flex-col p-4", selectedFolder && isDragging ? "drag-active" : "")}
      {...(selectedFolder ? dragHandlers : {})}
    >
      {/* Drag overlay - only visible when dragging files over the folder */}
      {isDragging && selectedFolder && (
        <div className="absolute inset-0 z-50 flex animate-pulse items-center justify-center rounded-lg border-2 border-dashed border-primary bg-primary/10 backdrop-blur-sm">
          <div className="rounded-lg bg-background p-8 text-center shadow-lg">
            <Upload className="mx-auto mb-4 h-12 w-12 text-primary" />
            <h3 className="mb-2 text-xl font-medium">Drop to Upload</h3>
            <p className="text-muted-foreground">
              Files will be added to {selectedFolder === "all" ? "your documents" : `folder "${selectedFolder}"`}
            </p>
          </div>
        </div>
      )}
      {/* Folder view controls - only show when not in a specific folder */}
      {/* No longer needed - controls will be provided in FolderList */}

      {/* Render the FolderList with header at all times when selectedFolder is not null */}
      {selectedFolder !== null && (
        <FolderList
          folders={folders}
          selectedFolder={selectedFolder}
          setSelectedFolder={handleFolderSelect}
          apiBaseUrl={effectiveApiUrl}
          authToken={authToken}
          refreshFolders={fetchFolders}
          loading={foldersLoading}
          refreshAction={handleRefresh}
          selectedDocuments={selectedDocuments}
          handleDeleteMultipleDocuments={handleDeleteMultipleDocuments}
          uploadDialogComponent={
            <UploadDialog
              showUploadDialog={showUploadDialog}
              setShowUploadDialog={setShowUploadDialog}
              loading={loading}
              onFileUpload={handleFileUpload}
              onBatchFileUpload={handleBatchFileUpload}
              onTextUpload={handleTextUpload}
            />
          }
          onFolderCreate={onFolderCreate}
        />
      )}

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={showDeleteModal}
        onClose={() => {
          setShowDeleteModal(false);
          setItemToDelete(null);
          setItemsToDeleteCount(0);
        }}
        onConfirm={itemToDelete ? confirmDeleteSingleDocument : confirmDeleteMultipleDocuments}
        itemName={
          itemToDelete ? documents.find(doc => doc.external_id === itemToDelete)?.filename || itemToDelete : undefined
        }
        itemCount={itemsToDeleteCount > 0 ? itemsToDeleteCount : undefined}
        loading={loading}
      />

      {/* Folder Grid View (selectedFolder is null) */}
      {selectedFolder === null ? (
        <div className="flex flex-1 flex-col gap-4">
          {/* Skeleton for Folder List loading state */}
          {foldersLoading ? (
            <div className="grid grid-cols-1 gap-4 p-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="flex h-32 flex-col items-center justify-center rounded-lg border p-4">
                  <Skeleton className="mb-2 h-8 w-8 rounded-md" />
                  <Skeleton className="h-4 w-20" />
                </div>
              ))}
            </div>
          ) : (
            <FolderList
              folders={folders}
              selectedFolder={selectedFolder}
              setSelectedFolder={handleFolderSelect}
              apiBaseUrl={effectiveApiUrl}
              authToken={authToken}
              refreshFolders={fetchFolders}
              loading={foldersLoading}
              refreshAction={handleRefresh}
              selectedDocuments={selectedDocuments}
              handleDeleteMultipleDocuments={handleDeleteMultipleDocuments}
              uploadDialogComponent={
                <UploadDialog
                  showUploadDialog={showUploadDialog}
                  setShowUploadDialog={setShowUploadDialog}
                  loading={loading}
                  onFileUpload={handleFileUpload}
                  onBatchFileUpload={handleBatchFileUpload}
                  onTextUpload={handleTextUpload}
                />
              }
              onFolderCreate={onFolderCreate}
            />
          )}
        </div>
      ) : (
        <div className="flex flex-1 flex-col gap-4 md:flex-row">
          {/* Left Panel: Document List or Skeleton or Empty State */}
          <div
            className={cn(
              "flex w-full flex-col transition-all duration-300",
              selectedDocument ? "md:w-2/3" : "md:w-full"
            )}
          >
            {loading && documents.length === 0 ? (
              // Initial skeleton only when no docs are yet loaded
              <div className="flex-1 space-y-3 p-4">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-8 w-3/4" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-5/6" />
                <Skeleton className="h-8 w-full" />
              </div>
            ) : documents.length === 0 ? (
              // Empty State (kept as-is)
              <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed py-8 text-center">
                <div>
                  <Upload className="mx-auto mb-2 h-12 w-12 text-muted-foreground" />
                  <p className="text-muted-foreground">Drag and drop files here to upload to this folder.</p>
                  <p className="mt-2 text-xs text-muted-foreground">Or use the upload button in the top right.</p>
                </div>
              </div>
            ) : (
              // Document list with subtle background refresh indicator
              <div className={cn("relative transition-opacity", loading && documents.length > 0 ? "opacity-60" : "")}>
                {/* Tiny corner spinner instead of full overlay */}
                {loading && documents.length > 0 && (
                  <div className="absolute left-2 top-2 z-10 flex items-center">
                    <div className="\\ h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  </div>
                )}

                <DocumentList
                  documents={documents}
                  selectedDocument={selectedDocument}
                  selectedDocuments={selectedDocuments}
                  handleDocumentClick={handleDocumentClick}
                  handleCheckboxChange={handleCheckboxChange}
                  getSelectAllState={getSelectAllState}
                  setSelectedDocuments={setSelectedDocuments}
                  setDocuments={setDocuments}
                  loading={loading}
                  apiBaseUrl={effectiveApiUrl}
                  authToken={authToken}
                  selectedFolder={selectedFolder}
                  onViewInPDFViewer={onViewInPDFViewer}
                  onDownloadDocument={handleDownloadDocument}
                  onDeleteDocument={handleDeleteDocument}
                  folders={folders}
                />
              </div>
            )}
          </div>

          {/* Right Panel: Document Detail (conditionally rendered) */}
          {selectedDocument && (
            <div className="w-full duration-300 animate-in slide-in-from-right md:w-1/3">
              <DocumentDetail
                selectedDocument={selectedDocument}
                handleDeleteDocument={handleDeleteDocument}
                folders={folders}
                apiBaseUrl={effectiveApiUrl}
                authToken={authToken}
                refreshDocuments={fetchDocuments}
                refreshFolders={fetchFolders}
                loading={loading}
                onClose={() => setSelectedDocument(null)}
                onViewInPDFViewer={onViewInPDFViewer}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentsSection;
