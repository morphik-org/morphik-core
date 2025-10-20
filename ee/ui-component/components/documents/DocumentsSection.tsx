"use client";

import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
// import { useDebounce } from '../../lib/hooks/useDebounce'; // Commented for future use
import { Upload, Search, ArrowLeft } from "lucide-react";
import { showAlert, removeAlert } from "@/components/ui/alert-system";
import DocumentList from "./DocumentList";
import DocumentDetail from "./DocumentDetail";
import { UploadDialog, useUploadDialog } from "./UploadDialog";
import { cn } from "../../lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import DeleteConfirmationModal from "./DeleteConfirmationModal";
import { useFolders, clearFoldersCache } from "../../hooks/useFolders";
import { useDocuments, clearDocumentsCache } from "../../hooks/useDocuments";
import { useUnorganizedDocuments, clearUnorganizedDocumentsCache } from "../../hooks/useUnorganizedDocuments";

import { Document, FolderSummary } from "../types";

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

  // Callback props provided by parent
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderClick?: (folderName: string | null) => void;
  onFolderCreate?: (folderName: string) => void;
  onRefresh?: () => void;
  onViewInPDFViewer?: (documentId: string) => void; // Add PDF viewer navigation

  // New props for state management from parent
  allFoldersExpanded?: boolean;
  showNewFolderDialog?: boolean;
  setShowNewFolderDialog?: (show: boolean) => void;
  showUploadDialog?: boolean;
  setShowUploadDialog?: (show: boolean) => void;
  onFoldersUpdate?: (folders: Array<{ id: string; name: string }>) => void;
}

// Helper to generate temporary IDs for optimistic updates
const generateTempId = () => `uploading-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

const DocumentsSection = React.forwardRef<
  {
    handleRefresh: () => void;
    handleDeleteMultipleDocuments: () => void;
    selectedDocuments: string[];
  },
  DocumentsSectionProps
>(
  (
    {
      apiBaseUrl,
      authToken,
      initialFolder = null,
      // Destructure new props
      onDocumentUpload,
      onDocumentDelete,
      onDocumentClick,
      onFolderClick,
      onFolderCreate,
      onRefresh,
      onViewInPDFViewer,
      // New state props
      allFoldersExpanded: allFoldersExpandedProp = false,
      showNewFolderDialog: showNewFolderDialogProp,
      setShowNewFolderDialog: setShowNewFolderDialogProp,
      showUploadDialog: showUploadDialogProp,
      setShowUploadDialog: setShowUploadDialogProp,
      onFoldersUpdate,
    },
    ref
  ) => {
    // Ensure apiBaseUrl is correctly formatted
    const effectiveApiUrl = React.useMemo(() => {
      return apiBaseUrl;
    }, [apiBaseUrl]);

    // State for selected folder and documents
    const [selectedFolder, setSelectedFolder] = useState<string | null>(initialFolder);
    const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
    const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
    // Use prop value if provided, otherwise manage locally
    const [allFoldersExpandedLocal] = useState(false);
    const allFoldersExpanded = allFoldersExpandedProp !== undefined ? allFoldersExpandedProp : allFoldersExpandedLocal;

    // Sync selectedFolder with initialFolder prop changes
    useEffect(() => {
      setSelectedFolder(initialFolder);
    }, [initialFolder]);

    // Search state for folder view (when selectedFolder !== null)
    const [folderSearchQuery, setFolderSearchQuery] = useState("");

    // Reset folder search whenever the selected folder changes
    useEffect(() => {
      setFolderSearchQuery("");
    }, [selectedFolder]);

    // Use cached hooks for folders and documents
    const {
      folders,
      loading: foldersLoading,
      refresh: refreshFolders,
    } = useFolders({
      apiBaseUrl: effectiveApiUrl,
      authToken,
    });

    const {
      documents,
      loading: documentsLoading,
      loadingMore: documentsLoadingMore,
      refresh: refreshDocuments,
      goToPage: goToDocumentsPage,
      setPageSize: setDocumentsPageSize,
      pageInfo: documentsPageInfo,
      addOptimisticDocument,
      updateOptimisticDocument,
    } = useDocuments({
      apiBaseUrl: effectiveApiUrl,
      authToken,
      selectedFolder,
      folders,
      pageSize: 100,
      fields: [
        "external_id",
        "filename",
        "content_type",
        "metadata",
        "additional_metadata",
        "system_metadata",
        "folder_name",
      ],
      includeTotalCount: true,
    });

    const {
      unorganizedDocuments,
      loading: unorganizedDocumentsLoading,
      refresh: refreshUnorganizedDocuments,
    } = useUnorganizedDocuments({
      apiBaseUrl: effectiveApiUrl,
      authToken,
      enabled: selectedFolder === null,
    });

    const loading = documentsLoading || documentsLoadingMore || unorganizedDocumentsLoading;
    const unorganizedCacheKey = `${effectiveApiUrl}-${authToken ?? "anon"}-unorganized`;

    const documentsPagination = useMemo(() => {
      return {
        skip: documentsPageInfo.skip,
        limit: documentsPageInfo.limit,
        returnedCount: documentsPageInfo.returnedCount,
        totalCount: documentsPageInfo.totalCount,
        hasMore: documentsPageInfo.hasMore,
        nextSkip: documentsPageInfo.nextSkip,
        onPageChange: (nextSkip: number) => goToDocumentsPage(nextSkip),
        onPageSizeChange: (nextLimit: number) => setDocumentsPageSize(nextLimit),
        pageSizeOptions: [25, 50, 100, 250],
        loading: documentsLoadingMore,
      };
    }, [documentsLoadingMore, documentsPageInfo, goToDocumentsPage, setDocumentsPageSize]);

    // Search state for root level
    const [searchQuery, setSearchQuery] = useState("");

    const resolveDocumentFolderName = useCallback((doc: Document) => {
      const direct = typeof doc.folder_name === "string" ? doc.folder_name.trim() : "";
      if (direct) {
        return direct;
      }

      const systemMetadata = (doc.system_metadata ?? {}) as Record<string, unknown>;
      const metaFolder =
        typeof systemMetadata.folder_name === "string" ? (systemMetadata.folder_name as string).trim() : "";

      return metaFolder;
    }, []);

    const optimisticUnorganizedDocuments = useMemo(() => {
      if (selectedFolder !== null) {
        return [];
      }

      const existingIds = new Set(unorganizedDocuments.map(doc => doc.external_id));

      return documents.filter(doc => {
        const folderName = resolveDocumentFolderName(doc);
        const isUnorganized = folderName === "";
        const alreadyPresent = existingIds.has(doc.external_id);
        return isUnorganized && !alreadyPresent;
      });
    }, [selectedFolder, unorganizedDocuments, documents, resolveDocumentFolderName]);

    // Create combined list of documents and folders for root level display
    const combinedRootItems = useMemo(() => {
      const items: (Document & { itemType?: "document" | "folder" | "all"; folderData?: FolderSummary })[] = [];

      // Add folders as document-like objects
      folders.forEach(folder => {
        items.push({
          external_id: `folder-${folder.name}`,
          filename: folder.name,
          content_type: "folder",
          metadata: {},
          system_metadata: {
            created_at: folder.updated_at || new Date().toISOString(),
            file_size: folder.doc_count || 0,
          },
          additional_metadata: {},
          itemType: "folder",
          // Add folder-specific data
          folderData: folder,
        });
      });

      if (selectedFolder === null) {
        const seenDocumentIds = new Set<string>();

        // Add unorganized documents from the API first
        unorganizedDocuments.forEach(doc => {
          if (!seenDocumentIds.has(doc.external_id)) {
            items.push({ ...doc, itemType: "document" });
            seenDocumentIds.add(doc.external_id);
          }
        });

        // Append optimistic unorganized documents so they appear immediately
        optimisticUnorganizedDocuments.forEach(doc => {
          if (!seenDocumentIds.has(doc.external_id)) {
            items.push({ ...doc, itemType: "document" });
            seenDocumentIds.add(doc.external_id);
          }
        });
      }

      return items;
    }, [unorganizedDocuments, optimisticUnorganizedDocuments, folders, selectedFolder]);

    // Filter combined items based on search query
    const filteredRootItems = useMemo(() => {
      if (!searchQuery.trim()) return combinedRootItems;
      const query = searchQuery.toLowerCase();
      return combinedRootItems.filter(item => {
        if (item.itemType === "all") {
          return "all documents".includes(query);
        }
        return (item.filename || item.external_id).toLowerCase().includes(query);
      });
    }, [combinedRootItems, searchQuery]);

    const [rootSkip, setRootSkip] = useState(0);
    const [rootPageSize, setRootPageSize] = useState(100);

    useEffect(() => {
      setRootSkip(0);
    }, [searchQuery]);

    useEffect(() => {
      setRootSkip(prev => {
        if (filteredRootItems.length === 0) {
          return 0;
        }

        const maxValidStart = Math.max(filteredRootItems.length - rootPageSize, 0);
        const clamped = Math.min(prev, maxValidStart);
        const aligned = Math.floor(clamped / rootPageSize) * rootPageSize;

        return aligned;
      });
    }, [filteredRootItems.length, rootPageSize]);

    const pagedRootItems = useMemo(
      () => filteredRootItems.slice(rootSkip, rootSkip + rootPageSize),
      [filteredRootItems, rootSkip, rootPageSize]
    );

    const rootPagination = useMemo(() => {
      const totalCount = filteredRootItems.length;
      const returnedCount = pagedRootItems.length;
      const hasMore = returnedCount > 0 && rootSkip + returnedCount < totalCount;
      const nextSkip = hasMore ? rootSkip + returnedCount : null;

      return {
        skip: rootSkip,
        limit: rootPageSize,
        returnedCount,
        totalCount,
        hasMore,
        nextSkip,
        onPageChange: (next: number) => {
          const clampedNext = Math.max(0, Math.min(next, Math.max(totalCount - 1, 0)));
          const alignedNext = Math.floor(clampedNext / rootPageSize) * rootPageSize;
          setRootSkip(alignedNext);
        },
        onPageSizeChange: (nextLimit: number) => {
          if (!Number.isFinite(nextLimit) || nextLimit <= 0) {
            return;
          }
          const normalized = Math.max(1, Math.floor(nextLimit));
          setRootPageSize(normalized);
          setRootSkip(0);
        },
        pageSizeOptions: [25, 50, 100, 250],
        loading,
      };
    }, [filteredRootItems.length, pagedRootItems, rootPageSize, rootSkip, loading]);

    // State for delete confirmation modal
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [itemToDelete, setItemToDelete] = useState<string | null>(null); // For single delete: stores ID
    const [itemsToDeleteCount, setItemsToDeleteCount] = useState<number>(0); // For multiple delete: stores count

    // State for polling
    const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // State for New Folder dialog - use prop if provided
    const [showNewFolderDialogLocal, setShowNewFolderDialogLocal] = useState(false);
    const showNewFolderDialog =
      showNewFolderDialogProp !== undefined ? showNewFolderDialogProp : showNewFolderDialogLocal;
    const setShowNewFolderDialog = setShowNewFolderDialogProp || setShowNewFolderDialogLocal;

    const [newFolderName, setNewFolderName] = useState("");
    const [newFolderDescription, setNewFolderDescription] = useState("");
    const [isCreatingFolder, setIsCreatingFolder] = useState(false);

    // Upload dialog state from custom hook
    const uploadDialogState = useUploadDialog();
    // Extract only the state variables we actually use in this component
    const { metadata, rules, useColpali, resetUploadDialog } = uploadDialogState;

    // Use prop for upload dialog if provided
    const [showUploadDialogLocal, setShowUploadDialogLocal] = useState(false);
    const showUploadDialog = showUploadDialogProp !== undefined ? showUploadDialogProp : showUploadDialogLocal;
    const setShowUploadDialog = setShowUploadDialogProp || setShowUploadDialogLocal;

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

    // Polling function to check status of processing documents
    const pollProcessingDocuments = useCallback(async () => {
      // Get all documents that are in processing status
      const processingDocs = documents.filter(doc => doc.system_metadata?.status === "processing");

      if (processingDocs.length === 0) {
        // No documents to poll, clear the interval
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
          console.log("Stopped polling - no processing documents");
        }
        return;
      }

      console.log(`Polling status for ${processingDocs.length} processing documents`);

      try {
        // Fetch status for each processing document
        const statusPromises = processingDocs.map(async doc => {
          const response = await fetch(`${effectiveApiUrl}/documents/${doc.external_id}`, {
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

        // Check if any documents have changed status
        let hasChanges = false;
        updatedDocs.forEach(updatedDoc => {
          if (updatedDoc && updatedDoc.system_metadata?.status !== "processing") {
            hasChanges = true;
            console.log(`Document ${updatedDoc.external_id} status changed to: ${updatedDoc.system_metadata?.status}`);
          }
        });

        // If any document status changed, refresh the documents list
        if (hasChanges) {
          await refreshDocuments();
        }
      } catch (error) {
        console.error("Error polling document status:", error);
      }
    }, [documents, effectiveApiUrl, authToken, refreshDocuments]);

    // Effect to manage polling
    useEffect(() => {
      const processingDocs = documents.filter(doc => doc.system_metadata?.status === "processing");

      if (processingDocs.length > 0 && !pollingIntervalRef.current) {
        // Start polling if we have processing documents and not already polling
        console.log(`Starting polling for ${processingDocs.length} processing documents`);

        // Do an immediate poll
        pollProcessingDocuments();

        // Then set up interval for every 2 seconds
        pollingIntervalRef.current = setInterval(pollProcessingDocuments, 2000);
      } else if (processingDocs.length === 0 && pollingIntervalRef.current) {
        // Stop polling if no processing documents
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
        console.log("Stopped polling - no processing documents");
      }

      // Cleanup on unmount
      return () => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      };
    }, [documents, pollProcessingDocuments]);

    // Removed automatic sidebar collapse when folder is selected
    // The sidebar should only be controlled by the dedicated open/close button
    // useEffect(() => {
    //   if (selectedFolder !== null && setSidebarCollapsed) {
    //     setSidebarCollapsed(true);
    //   } else if (setSidebarCollapsed) {
    //     setSidebarCollapsed(false);
    //   }
    // }, [selectedFolder, setSidebarCollapsed]);

    // Fetch a specific document by ID
    const fetchDocument = useCallback(
      async (documentId: string) => {
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
              if (!data.system_metadata.status && typeof data.folder_name === "string") {
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
      },
      [effectiveApiUrl, authToken]
    );

    // Handle document click
    const handleDocumentClick = useCallback(
      (document: Document) => {
        // Invoke callback prop before fetching
        const docName = document.filename || document.external_id; // Use filename, fallback to ID
        console.log(`handleDocumentClick: Calling onDocumentClick with '${docName}'`);
        onDocumentClick?.(docName);
        fetchDocument(document.external_id);
      },
      [onDocumentClick, fetchDocument]
    );

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
    const handleDeleteDocument = useCallback(async (documentId: string) => {
      setItemToDelete(documentId);
      setItemsToDeleteCount(0); // Ensure this is 0 for single delete scenario
      setShowDeleteModal(true);
    }, []);

    // Handle document download
    const handleDownloadDocument = useCallback(
      async (documentId: string) => {
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
      },
      [effectiveApiUrl, authToken, documents]
    );

    const confirmDeleteSingleDocument = async () => {
      if (!itemToDelete) return;

      try {
        // Find document name before deleting (for callback)
        const docToDelete = documents.find(doc => doc.external_id === itemToDelete);
        const docName = docToDelete?.filename || itemToDelete; // Use filename, fallback to ID
        console.log(`confirmDeleteSingleDocument: Calling onDocumentDelete with '${docName}'`);
        onDocumentDelete?.(docName); // Invoke callback

        setShowDeleteModal(false); // Close modal before starting deletion

        console.log("DocumentsSection: Deleting document:", itemToDelete);

        await deleteDocumentApi(itemToDelete);

        // Clear selected document if it was the one deleted
        if (selectedDocument?.external_id === itemToDelete) {
          setSelectedDocument(null);
        }

        // Clear caches and refresh data
        clearFoldersCache(effectiveApiUrl);
        clearDocumentsCache();
        clearUnorganizedDocumentsCache(unorganizedCacheKey);
        await refreshFolders();
        await refreshDocuments();
        await refreshUnorganizedDocuments();

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
        setItemToDelete(null);
      }
    };

    // Handle multiple document deletion
    const handleDeleteMultipleDocuments = useCallback(async () => {
      if (selectedDocuments.length === 0) return;
      setItemsToDeleteCount(selectedDocuments.length);
      setItemToDelete(null); // Ensure this is null for multiple delete scenario
      setShowDeleteModal(true);
    }, [selectedDocuments.length]);

    const confirmDeleteMultipleDocuments = async () => {
      if (selectedDocuments.length === 0) return;

      try {
        setShowDeleteModal(false); // Close modal before starting deletion

        // Separate folders and documents
        const itemsToDelete = selectedDocuments.map(id => {
          // Check if it's a folder by looking at combinedRootItems or documents
          const item =
            selectedFolder === null
              ? combinedRootItems.find(item => item.external_id === id)
              : documents.find(doc => doc.external_id === id);

          const isFolder = item && (item as Document & { itemType?: string }).itemType === "folder";
          const folderName = isFolder ? (item as Document & { filename?: string }).filename : null;

          return { id, isFolder, folderName, item };
        });

        const foldersToDelete = itemsToDelete.filter(item => item.isFolder);
        const docsToDelete = itemsToDelete.filter(item => !item.isFolder);

        // Show initial alert for deletion progress
        const alertId = "delete-multiple-progress";
        const itemTypeText =
          foldersToDelete.length > 0 && docsToDelete.length > 0
            ? `${docsToDelete.length} documents and ${foldersToDelete.length} folders`
            : foldersToDelete.length > 0
              ? `${foldersToDelete.length} folder${foldersToDelete.length > 1 ? "s" : ""}`
              : `${docsToDelete.length} document${docsToDelete.length > 1 ? "s" : ""}`;

        showAlert(`Deleting ${itemTypeText}...`, {
          type: "info",
          dismissible: false,
          id: alertId,
        });

        console.log("DocumentsSection: Deleting items:", { folders: foldersToDelete, documents: docsToDelete });

        // Delete folders first (they might contain documents)
        const folderResults = await Promise.all(
          foldersToDelete.map(async ({ folderName }) => {
            if (!folderName) return { ok: false };
            try {
              const response = await fetch(`${effectiveApiUrl}/folders/${folderName}`, {
                method: "DELETE",
                headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
              });
              return response;
            } catch (error) {
              console.error(`Error deleting folder ${folderName}:`, error);
              return { ok: false };
            }
          })
        );

        // Then delete documents
        const docResults = await Promise.all(docsToDelete.map(({ id }) => deleteDocumentApi(id)));

        // Combine results
        const allResults = [...folderResults, ...docResults];
        const failedCount = allResults.filter(res => !res.ok).length;

        // Clear selected document if it was among deleted ones
        if (selectedDocument && selectedDocuments.includes(selectedDocument.external_id)) {
          setSelectedDocument(null);
        }

        // Clear selection
        setSelectedDocuments([]);

        // Clear caches and refresh data
        clearFoldersCache(effectiveApiUrl);
        clearDocumentsCache();
        clearUnorganizedDocumentsCache(unorganizedCacheKey);
        await Promise.all([refreshFolders(), refreshDocuments(), refreshUnorganizedDocuments()]);

        // Remove progress alert
        removeAlert(alertId);

        // Show final result alert
        if (failedCount > 0) {
          showAlert(`Deleted ${selectedDocuments.length - failedCount} items. ${failedCount} deletions failed.`, {
            type: "warning",
            duration: 4000,
          });
        } else {
          const successText =
            foldersToDelete.length > 0 && docsToDelete.length > 0
              ? `Successfully deleted ${docsToDelete.length} documents and ${foldersToDelete.length} folders`
              : foldersToDelete.length > 0
                ? `Successfully deleted ${foldersToDelete.length} folder${foldersToDelete.length > 1 ? "s" : ""}`
                : `Successfully deleted ${docsToDelete.length} document${docsToDelete.length > 1 ? "s" : ""}`;

          showAlert(successText, {
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
        setSelectedDocuments([]); // Clear selection after attempting deletion
        setItemsToDeleteCount(0);
      }
    };

    // Expose methods via ref
    React.useImperativeHandle(ref, () => ({
      handleRefresh,
      handleDeleteMultipleDocuments,
      selectedDocuments,
    }));

    // Send folders updates to parent
    useEffect(() => {
      if (onFoldersUpdate && folders.length > 0) {
        const simpleFolders = folders.map(f => ({ id: f.id || f.name, name: f.name }));
        onFoldersUpdate(simpleFolders);
      }
    }, [folders, onFoldersUpdate]);

    // Handle checkbox change (wrapper function for use with shadcn checkbox)
    const handleCheckboxChange = useCallback((checked: boolean | "indeterminate", docId: string) => {
      setSelectedDocuments(prev => {
        if (checked === true && !prev.includes(docId)) {
          return [...prev, docId];
        } else if (checked === false && prev.includes(docId)) {
          return prev.filter(id => id !== docId);
        }
        return prev;
      });
    }, []);

    // Handle file upload
    const handleFileUpload = async (
      file: File | null,
      metadataParam?: string,
      rulesParam?: string,
      useColpaliParam?: boolean
    ) => {
      if (!file) {
        showAlert("Please select a file to upload", {
          type: "error",
          duration: 3000,
        });
        return;
      }

      // Close dialog
      setShowUploadDialog(false);

      // Generate temporary ID for optimistic update
      const tempId = generateTempId();

      // Add document immediately with uploading status
      const optimisticDoc: Document = {
        external_id: tempId,
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        metadata: {},
        folder_name: selectedFolder && selectedFolder !== "all" ? selectedFolder : undefined,
        system_metadata: {
          status: "uploading",
          folder_name: selectedFolder && selectedFolder !== "all" ? selectedFolder : undefined,
        },
        additional_metadata: {},
      };

      addOptimisticDocument(optimisticDoc);

      // Use passed parameters or fall back to hook values
      const fileToUploadRef = file;
      const metadataRef = metadataParam ?? metadata;
      const rulesRef = rulesParam ?? rules;
      const useColpaliRef = useColpaliParam ?? useColpali;

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
            // Replace the optimistic placeholder with the real document metadata while it finishes processing
            const normalizedDocument: Document = {
              ...optimisticDoc,
              ...newDocument,
              external_id: newDocument.external_id ?? optimisticDoc.external_id,
              folder_name: newDocument.folder_name ?? optimisticDoc.folder_name,
              system_metadata: {
                ...(optimisticDoc.system_metadata ?? {}),
                ...(newDocument.system_metadata ?? {}),
                status: newDocument.system_metadata?.status ?? optimisticDoc.system_metadata?.status ?? "processing",
                folder_name:
                  newDocument.system_metadata?.folder_name ??
                  newDocument.folder_name ??
                  optimisticDoc.system_metadata?.folder_name ??
                  optimisticDoc.folder_name,
              },
              metadata: {
                ...(optimisticDoc.metadata ?? {}),
                ...(newDocument.metadata ?? {}),
              },
              additional_metadata: {
                ...(optimisticDoc.additional_metadata ?? {}),
                ...(newDocument.additional_metadata ?? {}),
              },
            };

            updateOptimisticDocument(tempId, normalizedDocument);

            // Invoke callback on success
            console.log(
              `handleFileUpload: Calling onDocumentUpload with '${fileToUploadRef.name}', size: ${fileToUploadRef.size}`
            );
            onDocumentUpload?.(fileToUploadRef.name, fileToUploadRef.size);

            // Log processing status of uploaded document
            if (newDocument && newDocument.system_metadata && newDocument.system_metadata.status === "processing") {
              console.log(`Document ${newDocument.external_id} is in processing status`);
            }

            // Force a fresh refresh after upload
            const refreshAfterUpload = async () => {
              try {
                console.log("Performing fresh refresh after upload (file)");
                // Clear caches and refresh data
                clearFoldersCache(effectiveApiUrl);
                clearDocumentsCache();
                clearUnorganizedDocumentsCache(unorganizedCacheKey);
                await Promise.all([refreshFolders(), refreshDocuments(), refreshUnorganizedDocuments()]);
              } catch (err) {
                console.error("Error refreshing after file upload:", err);
              }
            };

            // Execute the refresh with a small delay to ensure backend has committed the document
            setTimeout(() => {
              refreshAfterUpload();
            }, 1000); // 1 second delay to ensure document is fully committed

            // Show success message
            showAlert(`File uploaded successfully!`, {
              type: "success",
              duration: 3000,
            });
          })
          .catch(err => {
            const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
            const errorMsg = `Error uploading ${fileToUploadRef.name}: ${errorMessage}`;

            // Update the optimistic document to show failed status
            updateOptimisticDocument(tempId, {
              system_metadata: {
                ...optimisticDoc.system_metadata,
                status: "failed",
                error: errorMessage,
              },
            });

            // Show error alert
            showAlert(errorMsg, {
              type: "error",
              title: "Upload Failed",
              duration: 5000,
            });
          });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
        const errorMsg = `Error uploading ${fileToUploadRef.name}: ${errorMessage}`;

        // Update the optimistic document to show failed status
        updateOptimisticDocument(tempId, {
          system_metadata: {
            ...optimisticDoc.system_metadata,
            status: "failed",
            error: errorMessage,
          },
        });

        // Show error alert
        showAlert(errorMsg, {
          type: "error",
          title: "Upload Failed",
          duration: 5000,
        });
      }
    };

    // Handle batch file upload
    const handleBatchFileUpload = async (
      files: File[],
      metadataParamOrFromDragAndDrop?: string | boolean,
      rulesParam?: string,
      useColpaliParam?: boolean
    ) => {
      // Handle overloaded parameters - check if second param is boolean (old signature) or string (new signature)
      const fromDragAndDrop =
        typeof metadataParamOrFromDragAndDrop === "boolean" ? metadataParamOrFromDragAndDrop : false;
      const metadataParam =
        typeof metadataParamOrFromDragAndDrop === "string" ? metadataParamOrFromDragAndDrop : undefined;
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

      // Add optimistic documents for each file
      const tempIdMap = new Map<string, Document>(); // Map temp ID to optimistic document
      files.forEach(file => {
        const tempId = generateTempId();
        const optimisticDoc: Document = {
          external_id: tempId,
          filename: file.name,
          content_type: file.type || "application/octet-stream",
          metadata: {},
          folder_name: selectedFolder && selectedFolder !== "all" ? selectedFolder : undefined,
          system_metadata: {
            status: "uploading",
            folder_name: selectedFolder && selectedFolder !== "all" ? selectedFolder : undefined,
          },
          additional_metadata: {},
        };

        addOptimisticDocument(optimisticDoc);
        tempIdMap.set(tempId, optimisticDoc);
      });

      // Save form data locally - use passed parameters or fall back to hook values
      const batchFilesRef = [...files];
      const metadataRef = metadataParam ?? metadata;
      const rulesRef = rulesParam ?? rules;
      const useColpaliRef = useColpaliParam ?? useColpali;

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
            const returnedDocuments: Document[] = Array.isArray(result?.documents) ? [...result.documents] : [];

            // Update each optimistic document with the real metadata so progress stays visible
            tempIdMap.forEach((optimisticDoc, tempId) => {
              const backendDoc = returnedDocuments.shift();
              if (!backendDoc) {
                return;
              }

              const normalizedDocument: Document = {
                ...optimisticDoc,
                ...backendDoc,
                external_id: backendDoc.external_id ?? optimisticDoc.external_id,
                folder_name:
                  backendDoc.folder_name ??
                  optimisticDoc.folder_name ??
                  (selectedFolder && selectedFolder !== "all" ? selectedFolder : undefined),
                system_metadata: {
                  ...(optimisticDoc.system_metadata ?? {}),
                  ...(backendDoc.system_metadata ?? {}),
                  status: backendDoc.system_metadata?.status ?? "processing",
                  folder_name:
                    backendDoc.system_metadata?.folder_name ??
                    backendDoc.folder_name ??
                    optimisticDoc.system_metadata?.folder_name ??
                    optimisticDoc.folder_name ??
                    (selectedFolder && selectedFolder !== "all" ? selectedFolder : undefined),
                },
                metadata: {
                  ...(optimisticDoc.metadata ?? {}),
                  ...(backendDoc.metadata ?? {}),
                },
                additional_metadata: {
                  ...(optimisticDoc.additional_metadata ?? {}),
                  ...(backendDoc.additional_metadata ?? {}),
                },
              };

              updateOptimisticDocument(tempId, normalizedDocument);
            });

            // Invoke callback on success
            console.log(
              `handleBatchFileUpload: Calling onDocumentUpload with '${batchFilesRef[0].name}', size: ${batchFilesRef[0].size} (for first file in batch)`
            );
            onDocumentUpload?.(batchFilesRef[0].name, batchFilesRef[0].size);

            // Log processing status of uploaded documents
            if (result && result.document_ids && result.document_ids.length > 0) {
              console.log(`${result.document_ids.length} documents are in processing status`);
            }

            // Force a fresh refresh after upload
            const refreshAfterUpload = async () => {
              try {
                console.log("Performing fresh refresh after upload (batch)");
                // Clear caches and refresh data
                clearFoldersCache(effectiveApiUrl);
                clearDocumentsCache();
                clearUnorganizedDocumentsCache(unorganizedCacheKey);
                await Promise.all([refreshFolders(), refreshDocuments(), refreshUnorganizedDocuments()]);
              } catch (err) {
                console.error("Error refreshing after batch upload:", err);
              }
            };

            // Execute the refresh with a small delay to ensure backend has committed the documents
            setTimeout(() => {
              refreshAfterUpload();
            }, 1000); // 1 second delay to ensure documents are fully committed

            // If there are errors, show them in the error alert
            if (result.errors && result.errors.length > 0) {
              const errorMsg = `${result.errors.length} of ${files.length} files failed to upload`;

              showAlert(errorMsg, {
                type: "error",
                title: "Upload Partially Failed",
                duration: 5000,
              });
            } else {
              // Show success message
              showAlert(`${files.length} files uploaded successfully!`, {
                type: "success",
                duration: 3000,
              });
            }
          })
          .catch(err => {
            const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
            const errorMsg = `Error uploading files: ${errorMessage}`;

            // Update all optimistic documents to show failed status
            tempIdMap.forEach((optimisticDoc, tempId) => {
              updateOptimisticDocument(tempId, {
                system_metadata: {
                  status: "failed",
                  error: errorMessage,
                  folder_name:
                    selectedFolder && selectedFolder !== "all"
                      ? selectedFolder
                      : (optimisticDoc.folder_name ?? optimisticDoc.system_metadata?.folder_name),
                },
              });
            });

            // Show error alert
            showAlert(errorMsg, {
              type: "error",
              title: "Upload Failed",
              duration: 5000,
            });
          });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An unknown error occurred";
        const errorMsg = `Error uploading files: ${errorMessage}`;

        // Update all optimistic documents to show failed status
        tempIdMap.forEach((optimisticDoc, tempId) => {
          updateOptimisticDocument(tempId, {
            system_metadata: {
              status: "failed",
              error: errorMessage,
              folder_name:
                selectedFolder && selectedFolder !== "all"
                  ? selectedFolder
                  : (optimisticDoc.folder_name ?? optimisticDoc.system_metadata?.folder_name),
            },
          });
        });

        // Show error alert
        showAlert(errorMsg, {
          type: "error",
          title: "Upload Failed",
          duration: 5000,
        });
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
      let metadataObj: Record<string, unknown> = {};
      let folderToUse: string | null = null;

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

      const tempId = generateTempId();
      const metadataTitle =
        typeof metadataObj["title"] === "string" && metadataObj["title"].toString().trim().length > 0
          ? metadataObj["title"].toString().trim()
          : null;
      const optimisticFilename = metadataTitle || `Text upload ${new Date().toISOString()}`;

      const optimisticTextDoc: Document = {
        external_id: tempId,
        filename: optimisticFilename,
        content_type: "text/plain",
        metadata: metadataObj,
        system_metadata: {
          status: "uploading",
          folder_name: folderToUse ?? undefined,
        },
        additional_metadata: {},
        folder_name: folderToUse ?? undefined,
      };

      addOptimisticDocument(optimisticTextDoc);

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

            const normalizedDocument: Document = {
              ...optimisticTextDoc,
              ...newDocument,
              external_id: newDocument.external_id ?? optimisticTextDoc.external_id,
              folder_name: newDocument.folder_name ?? optimisticTextDoc.folder_name,
              system_metadata: {
                ...(optimisticTextDoc.system_metadata ?? {}),
                ...(newDocument.system_metadata ?? {}),
                status:
                  newDocument.system_metadata?.status ?? optimisticTextDoc.system_metadata?.status ?? "processing",
                folder_name:
                  newDocument.system_metadata?.folder_name ??
                  newDocument.folder_name ??
                  optimisticTextDoc.system_metadata?.folder_name ??
                  optimisticTextDoc.folder_name,
              },
              metadata: {
                ...(optimisticTextDoc.metadata ?? {}),
                ...(newDocument.metadata ?? {}),
              },
              additional_metadata: {
                ...(optimisticTextDoc.additional_metadata ?? {}),
                ...(newDocument.additional_metadata ?? {}),
              },
            };

            updateOptimisticDocument(tempId, normalizedDocument);

            // Force a fresh refresh after upload
            const refreshAfterUpload = async () => {
              try {
                console.log("Performing fresh refresh after upload (text)");
                // Clear caches and refresh data
                clearFoldersCache(effectiveApiUrl);
                clearDocumentsCache();
                clearUnorganizedDocumentsCache(unorganizedCacheKey);
                await Promise.all([refreshFolders(), refreshDocuments(), refreshUnorganizedDocuments()]);
              } catch (err) {
                console.error("Error refreshing after text upload:", err);
              }
            };

            // Execute the refresh with a small delay to ensure backend has committed the document
            setTimeout(() => {
              refreshAfterUpload();
            }, 1000); // 1 second delay to ensure document is fully committed

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

            updateOptimisticDocument(tempId, {
              system_metadata: {
                ...optimisticTextDoc.system_metadata,
                status: "failed",
                error: errorMessage,
              },
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

        updateOptimisticDocument(tempId, {
          system_metadata: {
            ...optimisticTextDoc.system_metadata,
            status: "failed",
            error: errorMessage,
          },
        });

        // Remove the upload progress alert
        removeAlert("text-upload-progress");
      }
    };

    // Function to trigger refresh
    const handleRefresh = useCallback(async () => {
      // Invoke callback
      onRefresh?.();

      try {
        // Clear caches and refresh both folders and documents
        clearFoldersCache(effectiveApiUrl);
        clearDocumentsCache();
        clearUnorganizedDocumentsCache(unorganizedCacheKey);

        await Promise.all([refreshFolders(), refreshDocuments(), refreshUnorganizedDocuments()]);

        showAlert("Data refreshed successfully.", {
          type: "success",
          duration: 1500,
        });
      } catch (error) {
        console.error("Error during refresh:", error);
        showAlert(`Error refreshing: ${error instanceof Error ? error.message : "Unknown error"}`, {
          type: "error",
          duration: 3000,
        });
      }
    }, [
      onRefresh,
      effectiveApiUrl,
      unorganizedCacheKey,
      refreshFolders,
      refreshDocuments,
      refreshUnorganizedDocuments,
    ]);

    // Debounced version of refresh for rapid refresh calls (kept for future use)
    // const handleDebouncedRefresh = useDebounce(handleRefresh, 500);

    // Wrapper for setSelectedFolder to include callback invocation
    const handleFolderSelect = useCallback(
      (folderName: string | null) => {
        console.log(`handleFolderSelect: Calling onFolderClick with '${folderName}'`);
        onFolderClick?.(folderName);
        setSelectedFolder(folderName);
      },
      [onFolderClick]
    ); // Add setSelectedFolder if its identity matters, but it usually doesn't

    // Custom click handler for root level items
    const handleRootItemClick = useCallback(
      (item: Document & { itemType?: "document" | "folder" }) => {
        if (item.itemType === "document") {
          // Handle document click
          handleDocumentClick(item);
        } else if (item.itemType === "folder") {
          // Handle folder click
          const folderName = item.filename || "";
          handleFolderSelect(folderName);
        }
      },
      [handleDocumentClick, handleFolderSelect]
    );

    // Handle folder creation
    const handleCreateFolder = async () => {
      if (!newFolderName.trim()) return;

      setIsCreatingFolder(true);

      try {
        console.log(`Creating folder: ${newFolderName}`);

        const response = await fetch(`${effectiveApiUrl}/folders`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify({
            name: newFolderName.trim(),
            description: newFolderDescription.trim() || undefined,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to create folder: ${response.statusText}`);
        }

        // Get the created folder data
        const folderData = await response.json();
        console.log(`Created folder with ID: ${folderData.id} and name: ${folderData.name}`);

        // Close dialog and reset form
        setShowNewFolderDialog(false);
        setNewFolderName("");
        setNewFolderDescription("");

        // Refresh folder list
        clearFoldersCache(effectiveApiUrl);
        clearDocumentsCache();
        clearUnorganizedDocumentsCache(unorganizedCacheKey);
        await Promise.all([refreshFolders(), refreshDocuments(), refreshUnorganizedDocuments()]);

        // Invoke callback
        console.log(`handleCreateFolder: Calling onFolderCreate with '${folderData.name}'`);
        onFolderCreate?.(folderData.name);

        // Show success message
        showAlert("Folder created successfully", {
          type: "success",
          duration: 3000,
        });
      } catch (error) {
        console.error("Error creating folder:", error);
        showAlert(`Failed to create folder: ${error instanceof Error ? error.message : "Unknown error"}`, {
          type: "error",
          duration: 5000,
        });
      } finally {
        setIsCreatingFolder(false);
      }
    };

    return (
      <div
        className={cn("relative flex h-full flex-1 flex-col", selectedFolder && isDragging ? "drag-active" : "")}
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

        {selectedFolder !== null && (
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleFolderSelect(null)}
              className="flex items-center gap-2 px-0 text-sm font-medium"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Documents
            </Button>
            <span className="text-sm font-semibold text-foreground">
              {selectedFolder === "all" ? "All Documents" : selectedFolder}
            </span>
            <div className="min-w-[200px] flex-1 md:ml-auto md:max-w-md">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search documents in this folder..."
                  value={folderSearchQuery}
                  onChange={e => setFolderSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
          </div>
        )}

        {/* Root Level List View (selectedFolder is null) */}
        {selectedFolder === null ? (
          <>
            {/* Separate Search Bar for Root Level - Outside the flex container to match folder view */}
            {folders.length !== 0 || unorganizedDocuments.length !== 0 ? (
              <div className="mb-4 bg-background">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search documents..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>
              </div>
            ) : null}

            <div className="flex min-h-0 flex-1 flex-col gap-4 md:flex-row md:items-start">
              {/* Left Panel: Combined Folders and Documents List */}
              <div
                className={cn(
                  "flex min-h-0 w-full flex-col transition-all duration-300",
                  selectedDocument ? "md:w-2/3" : "md:w-full"
                )}
              >
                {loading && folders.length === 0 && unorganizedDocuments.length === 0 ? (
                  // Initial skeleton only when no data is yet loaded
                  <div className="flex-1 space-y-3 p-4">
                    <Skeleton className="h-10 w-full" />
                    <Skeleton className="h-8 w-3/4" />
                    <Skeleton className="h-8 w-full" />
                    <Skeleton className="h-8 w-5/6" />
                    <Skeleton className="h-8 w-full" />
                  </div>
                ) : folders.length === 0 && unorganizedDocuments.length === 0 ? (
                  // Empty State
                  <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed py-8 text-center">
                    <div>
                      <Upload className="mx-auto mb-2 h-12 w-12 text-muted-foreground" />
                      <p className="text-muted-foreground">No folders or documents found.</p>
                      <p className="mt-2 text-xs text-muted-foreground">Upload files to get started.</p>
                    </div>
                  </div>
                ) : (
                  // Document list with loading indicator
                  <div className={cn("relative min-h-0 flex-1 transition-opacity", loading ? "opacity-60" : "")}>
                    {/* Tiny corner spinner for loading state */}
                    {loading && (
                      <div className="absolute left-2 top-2 z-10 flex items-center">
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                      </div>
                    )}

                    <DocumentList
                      documents={pagedRootItems}
                      selectedDocument={selectedDocument}
                      selectedDocuments={selectedDocuments}
                      handleDocumentClick={handleRootItemClick}
                      handleCheckboxChange={handleCheckboxChange}
                      setSelectedDocuments={setSelectedDocuments}
                      loading={loading || foldersLoading}
                      apiBaseUrl={effectiveApiUrl}
                      authToken={authToken}
                      selectedFolder={null} // Root level
                      onViewInPDFViewer={onViewInPDFViewer}
                      onDownloadDocument={handleDownloadDocument}
                      onDeleteDocument={handleDeleteDocument}
                      onDeleteMultipleDocuments={handleDeleteMultipleDocuments}
                      folders={folders}
                      showBorder={true} // Keep border for the table
                      hideSearchBar={true} // Hide the search bar inside DocumentList
                      externalSearchQuery={searchQuery} // Pass the external search query
                      onSearchChange={setSearchQuery} // Handle search changes
                      allFoldersExpanded={allFoldersExpanded}
                      pagination={rootPagination}
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
                    refreshDocuments={refreshDocuments}
                    refreshFolders={refreshFolders}
                    loading={loading}
                    onClose={() => setSelectedDocument(null)}
                    onViewInPDFViewer={onViewInPDFViewer}
                    onMetadataUpdate={documentId => fetchDocument(documentId)}
                  />
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-4 md:flex-row md:items-start">
            {/* Left Panel: Document List or Skeleton or Empty State */}
            <div
              className={cn(
                "flex min-h-0 w-full flex-col transition-all duration-300",
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
                <div
                  className={cn(
                    "relative min-h-0 flex-1 transition-opacity",
                    loading && documents.length > 0 ? "opacity-60" : ""
                  )}
                >
                  {/* Tiny corner spinner instead of full overlay */}
                  {loading && documents.length > 0 && (
                    <div className="absolute left-2 top-2 z-10 flex items-center">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    </div>
                  )}

                  <DocumentList
                    documents={documents}
                    selectedDocument={selectedDocument}
                    selectedDocuments={selectedDocuments}
                    handleDocumentClick={handleDocumentClick}
                    handleCheckboxChange={handleCheckboxChange}
                    setSelectedDocuments={setSelectedDocuments}
                    loading={loading}
                    apiBaseUrl={effectiveApiUrl}
                    authToken={authToken}
                    selectedFolder={selectedFolder}
                    onViewInPDFViewer={onViewInPDFViewer}
                    onDownloadDocument={handleDownloadDocument}
                    onDeleteDocument={handleDeleteDocument}
                    onDeleteMultipleDocuments={handleDeleteMultipleDocuments}
                    folders={folders}
                    showBorder={true} // Keep border for the table
                    hideSearchBar={true} // Hide the search bar inside DocumentList
                    externalSearchQuery={folderSearchQuery} // Pass the external search query
                    onSearchChange={setFolderSearchQuery} // Handle search changes
                    pagination={documentsPagination}
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
                  refreshDocuments={refreshDocuments}
                  refreshFolders={refreshFolders}
                  loading={loading}
                  onClose={() => setSelectedDocument(null)}
                  onViewInPDFViewer={onViewInPDFViewer}
                  onMetadataUpdate={fetchDocument}
                />
              </div>
            )}
          </div>
        )}

        {/* Dialog for creating new folder */}
        <Dialog open={showNewFolderDialog} onOpenChange={setShowNewFolderDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Folder</DialogTitle>
              <DialogDescription>Create a new folder to organize your documents.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div>
                <Label htmlFor="folderName">Folder Name</Label>
                <Input
                  id="folderName"
                  value={newFolderName}
                  onChange={e => setNewFolderName(e.target.value)}
                  placeholder="Enter folder name"
                />
              </div>
              <div>
                <Label htmlFor="folderDescription">Description (Optional)</Label>
                <Textarea
                  id="folderDescription"
                  value={newFolderDescription}
                  onChange={e => setNewFolderDescription(e.target.value)}
                  placeholder="Enter folder description"
                  rows={3}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="ghost" onClick={() => setShowNewFolderDialog(false)} disabled={isCreatingFolder}>
                Cancel
              </Button>
              <Button onClick={handleCreateFolder} disabled={!newFolderName.trim() || isCreatingFolder}>
                {isCreatingFolder ? "Creating..." : "Create Folder"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Upload Dialog - Always rendered at top level */}
        <UploadDialog
          showUploadDialog={showUploadDialog}
          setShowUploadDialog={setShowUploadDialog}
          loading={loading}
          onFileUpload={handleFileUpload}
          onBatchFileUpload={handleBatchFileUpload}
          onTextUpload={handleTextUpload}
        />
      </div>
    );
  }
);

DocumentsSection.displayName = "DocumentsSection";

export default DocumentsSection;
