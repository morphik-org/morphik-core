"use client";

import React, { createContext, useContext, useState, useCallback, useEffect, useRef, useMemo } from "react";
import { Document } from "../types";
import { useHeader } from "@/contexts/header-context";
import { Button } from "../ui/button";
import { Upload, FolderPlus, RefreshCw, Trash2 } from "lucide-react";
import { DocumentsApi, FolderSummary } from "./api/documents-api";

// Union type for table items
export type FileSystemItem = { type: "folder"; data: FolderSummary } | { type: "document"; data: Document };

// Processing status for documents
export type ProcessingStatus = "processing" | "completed" | "failed";

interface DocumentsContextType {
  // API configuration
  apiBaseUrl: string;
  authToken: string | null;

  // State
  documents: Document[];
  folders: FolderSummary[];
  currentFolder: string | null;
  selectedIds: Set<string>;
  selectedDocument: Document | null;
  searchQuery: string;
  isLoading: boolean;
  loadingStates: Map<string, ProcessingStatus>;

  // Combined items for table display
  tableItems: FileSystemItem[];

  // Actions
  setDocuments: (docs: Document[]) => void;
  setFolders: (folders: FolderSummary[]) => void;
  setCurrentFolder: (folder: string | null) => void;
  setSelectedIds: (ids: Set<string>) => void;
  setSelectedDocument: (doc: Document | null) => void;
  setSearchQuery: (query: string) => void;
  setIsLoading: (loading: boolean) => void;
  updateDocumentStatus: (docId: string, status: ProcessingStatus) => void;

  // Fetch operations
  fetchDocuments: () => Promise<void>;
  fetchFolders: () => Promise<void>;
  fetchDocumentDetail: (docId: string) => Promise<Document | null>;

  // CRUD operations
  deleteDocument: (docId: string) => Promise<boolean>;
  deleteFolder: (folderId: string) => Promise<boolean>;
  deleteMultipleItems: (ids: string[]) => Promise<void>;
  createFolder: (name: string) => Promise<boolean>;

  // Callbacks from parent
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderClick?: (folderName: string | null) => void;
  onFolderCreate?: (folderName: string) => void;
  onRefresh?: () => void;
  onViewInPDFViewer?: (documentId: string) => void;
}

const DocumentsContext = createContext<DocumentsContextType | undefined>(undefined);

export function useDocumentsV2() {
  const context = useContext(DocumentsContext);
  if (!context) {
    throw new Error("useDocumentsV2 must be used within DocumentsProviderV2");
  }
  return context;
}

interface DocumentsProviderV2Props {
  children: React.ReactNode;
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

export function DocumentsProviderV2({
  children,
  apiBaseUrl,
  authToken,
  initialFolder = null,
  onDocumentUpload,
  onDocumentDelete,
  onDocumentClick,
  onFolderClick,
  onFolderCreate,
  onRefresh,
  onViewInPDFViewer,
}: DocumentsProviderV2Props) {
  // Core state
  const [documents, setDocuments] = useState<Document[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [currentFolder, setCurrentFolder] = useState<string | null>(initialFolder);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [loadingStates, setLoadingStates] = useState<Map<string, ProcessingStatus>>(new Map());

  // Track mounted state
  const isMountedRef = useRef(false);

  // Get header context for breadcrumbs and right content
  const { setCustomBreadcrumbs, setRightContent } = useHeader();

  // Initialize API client
  const api = useMemo(() => new DocumentsApi({ apiBaseUrl, authToken }), [apiBaseUrl, authToken]);

  // Update document status
  const updateDocumentStatus = useCallback((docId: string, status: ProcessingStatus) => {
    setLoadingStates(prev => {
      const next = new Map(prev);
      next.set(docId, status);
      return next;
    });

    setDocuments(prev =>
      prev.map(doc =>
        doc.external_id === docId ? { ...doc, system_metadata: { ...doc.system_metadata, status } } : doc
      )
    );
  }, []);

  // Fetch folders from API
  const fetchFolders = useCallback(async () => {
    const folderData = await api.fetchFolders();
    setFolders(folderData);
  }, [api]);

  // Fetch documents from API
  const fetchDocuments = useCallback(async () => {
    try {
      let docs: Document[] = [];

      // Root: show unorganized documents only
      if (currentFolder === null) {
        docs = await api.fetchUnorganizedDocuments();
      }
      // All: fetch all documents
      else if (currentFolder === "all") {
        docs = await api.fetchAllDocuments();
      }
      // Specific folder: fetch documents in that folder
      else {
        // Ensure folders are loaded first
        if (folders.length === 0) {
          await fetchFolders();
        }
        docs = await api.fetchDocumentsInFolder(currentFolder);
      }

      setDocuments(docs);

      // Initialize loading states for processing documents
      const newLoadingStates = new Map<string, ProcessingStatus>();
      docs.forEach(doc => {
        const status = doc.system_metadata?.status as string;
        if (status === "processing" || status === "failed") {
          newLoadingStates.set(doc.external_id, status as ProcessingStatus);
        }
      });
      setLoadingStates(newLoadingStates);
    } catch (error) {
      console.error("[DocumentsProviderV2] Error fetching documents:", error);
      setDocuments([]);
    } finally {
      setIsLoading(false);
    }
  }, [api, currentFolder, folders, fetchFolders]);

  // Fetch document detail
  const fetchDocumentDetail = useCallback(
    async (docId: string): Promise<Document | null> => {
      return await api.fetchDocumentDetail(docId);
    },
    [api]
  );

  // Delete document
  const deleteDocument = useCallback(
    async (docId: string): Promise<boolean> => {
      const success = await api.deleteDocument(docId);

      if (success) {
        // Remove from local state and call callback
        setDocuments(prev => {
          const doc = prev.find(d => d.external_id === docId);
          if (doc && onDocumentDelete) {
            onDocumentDelete(doc.filename || doc.external_id);
          }
          return prev.filter(doc => doc.external_id !== docId);
        });
      }

      return success;
    },
    [api, onDocumentDelete]
  );

  // Delete folder
  const deleteFolder = useCallback(
    async (folderId: string): Promise<boolean> => {
      const success = await api.deleteFolder(folderId);

      if (success) {
        // Refresh folders list
        await fetchFolders();
        // If we're currently in the deleted folder, go back to root
        const folder = folders.find(f => f.id === folderId);
        if (folder && currentFolder === folder.name) {
          setCurrentFolder(null);
        }
      }

      return success;
    },
    [api, fetchFolders, folders, currentFolder]
  );

  // Delete multiple items (documents and folders)
  const deleteMultipleItems = useCallback(
    async (ids: string[]) => {
      // Delete all items
      const deletePromises = ids.map(async id => {
        // Try to delete as document first
        const success = await api.deleteDocument(id);
        if (!success) {
          // If not a document, try as folder
          await api.deleteFolder(id);
        }
      });

      await Promise.all(deletePromises);

      // Clear selection and refresh data
      setSelectedIds(new Set());
      await Promise.all([fetchDocuments(), fetchFolders()]);
    },
    [api, fetchDocuments, fetchFolders]
  );

  // Create folder
  const createFolder = useCallback(
    async (name: string): Promise<boolean> => {
      const folder = await api.createFolder(name);

      if (folder) {
        await fetchFolders();
        if (onFolderCreate) {
          onFolderCreate(name);
        }
        return true;
      }

      return false;
    },
    [api, fetchFolders, onFolderCreate]
  );

  // Combine folders and documents for table display
  const tableItems = React.useMemo<FileSystemItem[]>(() => {
    const items: FileSystemItem[] = [];

    // If at root level, show folders first
    if (!currentFolder) {
      folders.forEach(folder => {
        items.push({ type: "folder", data: folder });
      });
    }

    // Then add documents (filtered by search if needed)
    const filteredDocs = searchQuery
      ? documents.filter(doc => {
          const name = doc.filename || doc.external_id;
          return name.toLowerCase().includes(searchQuery.toLowerCase());
        })
      : documents;

    filteredDocs.forEach(doc => {
      items.push({ type: "document", data: doc });
    });

    return items;
  }, [folders, documents, currentFolder, searchQuery]);

  // Initial load effect - fetch folders once
  useEffect(() => {
    if (!isMountedRef.current) {
      fetchFolders().then(() => {
        isMountedRef.current = true;
      });
    }
  }, [fetchFolders]);

  // Folder change effect - clear documents and fetch new ones
  useEffect(() => {
    // Clear documents and show loading when folder changes
    setDocuments([]);
    setIsLoading(true);
    setSelectedIds(new Set());
    setSelectedDocument(null);

    // Fetch new documents after a small delay to ensure UI updates
    const timeoutId = setTimeout(() => {
      fetchDocuments();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [currentFolder, fetchDocuments]);

  // Update breadcrumbs and header buttons when folder changes
  useEffect(() => {
    const breadcrumbs = currentFolder
      ? [
          {
            label: "Documents V2",
            onClick: () => {
              setCurrentFolder(null);
              onFolderClick?.(null);
            },
          },
          {
            label: currentFolder === "all" ? "All Documents" : currentFolder,
            current: true,
          },
        ]
      : [{ label: "Documents V2", current: true }];

    setCustomBreadcrumbs(breadcrumbs);

    // Set header buttons
    const rightContent = currentFolder ? (
      // Folder view controls
      <div className="flex items-center gap-2">
        {selectedIds.size > 0 && (
          <Button
            variant="outline"
            size="icon"
            onClick={async () => {
              if (selectedIds.size > 0) {
                await deleteMultipleItems(Array.from(selectedIds));
              }
            }}
            className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
            title={`Delete ${selectedIds.size} selected item${selectedIds.size > 1 ? "s" : ""}`}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}

        <Button
          variant="outline"
          size="sm"
          onClick={async () => {
            setIsLoading(true);
            setDocuments([]);
            await Promise.all([fetchDocuments(), fetchFolders()]);
            onRefresh?.();
          }}
          title="Refresh documents"
        >
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>

        <Button
          variant="default"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openUploadDialog");
            window.dispatchEvent(event);
          }}
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </div>
    ) : (
      // Root level controls
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openNewFolderDialog");
            window.dispatchEvent(event);
          }}
        >
          <FolderPlus className="mr-2 h-4 w-4" />
          New Folder
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={async () => {
            setIsLoading(true);
            setDocuments([]);
            await Promise.all([fetchDocuments(), fetchFolders()]);
            onRefresh?.();
          }}
          title="Refresh documents"
        >
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>

        <Button
          variant="default"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openUploadDialog");
            window.dispatchEvent(event);
          }}
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </div>
    );

    setRightContent(rightContent);

    // Cleanup on unmount
    return () => {
      setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [
    currentFolder,
    selectedIds.size,
    setCustomBreadcrumbs,
    setRightContent,
    onFolderClick,
    deleteMultipleItems,
    fetchDocuments,
    fetchFolders,
    onRefresh,
  ]);

  const value: DocumentsContextType = {
    // API configuration
    apiBaseUrl,
    authToken,

    // State
    documents,
    folders,
    currentFolder,
    selectedIds,
    selectedDocument,
    searchQuery,
    isLoading,
    loadingStates,
    tableItems,

    // Actions
    setDocuments,
    setFolders,
    setCurrentFolder,
    setSelectedIds,
    setSelectedDocument,
    setSearchQuery,
    setIsLoading,
    updateDocumentStatus,

    // Fetch operations
    fetchDocuments,
    fetchFolders,
    fetchDocumentDetail,

    // CRUD operations
    deleteDocument,
    deleteFolder,
    deleteMultipleItems,
    createFolder,

    // Callbacks
    onDocumentUpload,
    onDocumentDelete,
    onDocumentClick,
    onFolderClick,
    onFolderCreate,
    onRefresh,
    onViewInPDFViewer,
  };

  return <DocumentsContext.Provider value={value}>{children}</DocumentsContext.Provider>;
}
