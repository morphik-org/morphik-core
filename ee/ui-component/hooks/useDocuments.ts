import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { Document, FolderSummary, Folder } from "@/components/types";

// Global cache for documents by folder
const documentsCache = new Map<string, { documents: Document[]; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Cache for folder details (document IDs)
const folderDetailsCache = new Map<string, string[]>();

export const clearDocumentsCache = (cacheKey?: string) => {
  if (cacheKey) {
    documentsCache.delete(cacheKey);
  } else {
    documentsCache.clear();
    folderDetailsCache.clear();
  }
};

interface UseDocumentsProps {
  apiBaseUrl: string;
  authToken: string | null;
  selectedFolder: string | null;
  folders: FolderSummary[];
}

interface UseDocumentsReturn {
  documents: Document[];
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  addOptimisticDocument: (doc: Document) => void;
  updateOptimisticDocument: (id: string, updates: Partial<Document>) => void;
  removeOptimisticDocument: (id: string) => void;
}

export function useDocuments({
  apiBaseUrl,
  authToken,
  selectedFolder,
  folders,
}: UseDocumentsProps): UseDocumentsReturn {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [optimisticDocuments, setOptimisticDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const isMountedRef = useRef(true);
  const previousFoldersLength = useRef(folders.length);
  const hasInitiallyFetched = useRef(false);

  const fetchDocuments = useCallback(
    async (forceRefresh = false) => {
      const cacheKey = `${apiBaseUrl}-${selectedFolder || "all"}`;
      const cached = documentsCache.get(cacheKey);

      // Check if we have valid cached data
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setDocuments(cached.documents);
        return;
      }

      if (!selectedFolder) {
        return;
      }

      try {
        setLoading(true);
        setError(null);

        let documentsToFetch: Document[] = [];

        if (selectedFolder === "all") {
          // Fetch all documents
          const response = await fetch(`${apiBaseUrl}/documents`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
            body: JSON.stringify({}),
          });

          if (!response.ok) {
            throw new Error(`Failed to fetch documents: ${response.statusText}`);
          }

          documentsToFetch = await response.json();
        } else {
          // Fetch documents for a specific folder
          const targetFolder = folders.find(folder => folder.name === selectedFolder);

          if (!targetFolder) {
            documentsToFetch = [];
          } else {
            // Get document IDs from cache or fetch
            let docIds = folderDetailsCache.get(targetFolder.id);

            if (!docIds) {
              const detailResp = await fetch(`${apiBaseUrl}/folders/${targetFolder.id}`, {
                headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
              });

              if (!detailResp.ok) {
                throw new Error(`Failed to fetch folder detail: ${detailResp.statusText}`);
              }

              const detail: Folder = await detailResp.json();
              docIds = Array.isArray(detail.document_ids) ? detail.document_ids : [];

              // Cache folder details
              folderDetailsCache.set(targetFolder.id, docIds);
            }

            if (docIds.length === 0) {
              documentsToFetch = [];
            } else {
              // Fetch document details via batch API
              const response = await fetch(`${apiBaseUrl}/batch/documents`, {
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
            }
          }
        }

        // Process documents (add status if needed)
        const processedData = documentsToFetch.map((doc: Document) => {
          if (!doc.system_metadata) {
            doc.system_metadata = {};
          }
          if (!doc.system_metadata.status && doc.folder_name) {
            doc.system_metadata.status = "processing";
          }
          return doc;
        });

        // Update cache
        documentsCache.set(cacheKey, {
          documents: processedData,
          timestamp: Date.now(),
        });

        if (isMountedRef.current) {
          setDocuments(processedData);
        }
      } catch (err) {
        console.error("Failed to fetch documents:", err);
        if (isMountedRef.current) {
          setError(err instanceof Error ? err : new Error("Failed to fetch documents"));
          setDocuments([]);
        }
      } finally {
        if (isMountedRef.current) {
          setLoading(false);
        }
      }
    },
    [apiBaseUrl, authToken, selectedFolder, folders]
  );

  // Reset the initial fetch flag when folder changes
  useEffect(() => {
    hasInitiallyFetched.current = false;
  }, [selectedFolder]);

  useEffect(() => {
    isMountedRef.current = true;

    // Skip if we don't have a selected folder
    if (!selectedFolder) {
      return;
    }

    // For "all" documents, fetch immediately
    if (selectedFolder === "all") {
      if (!hasInitiallyFetched.current) {
        fetchDocuments();
        hasInitiallyFetched.current = true;
      }
    }
    // For specific folders, only fetch if folders are loaded
    else if (folders.length > 0) {
      // Only fetch if we haven't fetched for this folder yet
      if (!hasInitiallyFetched.current) {
        fetchDocuments();
        hasInitiallyFetched.current = true;
      }
    }

    // Update the previous folders length
    previousFoldersLength.current = folders.length;

    return () => {
      isMountedRef.current = false;
    };
  }, [fetchDocuments, selectedFolder, folders.length]);

  const refresh = useCallback(async () => {
    // Clear cache for current selection
    const cacheKey = `${apiBaseUrl}-${selectedFolder || "all"}`;
    documentsCache.delete(cacheKey);

    // Also clear folder details cache if needed
    if (selectedFolder && selectedFolder !== "all") {
      const targetFolder = folders.find(folder => folder.name === selectedFolder);
      if (targetFolder) {
        folderDetailsCache.delete(targetFolder.id);
      }
    }

    // Clear optimistic documents on refresh
    setOptimisticDocuments([]);

    await fetchDocuments(true);
  }, [apiBaseUrl, selectedFolder, folders, fetchDocuments]);

  // Optimistic update functions
  const addOptimisticDocument = useCallback((doc: Document) => {
    setOptimisticDocuments(prev => [...prev, doc]);
  }, []);

  const updateOptimisticDocument = useCallback((id: string, updates: Partial<Document>) => {
    setOptimisticDocuments(prev => prev.map(doc => (doc.external_id === id ? { ...doc, ...updates } : doc)));
  }, []);

  const removeOptimisticDocument = useCallback((id: string) => {
    setOptimisticDocuments(prev => prev.filter(doc => doc.external_id !== id));
  }, []);

  // Merge regular documents with optimistic documents
  const mergedDocuments = useMemo(() => {
    // Create a map to track document IDs to avoid duplicates
    const docMap = new Map<string, Document>();

    // Add regular documents first
    documents.forEach(doc => docMap.set(doc.external_id, doc));

    // Add or update with optimistic documents
    optimisticDocuments.forEach(doc => docMap.set(doc.external_id, doc));

    return Array.from(docMap.values());
  }, [documents, optimisticDocuments]);

  return {
    documents: mergedDocuments,
    loading,
    error,
    refresh,
    addOptimisticDocument,
    updateOptimisticDocument,
    removeOptimisticDocument,
  };
}
