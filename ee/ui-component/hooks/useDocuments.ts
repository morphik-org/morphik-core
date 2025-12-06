import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { Document, FolderSummary } from "../components/types";

// Global cache for documents by folder/scope
interface CachedDocuments {
  documents: Document[];
  timestamp: number;
  totalCount?: number | null;
  hasMore: boolean;
  nextSkip: number | null;
  skip: number;
  limit: number;
  returnedCount: number;
}

const documentsCache = new Map<string, CachedDocuments>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Cache for folder details (document IDs)
const folderDetailsCache = new Map<string, string[]>();

export const clearDocumentsCache = (cacheKey?: string) => {
  if (cacheKey) {
    for (const key of Array.from(documentsCache.keys())) {
      if (key.startsWith(cacheKey)) {
        documentsCache.delete(key);
      }
    }
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
  documentFilters?: Record<string, unknown>;
  pageSize?: number;
  fields?: string[];
  includeTotalCount?: boolean;
  includeStatusCounts?: boolean;
  includeFolderCounts?: boolean;
  sortBy?: "created_at" | "updated_at" | "filename" | "external_id";
  sortDirection?: "asc" | "desc";
}

interface UseDocumentsReturn {
  documents: Document[];
  loading: boolean;
  loadingMore: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  loadMore: () => Promise<void>;
  hasMore: boolean;
  totalCount: number | null;
  pageInfo: {
    skip: number;
    limit: number;
    returnedCount: number;
    totalCount: number | null;
    hasMore: boolean;
    nextSkip: number | null;
  };
  goToPage: (skip: number) => Promise<void>;
  goToNextPage: () => Promise<void>;
  goToPreviousPage: () => Promise<void>;
  setPageSize: (limit: number) => Promise<void>;
  addOptimisticDocument: (doc: Document) => void;
  updateOptimisticDocument: (id: string, updates: Partial<Document>) => void;
  removeOptimisticDocument: (id: string) => void;
}

export function useDocuments({
  apiBaseUrl,
  authToken,
  selectedFolder,
  folders,
  documentFilters,
  pageSize = 100,
  fields,
  includeTotalCount = true,
  includeStatusCounts = false,
  includeFolderCounts = false,
  sortBy = "updated_at",
  sortDirection = "desc",
}: UseDocumentsProps): UseDocumentsReturn {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [optimisticDocuments, setOptimisticDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [nextSkip, setNextSkip] = useState<number | null>(null);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const [limit, setLimit] = useState<number>(pageSize);
  const [currentSkip, setCurrentSkip] = useState(0);
  const [returnedCount, setReturnedCount] = useState(0);
  const isMountedRef = useRef(true);
  const hasInitiallyFetched = useRef(false);
  const statusPollInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    setLimit(prev => (prev === pageSize ? prev : pageSize));
  }, [pageSize]);

  const getScopeKey = useCallback((folder: string | null) => {
    if (folder === null) {
      return "root";
    }
    if (!folder) {
      return "all";
    }
    return folder;
  }, []);

  const scopeKey = useMemo(() => getScopeKey(selectedFolder), [getScopeKey, selectedFolder]);

  const scopeCachePrefix = useMemo(
    () => `${apiBaseUrl}-${authToken ?? "anon"}-${scopeKey}`,
    [apiBaseUrl, authToken, scopeKey]
  );

  const requestSignature = useMemo(
    () =>
      JSON.stringify({
        documentFilters,
        limit,
        fields,
        includeTotalCount,
        includeStatusCounts,
        includeFolderCounts,
        sortBy,
        sortDirection,
      }),
    [documentFilters, limit, fields, includeTotalCount, includeStatusCounts, includeFolderCounts, sortBy, sortDirection]
  );

  const listDocsUrl = useMemo(() => {
    const queryParams = new URLSearchParams();

    if (selectedFolder !== null && selectedFolder !== "all") {
      queryParams.append("folder_name", selectedFolder);
      queryParams.append("folder_depth", "-1");
    }

    const queryString = queryParams.toString();
    return `${apiBaseUrl}/documents/list_docs${queryString ? `?${queryString}` : ""}`;
  }, [apiBaseUrl, selectedFolder]);

  const normalizeDocument = useCallback((doc: Document): Document => {
    const systemMetadata = { ...(doc.system_metadata ?? {}) };
    const normalized: Document = {
      ...doc,
      metadata: doc.metadata ?? {},
      additional_metadata: doc.additional_metadata ?? {},
      system_metadata: systemMetadata,
    };

    if (!normalized.folder_path) {
      const pathFromMetadata =
        typeof systemMetadata.folder_path === "string" ? (systemMetadata.folder_path as string).trim() : "";
      if (pathFromMetadata) {
        normalized.folder_path = pathFromMetadata;
      } else if (typeof normalized.folder_name === "string") {
        normalized.folder_path = normalized.folder_name;
      }
    }

    if (!normalized.system_metadata.status && typeof normalized.folder_name === "string") {
      normalized.system_metadata.status = "processing";
    }

    return normalized;
  }, []);

  const makeListDocsRequest = useCallback(
    async (skip: number, requestedLimit: number) => {
      const normalizedLimit =
        Number.isFinite(requestedLimit) && requestedLimit > 0
          ? Math.max(1, Math.floor(requestedLimit))
          : Math.max(1, Math.floor(limit));

      const requestBody: Record<string, unknown> = {
        skip,
        limit: normalizedLimit,
        return_documents: true,
        include_total_count: includeTotalCount,
        include_status_counts: includeStatusCounts,
        include_folder_counts: includeFolderCounts,
        sort_by: sortBy,
        sort_direction: sortDirection,
      };

      if (documentFilters && Object.keys(documentFilters).length > 0) {
        requestBody.document_filters = documentFilters;
      }

      if (fields && fields.length > 0) {
        requestBody.fields = fields;
      }

      const response = await fetch(listDocsUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`);
      }

      return response.json();
    },
    [
      listDocsUrl,
      authToken,
      limit,
      includeTotalCount,
      includeStatusCounts,
      includeFolderCounts,
      sortBy,
      sortDirection,
      documentFilters,
      fields,
    ]
  );

  const fetchDocuments = useCallback(
    async (forceRefresh = false, skipOverride?: number, limitOverride?: number, treatAsPageTransition = false) => {
      const targetSkip = typeof skipOverride === "number" && skipOverride >= 0 ? Math.floor(skipOverride) : 0;
      const targetLimit =
        typeof limitOverride === "number" && limitOverride > 0 ? Math.max(1, Math.floor(limitOverride)) : limit;

      const effectiveSignature = JSON.stringify({
        documentFilters,
        limit: targetLimit,
        fields,
        includeTotalCount,
        includeStatusCounts,
        includeFolderCounts,
        sortBy,
        sortDirection,
      });

      const pageCacheKey = `${apiBaseUrl}-${authToken ?? "anon"}-${scopeKey}-${effectiveSignature}-skip-${targetSkip}`;
      const cached = documentsCache.get(pageCacheKey);

      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        if (isMountedRef.current) {
          setDocuments(cached.documents);
          setHasMore(cached.hasMore);
          setNextSkip(cached.nextSkip);
          setTotalCount(includeTotalCount ? (cached.totalCount ?? cached.documents.length) : null);
          setCurrentSkip(cached.skip);
          setReturnedCount(cached.returnedCount);
          setLimit(cached.limit);
        }
        return;
      }

      if (isMountedRef.current) {
        if (treatAsPageTransition) {
          setLoadingMore(true);
        } else {
          setLoading(true);
        }
      }

      try {
        setError(null);

        const result = await makeListDocsRequest(targetSkip, targetLimit);
        const rawDocuments: Document[] = Array.isArray(result.documents) ? result.documents : [];
        const processedData = rawDocuments.map(normalizeDocument);
        const resolvedReturnedCount =
          typeof result.returned_count === "number" && result.returned_count >= 0
            ? result.returned_count
            : processedData.length;
        const resolvedHasMore = Boolean(result.has_more);
        const resolvedNextSkip =
          typeof result.next_skip === "number"
            ? result.next_skip
            : resolvedHasMore
              ? targetSkip + resolvedReturnedCount
              : null;
        const resolvedTotalCount =
          includeTotalCount && typeof result.total_count === "number"
            ? result.total_count
            : includeTotalCount
              ? targetSkip + resolvedReturnedCount
              : null;

        documentsCache.set(pageCacheKey, {
          documents: processedData,
          timestamp: Date.now(),
          hasMore: resolvedHasMore,
          nextSkip: resolvedNextSkip,
          totalCount: resolvedTotalCount,
          skip: targetSkip,
          limit: targetLimit,
          returnedCount: resolvedReturnedCount,
        });

        if (isMountedRef.current) {
          setDocuments(processedData);
          setHasMore(resolvedHasMore);
          setNextSkip(resolvedNextSkip);
          setTotalCount(includeTotalCount ? resolvedTotalCount : null);
          setCurrentSkip(targetSkip);
          setReturnedCount(resolvedReturnedCount);
          setLimit(targetLimit);
        }
      } catch (err) {
        console.error("Failed to fetch documents:", err);
        if (isMountedRef.current) {
          setError(err instanceof Error ? err : new Error("Failed to fetch documents"));
          setDocuments([]);
          setHasMore(false);
          setNextSkip(null);
          setTotalCount(includeTotalCount ? 0 : null);
          setReturnedCount(0);
          setCurrentSkip(targetSkip);
        }
      } finally {
        if (isMountedRef.current) {
          if (treatAsPageTransition) {
            setLoadingMore(false);
          } else {
            setLoading(false);
          }
        }
      }
    },
    [
      apiBaseUrl,
      authToken,
      documentFilters,
      fields,
      includeFolderCounts,
      includeStatusCounts,
      includeTotalCount,
      limit,
      makeListDocsRequest,
      normalizeDocument,
      scopeKey,
      sortBy,
      sortDirection,
    ]
  );

  const goToPage = useCallback(
    async (targetSkip: number, options?: { limit?: number; treatAsPageTransition?: boolean }) => {
      const normalizedSkip = Number.isFinite(targetSkip) && targetSkip >= 0 ? Math.floor(targetSkip) : 0;
      const effectiveLimit =
        options && typeof options.limit === "number" && options.limit > 0
          ? Math.max(1, Math.floor(options.limit))
          : limit;

      await fetchDocuments(false, normalizedSkip, effectiveLimit, options?.treatAsPageTransition ?? true);
    },
    [fetchDocuments, limit]
  );

  const goToNextPage = useCallback(async () => {
    const reachedEnd = typeof totalCount === "number" ? currentSkip + returnedCount >= totalCount : !hasMore;

    if (reachedEnd) {
      return;
    }

    const targetSkip = typeof nextSkip === "number" ? nextSkip : currentSkip + limit;

    await fetchDocuments(false, targetSkip, limit, true);
  }, [currentSkip, fetchDocuments, hasMore, limit, nextSkip, returnedCount, totalCount]);

  const goToPreviousPage = useCallback(async () => {
    if (currentSkip <= 0) {
      return;
    }

    const prevSkip = Math.max(0, currentSkip - limit);
    await fetchDocuments(false, prevSkip, limit, true);
  }, [currentSkip, fetchDocuments, limit]);

  const loadMore = useCallback(async () => {
    await goToNextPage();
  }, [goToNextPage]);

  const setPageSizeAndFetch = useCallback(
    async (nextLimit: number) => {
      const normalizedLimit = Number.isFinite(nextLimit) && nextLimit > 0 ? Math.max(1, Math.floor(nextLimit)) : limit;

      if (normalizedLimit === limit) {
        return;
      }

      setLimit(normalizedLimit);
      setCurrentSkip(0);
      setNextSkip(null);
      setHasMore(false);
      setReturnedCount(0);

      clearDocumentsCache(scopeCachePrefix);

      await fetchDocuments(true, 0, normalizedLimit, true);
    },
    [scopeCachePrefix, fetchDocuments, limit]
  );

  // Reset pagination metadata when folder or filters change
  useEffect(() => {
    hasInitiallyFetched.current = false;
    setHasMore(false);
    setNextSkip(null);
    setCurrentSkip(0);
    setReturnedCount(0);
    if (includeTotalCount) {
      setTotalCount(null);
    }
  }, [selectedFolder, requestSignature, includeTotalCount]);

  useEffect(() => {
    hasInitiallyFetched.current = false;
    setDocuments([]);
    setOptimisticDocuments([]);
    setHasMore(false);
    setNextSkip(null);
    setTotalCount(null);
    setCurrentSkip(0);
    setReturnedCount(0);
  }, [apiBaseUrl, authToken]);

  useEffect(() => {
    isMountedRef.current = true;

    const shouldFetch =
      selectedFolder === null || selectedFolder === "all" || (selectedFolder !== null && folders.length > 0);

    if (shouldFetch && !hasInitiallyFetched.current) {
      fetchDocuments(false, 0, limit, false);
      hasInitiallyFetched.current = true;
    }

    return () => {
      isMountedRef.current = false;
    };
  }, [fetchDocuments, selectedFolder, folders.length]);

  const refresh = useCallback(async () => {
    clearDocumentsCache(scopeCachePrefix);

    // Also clear folder details cache if needed
    if (selectedFolder && selectedFolder !== "all") {
      const targetFolder = folders.find(
        folder => folder.full_path === selectedFolder || folder.name === selectedFolder
      );
      if (targetFolder) {
        folderDetailsCache.delete(targetFolder.id);
      }
    }

    // Clear optimistic documents on refresh
    setOptimisticDocuments([]);
    await fetchDocuments(true, currentSkip, limit, false);
  }, [scopeCachePrefix, selectedFolder, folders, fetchDocuments, currentSkip, limit]);

  // Optimistic update functions
  const addOptimisticDocument = useCallback((doc: Document) => {
    setOptimisticDocuments(prev => [...prev, doc]);
  }, []);

  const updateOptimisticDocument = useCallback((id: string, updates: Partial<Document>) => {
    setOptimisticDocuments(prev =>
      prev.map(doc => {
        if (doc.external_id !== id) {
          return doc;
        }

        const updatedDoc = { ...doc, ...updates };

        // Ensure nested maps merge instead of replace when provided
        if (updates.system_metadata) {
          const mergedSystemMetadata: Record<string, unknown> = {
            ...((doc.system_metadata ?? {}) as Record<string, unknown>),
          };
          Object.entries(updates.system_metadata).forEach(([key, value]) => {
            if (value === undefined) {
              delete mergedSystemMetadata[key as keyof typeof mergedSystemMetadata];
            } else {
              mergedSystemMetadata[key as keyof typeof mergedSystemMetadata] = value;
            }
          });
          updatedDoc.system_metadata = mergedSystemMetadata;
        }

        if (updates.metadata) {
          const mergedMetadata: Record<string, unknown> = {
            ...((doc.metadata ?? {}) as Record<string, unknown>),
          };
          Object.entries(updates.metadata).forEach(([key, value]) => {
            if (value === undefined) {
              delete mergedMetadata[key as keyof typeof mergedMetadata];
            } else {
              mergedMetadata[key as keyof typeof mergedMetadata] = value;
            }
          });
          updatedDoc.metadata = mergedMetadata;
        }

        if (updates.additional_metadata) {
          const mergedAdditionalMetadata: Record<string, unknown> = {
            ...((doc.additional_metadata ?? {}) as Record<string, unknown>),
          };
          Object.entries(updates.additional_metadata).forEach(([key, value]) => {
            if (value === undefined) {
              delete mergedAdditionalMetadata[key as keyof typeof mergedAdditionalMetadata];
            } else {
              mergedAdditionalMetadata[key as keyof typeof mergedAdditionalMetadata] = value;
            }
          });
          updatedDoc.additional_metadata = mergedAdditionalMetadata;
        }

        return updatedDoc;
      })
    );
  }, []);

  const removeOptimisticDocument = useCallback((id: string) => {
    setOptimisticDocuments(prev => prev.filter(doc => doc.external_id !== id));
  }, []);

  // Poll for status updates of processing documents
  const pollDocumentStatuses = useCallback(async () => {
    const processingDocs = documents.filter(doc => doc.system_metadata?.status === "processing");

    if (processingDocs.length === 0 || !authToken) {
      return;
    }

    try {
      // Fetch status updates for all processing documents
      const statusPromises = processingDocs.map(async doc => {
        const response = await fetch(`${apiBaseUrl}/documents/${doc.external_id}/status`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });

        if (response.ok) {
          const status = await response.json();
          return { id: doc.external_id, status };
        }
        return null;
      });

      const statusUpdates = await Promise.all(statusPromises);

      // Check if any documents have completed processing
      const completedDocIds = statusUpdates
        .filter(u => u && u.status && u.status.status === "completed")
        .map(u => u!.id);

      // Update documents with new status information
      setDocuments(prevDocs => {
        return prevDocs.map(doc => {
          const update = statusUpdates.find(u => u && u.id === doc.external_id);
          if (update && update.status) {
            // Build updated system_metadata
            const updatedSystemMetadata: any = {
              ...doc.system_metadata,
              status: update.status.status,
            };

            // Only add progress if still processing
            if (update.status.status === "processing" && update.status.progress) {
              updatedSystemMetadata.progress = update.status.progress;
            } else {
              // Remove progress field when completed or failed
              delete updatedSystemMetadata.progress;
            }

            // Add error if failed
            if (update.status.error) {
              updatedSystemMetadata.error = update.status.error;
            }

            return {
              ...doc,
              system_metadata: updatedSystemMetadata,
            };
          }
          return doc;
        });
      });

      // If any documents completed, trigger a full refresh
      if (completedDocIds.length > 0) {
        // Small delay to ensure backend has finished updating
        setTimeout(() => {
          fetchDocuments(true, currentSkip, limit, true);
        }, 1000);
      }
    } catch (error) {
      console.error("Error polling document statuses:", error);
    }
  }, [documents, apiBaseUrl, authToken, fetchDocuments]);

  // Set up polling interval
  useEffect(() => {
    // Clear existing interval
    if (statusPollInterval.current) {
      clearInterval(statusPollInterval.current);
    }

    // Check if we have any processing documents
    const hasProcessingDocs = documents.some(doc => doc.system_metadata?.status === "processing");

    if (hasProcessingDocs) {
      // Poll every 2 seconds
      statusPollInterval.current = setInterval(pollDocumentStatuses, 2000);
    }

    return () => {
      if (statusPollInterval.current) {
        clearInterval(statusPollInterval.current);
      }
    };
  }, [documents, pollDocumentStatuses]);

  // Merge regular documents with optimistic documents
  const getDocumentFolder = useCallback((doc: Document) => {
    const fromPath =
      typeof doc.folder_path === "string"
        ? doc.folder_path.trim()
        : typeof (doc.system_metadata as Record<string, unknown> | undefined)?.folder_path === "string"
          ? String((doc.system_metadata as Record<string, unknown>).folder_path).trim()
          : "";
    if (fromPath) {
      return fromPath;
    }

    const fromDoc = typeof doc.folder_name === "string" ? doc.folder_name.trim() : "";
    if (fromDoc) {
      return fromDoc;
    }

    const systemMetadata = (doc.system_metadata ?? {}) as Record<string, unknown>;
    const fromMetadataPath =
      typeof systemMetadata.folder_path === "string" ? (systemMetadata.folder_path as string).trim() : "";
    if (fromMetadataPath) {
      return fromMetadataPath;
    }

    const fromMetadata =
      typeof systemMetadata.folder_name === "string" ? (systemMetadata.folder_name as string).trim() : "";
    return fromMetadata;
  }, []);

  const relevantOptimisticDocuments = useMemo(() => {
    if (optimisticDocuments.length === 0) {
      return optimisticDocuments;
    }

    if (selectedFolder === null || selectedFolder === "all") {
      return optimisticDocuments;
    }

    const normalizedTarget = selectedFolder.trim();

    return optimisticDocuments.filter(doc => {
      const docFolder = getDocumentFolder(doc);
      if (normalizedTarget === "") {
        return docFolder === "";
      }
      return docFolder === normalizedTarget;
    });
  }, [optimisticDocuments, selectedFolder, getDocumentFolder]);

  const mergedDocuments = useMemo(() => {
    // Create a map to track document IDs to avoid duplicates
    const docMap = new Map<string, Document>();

    // Add optimistic documents first so API responses can overwrite them when ready
    relevantOptimisticDocuments.forEach(doc => docMap.set(doc.external_id, doc));

    // Add regular documents last to take precedence when IDs overlap
    documents.forEach(doc => docMap.set(doc.external_id, doc));

    return Array.from(docMap.values());
  }, [documents, relevantOptimisticDocuments]);

  // Drop optimistic entries once the real document shows up in fetched data
  useEffect(() => {
    if (optimisticDocuments.length === 0 || documents.length === 0) {
      return;
    }

    const filtered = optimisticDocuments.filter(
      optDoc => !documents.some(doc => doc.external_id === optDoc.external_id)
    );

    if (filtered.length !== optimisticDocuments.length) {
      setOptimisticDocuments(filtered);
    }
  }, [documents, optimisticDocuments]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (statusPollInterval.current) {
        clearInterval(statusPollInterval.current);
      }
    };
  }, []);

  const pageInfo = useMemo(
    () => ({
      skip: currentSkip,
      limit,
      returnedCount,
      totalCount,
      hasMore,
      nextSkip,
    }),
    [currentSkip, hasMore, limit, nextSkip, returnedCount, totalCount]
  );

  return {
    documents: mergedDocuments,
    loading,
    loadingMore,
    error,
    refresh,
    loadMore,
    hasMore,
    totalCount,
    pageInfo,
    goToPage,
    goToNextPage,
    goToPreviousPage,
    setPageSize: setPageSizeAndFetch,
    addOptimisticDocument,
    updateOptimisticDocument,
    removeOptimisticDocument,
  };
}
