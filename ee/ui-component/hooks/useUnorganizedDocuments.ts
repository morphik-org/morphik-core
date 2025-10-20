import { useState, useEffect, useCallback, useRef } from "react";
import { Document } from "../components/types";

// Global cache for unorganized documents
const unorganizedDocumentsCache = new Map<string, { documents: Document[]; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const clearUnorganizedDocumentsCache = (cacheKey?: string) => {
  if (cacheKey) {
    unorganizedDocumentsCache.delete(cacheKey);
  } else {
    unorganizedDocumentsCache.clear();
  }
};

interface UseUnorganizedDocumentsProps {
  apiBaseUrl: string;
  authToken: string | null;
  enabled: boolean; // Only fetch when enabled (i.e., when at root level)
}

interface UseUnorganizedDocumentsReturn {
  unorganizedDocuments: Document[];
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

export function useUnorganizedDocuments({
  apiBaseUrl,
  authToken,
  enabled,
}: UseUnorganizedDocumentsProps): UseUnorganizedDocumentsReturn {
  const [unorganizedDocuments, setUnorganizedDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const isMountedRef = useRef(true);

  const fetchUnorganizedDocuments = useCallback(
    async (forceRefresh = false) => {
      if (!enabled) {
        setUnorganizedDocuments([]);
        return;
      }

      const cacheKey = `${apiBaseUrl}-${authToken ?? "anon"}-unorganized`;
      const cached = unorganizedDocumentsCache.get(cacheKey);

      // Check if we have valid cached data
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setUnorganizedDocuments(cached.documents);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const PAGE_SIZE = 500;
        const aggregated: Document[] = [];
        const seenSkips = new Set<number>();
        let skip = 0;
        let hasMore = true;

        while (hasMore) {
          if (seenSkips.has(skip)) {
            console.warn("Detected repeated skip value when fetching unorganized documents, stopping pagination.", {
              skip,
            });
            break;
          }

          seenSkips.add(skip);

          const response = await fetch(`${apiBaseUrl}/documents/list_docs?folder_name=null`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
            body: JSON.stringify({
              skip,
              limit: PAGE_SIZE,
              return_documents: true,
              include_total_count: false,
              include_status_counts: false,
              include_folder_counts: false,
            }),
          });

          if (!response.ok) {
            throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`);
          }

          const result = await response.json();
          const rawDocuments: Document[] = Array.isArray(result?.documents) ? result.documents : [];
          const normalizedDocuments = rawDocuments.map(doc => {
            const normalized: Document = {
              ...doc,
              metadata: doc.metadata ?? {},
              additional_metadata: doc.additional_metadata ?? {},
              system_metadata: { ...(doc.system_metadata ?? {}) },
            };
            if (!normalized.system_metadata.status) {
              normalized.system_metadata.status = "processing";
            }
            return normalized;
          });

          aggregated.push(...normalizedDocuments);

          const returnedCount =
            typeof result?.returned_count === "number" && result.returned_count >= 0
              ? result.returned_count
              : normalizedDocuments.length;
          const nextSkip =
            typeof result?.next_skip === "number" ? result.next_skip : result?.has_more ? skip + returnedCount : null;

          hasMore = Boolean(result?.has_more) && returnedCount > 0 && nextSkip !== null && nextSkip !== skip;

          if (!hasMore) {
            break;
          }

          skip = nextSkip ?? skip + returnedCount;
        }

        const unorganized = aggregated;

        // Update cache
        unorganizedDocumentsCache.set(cacheKey, {
          documents: unorganized,
          timestamp: Date.now(),
        });

        if (isMountedRef.current) {
          setUnorganizedDocuments(unorganized);
        }
      } catch (err) {
        console.error("Failed to fetch unorganized documents:", err);
        if (isMountedRef.current) {
          setError(err instanceof Error ? err : new Error("Failed to fetch unorganized documents"));
          setUnorganizedDocuments([]);
        }
      } finally {
        if (isMountedRef.current) {
          setLoading(false);
        }
      }
    },
    [apiBaseUrl, authToken, enabled]
  );

  useEffect(() => {
    isMountedRef.current = true;

    if (enabled) {
      fetchUnorganizedDocuments();
    } else {
      setUnorganizedDocuments([]);
    }

    return () => {
      isMountedRef.current = false;
    };
  }, [fetchUnorganizedDocuments, enabled]);

  const refresh = useCallback(async () => {
    await fetchUnorganizedDocuments(true);
  }, [fetchUnorganizedDocuments]);

  return {
    unorganizedDocuments,
    loading,
    error,
    refresh,
  };
}
