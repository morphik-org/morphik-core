import { useState, useEffect, useCallback, useRef } from "react";
import { Document } from "@/components/types";

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

      const cacheKey = `${apiBaseUrl}-unorganized`;
      const cached = unorganizedDocumentsCache.get(cacheKey);

      // Check if we have valid cached data
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setUnorganizedDocuments(cached.documents);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        // Fetch all documents first
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

        const allDocuments: Document[] = await response.json();

        // Filter for unorganized documents (those without folder_name or with null/empty folder_name)
        const unorganized = allDocuments.filter(doc => !doc.folder_name || doc.folder_name.trim() === "");

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
