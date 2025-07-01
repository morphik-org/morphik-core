import { useState, useEffect, useCallback } from "react";
import { FolderSummary } from "@/components/types";

// Global cache for folders
const foldersCache = new Map<string, { folders: FolderSummary[]; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const clearFoldersCache = (apiBaseUrl?: string) => {
  if (apiBaseUrl) {
    foldersCache.delete(apiBaseUrl);
  } else {
    foldersCache.clear();
  }
};

interface UseFoldersProps {
  apiBaseUrl: string;
  authToken: string | null;
}

interface UseFoldersReturn {
  folders: FolderSummary[];
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

export function useFolders({ apiBaseUrl, authToken }: UseFoldersProps): UseFoldersReturn {
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchFolders = useCallback(
    async (forceRefresh = false) => {
      const cacheKey = apiBaseUrl;
      const cached = foldersCache.get(cacheKey);

      // Check if we have valid cached data
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setFolders(cached.folders);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${apiBaseUrl}/folders/summary`, {
          method: "GET",
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch folders: ${response.statusText}`);
        }

        const data = (await response.json()) as FolderSummary[];

        // Update cache
        foldersCache.set(cacheKey, {
          folders: data,
          timestamp: Date.now(),
        });

        setFolders(data);
      } catch (err) {
        console.error("Failed to fetch folders:", err);
        setError(err instanceof Error ? err : new Error("Failed to fetch folders"));
      } finally {
        setLoading(false);
      }
    },
    [apiBaseUrl, authToken]
  );

  useEffect(() => {
    fetchFolders();
  }, [fetchFolders]);

  const refresh = useCallback(async () => {
    await fetchFolders(true);
  }, [fetchFolders]);

  return { folders, loading, error, refresh };
}
