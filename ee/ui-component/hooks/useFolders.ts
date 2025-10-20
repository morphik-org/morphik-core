import { useState, useEffect, useCallback, useMemo } from "react";
import { FolderSummary } from "../components/types";

// Global cache for folders
const foldersCache = new Map<string, { folders: FolderSummary[]; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const clearFoldersCache = (apiBaseUrl?: string) => {
  if (apiBaseUrl) {
    const prefix = `${apiBaseUrl}-`;
    for (const key of Array.from(foldersCache.keys())) {
      if (key.startsWith(prefix)) {
        foldersCache.delete(key);
      }
    }
  } else {
    foldersCache.clear();
  }
};

interface UseFoldersProps {
  apiBaseUrl: string;
  authToken: string | null;
  identifiers?: string[];
  documentFilters?: Record<string, unknown>;
  includeDocumentCount?: boolean;
  includeStatusCounts?: boolean;
}

interface UseFoldersReturn {
  folders: FolderSummary[];
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

export function useFolders({
  apiBaseUrl,
  authToken,
  identifiers,
  documentFilters,
  includeDocumentCount = true,
  includeStatusCounts = false,
}: UseFoldersProps): UseFoldersReturn {
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const requestSignature = useMemo(
    () =>
      JSON.stringify({
        identifiers: identifiers ? [...identifiers].sort() : undefined,
        documentFilters,
        includeDocumentCount,
        includeStatusCounts,
      }),
    [identifiers, documentFilters, includeDocumentCount, includeStatusCounts]
  );

  const cacheKey = useMemo(
    () => `${apiBaseUrl}-${authToken ?? "anon"}-${requestSignature}`,
    [apiBaseUrl, authToken, requestSignature]
  );

  const fetchFolders = useCallback(
    async (forceRefresh = false) => {
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

        const requestBody: Record<string, unknown> = {
          include_document_count: includeDocumentCount,
          include_status_counts: includeStatusCounts,
          include_documents: false,
        };

        if (identifiers && identifiers.length > 0) {
          requestBody.identifiers = identifiers;
        }

        if (documentFilters && Object.keys(documentFilters).length > 0) {
          requestBody.document_filters = documentFilters;
        }

        const response = await fetch(`${apiBaseUrl}/folders/details`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch folders: ${response.status} ${response.statusText}`);
        }

        const responseBody = await response.json();
        const folderEntries = Array.isArray(responseBody?.folders) ? responseBody.folders : [];
        const summaries: FolderSummary[] = folderEntries.map((entry: any) => {
          const folder = entry?.folder ?? {};
          const documentInfo = entry?.document_info ?? {};
          const systemMetadata = folder.system_metadata ?? {};
          const updatedAt = systemMetadata?.updated_at ?? systemMetadata?.created_at ?? undefined;

          const summary: FolderSummary & { document_ids?: string[] } = {
            id: folder.id,
            name: folder.name,
            description: folder.description ?? undefined,
            doc_count:
              documentInfo?.document_count ??
              (Array.isArray(folder.document_ids) ? folder.document_ids.length : undefined),
            updated_at: typeof updatedAt === "string" ? updatedAt : updatedAt ? String(updatedAt) : undefined,
          };

          if (Array.isArray(folder.document_ids)) {
            summary.document_ids = folder.document_ids;
          }

          return summary;
        });

        // Update cache
        foldersCache.set(cacheKey, {
          folders: summaries,
          timestamp: Date.now(),
        });

        setFolders(summaries);
      } catch (err) {
        console.error("Failed to fetch folders:", err);
        setError(err instanceof Error ? err : new Error("Failed to fetch folders"));
      } finally {
        setLoading(false);
      }
    },
    [apiBaseUrl, authToken, cacheKey, documentFilters, identifiers, includeDocumentCount, includeStatusCounts]
  );

  useEffect(() => {
    fetchFolders();
  }, [fetchFolders]);

  const refresh = useCallback(async () => {
    await fetchFolders(true);
  }, [fetchFolders]);

  return { folders, loading, error, refresh };
}
