import { useState, useEffect, useCallback } from "react";

interface LogEntry {
  timestamp: string;
  user_id: string;
  operation_type: string;
  status: string;
  tokens_used: number;
  duration_ms: number;
  metadata?: Record<string, unknown>;
  error?: string;
}

// Simple in-memory cache keyed by apiBaseUrl
const logsCache = new Map<string, { logs: LogEntry[]; timestamp: number }>();
const CACHE_DURATION = 60 * 1000; // 1 minute

export const clearLogsCache = (apiBaseUrl?: string) => {
  if (apiBaseUrl) logsCache.delete(apiBaseUrl);
  else logsCache.clear();
};

interface UseLogsProps {
  apiBaseUrl: string;
  authToken: string | null;
  limit?: number;
}

interface UseLogsReturn {
  logs: LogEntry[];
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

export function useLogs({ apiBaseUrl, authToken, limit = 100 }: UseLogsProps): UseLogsReturn {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchLogs = useCallback(
    async (forceRefresh = false) => {
      const cacheKey = `${apiBaseUrl}|${limit}`;
      const cached = logsCache.get(cacheKey);
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setLogs(cached.logs);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const res = await fetch(`${apiBaseUrl}/logs?limit=${limit}`, {
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        if (!res.ok) throw new Error(`Failed to fetch logs: ${res.statusText}`);
        const data = (await res.json()) as LogEntry[];
        logsCache.set(cacheKey, { logs: data, timestamp: Date.now() });
        setLogs(data);
      } catch (err) {
        console.error("Failed to fetch logs", err);
        setError(err instanceof Error ? err : new Error("Failed to fetch logs"));
      } finally {
        setLoading(false);
      }
    },
    [apiBaseUrl, authToken, limit]
  );

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  const refresh = useCallback(async () => {
    await fetchLogs(true);
  }, [fetchLogs]);

  return { logs, loading, error, refresh };
}
