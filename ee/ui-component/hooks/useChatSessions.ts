import { useEffect, useState, useCallback } from "react";

interface ChatSessionMeta {
  chatId: string;
  createdAt?: string;
  updatedAt?: string;
  lastMessage?: { role: string; content: string } | null;
}

interface UseChatSessionsProps {
  apiBaseUrl: string;
  authToken: string | null;
  limit?: number;
}

interface UseChatSessionsReturn {
  sessions: ChatSessionMeta[];
  isLoading: boolean;
  reload: () => void;
}

export function useChatSessions({ apiBaseUrl, authToken, limit = 100 }: UseChatSessionsProps): UseChatSessionsReturn {
  const [sessions, setSessions] = useState<ChatSessionMeta[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchSessions = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/chats?limit=${limit}`, {
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });
      if (res.ok) {
        const data = await res.json();
        setSessions(
          data.map((c: any) => ({
            chatId: c.chat_id,
            createdAt: c.created_at,
            updatedAt: c.updated_at,
            lastMessage: c.last_message ?? null,
          }))
        );
      } else {
        console.error(`Failed to fetch chat sessions: ${res.status} ${res.statusText}`);
      }
    } catch (err) {
      console.error("Failed to fetch chat sessions", err);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl, authToken, limit]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  return { sessions, isLoading, reload: fetchSessions };
}
