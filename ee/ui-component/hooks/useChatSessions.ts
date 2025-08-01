import { useEffect, useState, useCallback } from "react";

interface ChatSessionMeta {
  chatId: string;
  title?: string;
  createdAt?: string;
  updatedAt?: string;
  lastMessage?: {
    role: string;
    content: string;
    agent_data?: {
      display_objects?: Array<{
        type: string;
        content: string;
        source?: string;
        caption?: string;
      }>;
      tool_history?: any[];
      sources?: any[];
    };
  } | null;
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

// Global cache for chat sessions
const chatSessionsCache = new Map<string, { sessions: ChatSessionMeta[]; timestamp: number }>();
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes - longer cache for better UX

export const clearChatSessionsCache = (apiBaseUrl?: string) => {
  if (apiBaseUrl) {
    chatSessionsCache.delete(apiBaseUrl);
  } else {
    chatSessionsCache.clear();
  }
};

export function useChatSessions({ apiBaseUrl, authToken, limit = 100 }: UseChatSessionsProps): UseChatSessionsReturn {
  const [sessions, setSessions] = useState<ChatSessionMeta[]>([]);
  const [isLoading, setIsLoading] = useState(true); // Start with loading state

  const fetchSessions = useCallback(
    async (forceRefresh = false) => {
      const cacheKey = `${apiBaseUrl}-${limit}`;
      const cached = chatSessionsCache.get(cacheKey);

      // Check if we have valid cached data
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setSessions(cached.sessions);
        setIsLoading(false);
        return;
      }

      if (!cached) {
        setIsLoading(true);
      }

      try {
        const res = await fetch(`${apiBaseUrl}/chats?limit=${limit}`, {
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        if (res.ok) {
          const data = await res.json();
          const transformedSessions = data.map((c: any) => ({
            chatId: c.chat_id,
            title: c.title,
            createdAt: c.created_at,
            updatedAt: c.updated_at,
            lastMessage: c.last_message ?? null,
          }));

          // Update cache
          chatSessionsCache.set(cacheKey, {
            sessions: transformedSessions,
            timestamp: Date.now(),
          });

          setSessions(transformedSessions);
        } else {
          console.error(`Failed to fetch chat sessions: ${res.status} ${res.statusText}`);
        }
      } catch (err) {
        console.error("Failed to fetch chat sessions", err);
      } finally {
        setIsLoading(false);
      }
    },
    [apiBaseUrl, authToken, limit]
  );

  useEffect(() => {
    if (apiBaseUrl) {
      fetchSessions();
    } else {
      setIsLoading(false);
    }
  }, [fetchSessions, apiBaseUrl]);

  const reload = useCallback(() => {
    fetchSessions(true);
  }, [fetchSessions]);

  return { sessions, isLoading, reload };
}

// New hook for PDF-specific chat session management

interface UsePDFChatSessionsProps {
  apiBaseUrl: string;
  authToken: string | null;
  documentName?: string;
}

interface UsePDFChatSessionsReturn {
  currentChatId: string | null;
  createNewSession: () => string;
}

export function usePDFChatSessions({
  apiBaseUrl,
  authToken,
  documentName,
}: UsePDFChatSessionsProps): UsePDFChatSessionsReturn {
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  // Create a new chat session for the current document
  const createNewSession = useCallback(() => {
    if (!documentName) return "";

    // Generate a truly unique chat ID for each new session
    const timestamp = Date.now();
    const randomId = Math.random().toString(36).substr(2, 9);
    const chatId = `pdf-${documentName}-${timestamp}-${randomId}`;

    setCurrentChatId(chatId);
    return chatId;
  }, [documentName]);

  // Initialize current session when document loads
  useEffect(() => {
    if (documentName && !currentChatId) {
      // Check if there's a saved active chat for this document in this browser session
      const activeKey = `morphik-active-chat-${documentName}`;
      const savedActiveChatId = sessionStorage.getItem(activeKey);

      if (savedActiveChatId) {
        // Resume the saved active chat
        setCurrentChatId(savedActiveChatId);
      } else {
        // Create a new session for this document
        createNewSession();
      }
    }
  }, [documentName, currentChatId, createNewSession]);

  // Save the current active chat ID to sessionStorage
  useEffect(() => {
    if (documentName && currentChatId) {
      const activeKey = `morphik-active-chat-${documentName}`;
      sessionStorage.setItem(activeKey, currentChatId);
    }
  }, [documentName, currentChatId]);

  return {
    currentChatId,
    createNewSession,
  };
}
