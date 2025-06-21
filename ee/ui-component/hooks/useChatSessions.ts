import { useEffect, useState, useCallback, useRef } from "react";

interface ChatSessionMeta {
  chatId: string;
  createdAt?: string;
  updatedAt?: string;
  name?: string | null;
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
  searchQuery?: string;
}

interface UseChatSessionsReturn {
  sessions: ChatSessionMeta[];
  isLoading: boolean;
  reload: () => void;
  renameChat: (chatId: string, newName: string) => Promise<boolean>;
  deleteChat: (chatId: string) => Promise<boolean>;
  searchChats: (query: string) => Promise<ChatSessionMeta[]>;
}

export function useChatSessions({ apiBaseUrl, authToken, limit = 100, searchQuery }: UseChatSessionsProps): UseChatSessionsReturn {
  const [sessions, setSessions] = useState<ChatSessionMeta[]>([]);
  const [allSessions, setAllSessions] = useState<ChatSessionMeta[]>([]); // Store all sessions separately
  const [isLoading, setIsLoading] = useState(false);
  const searchTimeoutRef = useRef<NodeJS.Timeout>();

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
        const sessionsData = data.map((c: any) => ({
          chatId: c.chat_id,
          createdAt: c.created_at,
          updatedAt: c.updated_at,
          name: c.name ?? null,
          lastMessage: c.last_message ?? null,
        }));
        setAllSessions(sessionsData);
        setSessions(sessionsData);
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

  const renameChat = useCallback(
    async (chatId: string, newName: string): Promise<boolean> => {
      try {
        const res = await fetch(`${apiBaseUrl}/chats/${chatId}`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify({ name: newName }),
        });
        
        if (res.ok) {
          // Update both local states
          const updateSession = (session: ChatSessionMeta) => 
            session.chatId === chatId ? { ...session, name: newName } : session;
          
          setSessions(prev => prev.map(updateSession));
          setAllSessions(prev => prev.map(updateSession));
          return true;
        } else {
          console.error(`Failed to rename chat: ${res.status} ${res.statusText}`);
          return false;
        }
      } catch (err) {
        console.error("Failed to rename chat", err);
        return false;
      }
    },
    [apiBaseUrl, authToken]
  );

  const deleteChat = useCallback(
    async (chatId: string): Promise<boolean> => {
      try {
        const res = await fetch(`${apiBaseUrl}/chats/${chatId}`, {
          method: "DELETE",
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        
        if (res.ok) {
          // Update both local states
          setSessions(prev => prev.filter(session => session.chatId !== chatId));
          setAllSessions(prev => prev.filter(session => session.chatId !== chatId));
          return true;
        } else {
          console.error(`Failed to delete chat: ${res.status} ${res.statusText}`);
          return false;
        }
      } catch (err) {
        console.error("Failed to delete chat", err);
        return false;
      }
    },
    [apiBaseUrl, authToken]
  );

  const searchChats = useCallback(
    async (query: string): Promise<ChatSessionMeta[]> => {
      if (!query.trim()) return [];
      
      try {
        const res = await fetch(`${apiBaseUrl}/chats/search?q=${encodeURIComponent(query)}&limit=${limit}`, {
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        
        if (res.ok) {
          const data = await res.json();
          return data.map((c: any) => ({
            chatId: c.chat_id,
            createdAt: c.created_at,
            updatedAt: c.updated_at,
            name: c.name ?? null,
            lastMessage: c.last_message ?? null,
          }));
        } else {
          console.error(`Failed to search chats: ${res.status} ${res.statusText}`);
          return [];
        }
      } catch (err) {
        console.error("Failed to search chats", err);
        return [];
      }
    },
    [apiBaseUrl, authToken, limit]
  );

  // Debounced search effect
  useEffect(() => {
    if (searchQuery && searchQuery.trim()) {
      // Clear existing timeout
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
      
      // Set new timeout for debounced search
      searchTimeoutRef.current = setTimeout(async () => {
        try {
          const searchResults = await searchChats(searchQuery);
          // Merge search results with existing sessions, prioritizing search results
          const mergedResults = [...searchResults];
          // Add non-duplicate local sessions
          allSessions.forEach(session => {
            if (!mergedResults.find(result => result.chatId === session.chatId)) {
              mergedResults.push(session);
            }
          });
          setSessions(mergedResults);
        } catch (error) {
          console.error("Search failed:", error);
          // Fallback to all sessions if search fails
          setSessions(allSessions);
        }
      }, 300);
    } else if (searchQuery === "") {
      // Clear timeout and restore all sessions when search is cleared
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
      setSessions(allSessions);
    }

    // Cleanup timeout on unmount or search change
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery, searchChats, allSessions]);

  return { sessions, isLoading, reload: fetchSessions, renameChat, deleteChat, searchChats };
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
