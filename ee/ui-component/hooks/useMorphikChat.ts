import { useState, useCallback, useEffect, useRef } from "react";
import type { UIMessage } from "@/components/chat/ChatMessages";
import { showAlert } from "@/components/ui/alert-system";
import { generateUUID } from "@/lib/utils";
import type { QueryOptions } from "@/components/types";

// Cache for chat histories to avoid re-fetching
const chatHistoryCache = new Map<string, UIMessage[]>();

// Function to clear cache for a specific chat or all chats
export const clearChatCache = (chatId?: string, apiBaseUrl?: string) => {
  if (chatId && apiBaseUrl) {
    chatHistoryCache.delete(`${apiBaseUrl}-${chatId}`);
  } else {
    chatHistoryCache.clear();
  }
};

// Define a simple Attachment type for our purposes
interface Attachment {
  url: string;
  name: string;
  contentType: string;
}

// Interface for the hook's return value
interface UseMorphikChatReturn {
  messages: UIMessage[];
  append: (message: Omit<UIMessage, "id" | "role" | "createdAt">) => Promise<void>;
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  isLoading: boolean;
  queryOptions: QueryOptions;
  setQueryOptions: React.Dispatch<React.SetStateAction<QueryOptions>>;
  chatId: string;
  reload: () => void;
  stop: () => void;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  handleSubmit: (e?: React.FormEvent<HTMLFormElement>) => void;
  attachments?: Attachment[];
  setAttachments?: React.Dispatch<React.SetStateAction<Attachment[]>>;
  updateQueryOption?: (key: keyof QueryOptions, value: QueryOptions[keyof QueryOptions]) => void;
  status?: string;
}

// Props for the hook
interface UseMorphikChatProps {
  chatId: string;
  apiBaseUrl: string;
  authToken: string | null;
  initialMessages?: UIMessage[];
  initialQueryOptions?: Partial<QueryOptions>;
  onChatSubmit?: (query: string, options: QueryOptions, currentMessages: UIMessage[]) => void;
  streamResponse?: boolean;
}

export function useMorphikChat({
  chatId,
  apiBaseUrl,
  authToken,
  initialMessages = [],
  initialQueryOptions = {},
  onChatSubmit,
  streamResponse = false,
}: UseMorphikChatProps): UseMorphikChatReturn {
  const [messages, setMessagesInternal] = useState<UIMessage[]>(initialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  // Helper to update cache when messages change
  const updateCache = useCallback(
    (newMessages: UIMessage[]) => {
      const cacheKey = `${apiBaseUrl}-${chatId}`;
      chatHistoryCache.set(cacheKey, newMessages);
    },
    [apiBaseUrl, chatId]
  );

  // Custom setMessages that also updates cache
  const setMessages = useCallback(
    (newMessages: UIMessage[] | ((prev: UIMessage[]) => UIMessage[])) => {
      setMessagesInternal(prev => {
        const updated = typeof newMessages === "function" ? newMessages(prev) : newMessages;
        updateCache(updated);
        return updated;
      });
    },
    [updateCache]
  );

  // Load existing chat history from server on mount with caching
  useEffect(() => {
    const fetchHistory = async () => {
      // Check cache first
      const cacheKey = `${apiBaseUrl}-${chatId}`;
      const cached = chatHistoryCache.get(cacheKey);
      if (cached) {
        setMessages(cached);
        return;
      }

      try {
        const response = await fetch(`${apiBaseUrl}/chat/${chatId}`, {
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        if (response.ok) {
          const data = await response.json();
          const messages = data.map((m: any) => ({
            id: generateUUID(),
            role: m.role,
            content: m.content,
            createdAt: new Date(m.timestamp),
          }));

          // Cache the messages
          chatHistoryCache.set(cacheKey, messages);
          setMessages(messages);
        }
      } catch (err) {
        console.error("Failed to load chat history", err);
      }
    };
    fetchHistory();
  }, [chatId, apiBaseUrl, authToken]);

  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: initialQueryOptions.filters ?? "{}",
    k: initialQueryOptions.k ?? 5,
    min_score: initialQueryOptions.min_score ?? 0.7,
    use_reranking: initialQueryOptions.use_reranking ?? false,
    use_colpali: initialQueryOptions.use_colpali ?? true,
    padding: initialQueryOptions.padding ?? 0,
    max_tokens: initialQueryOptions.max_tokens ?? 1024,
    temperature: initialQueryOptions.temperature ?? 0.3,
    graph_name: initialQueryOptions.graph_name,
    folder_name: initialQueryOptions.folder_name,
  });

  const status = isLoading ? "loading" : "idle";

  const updateQueryOption = useCallback((key: keyof QueryOptions, value: QueryOptions[keyof QueryOptions]) => {
    setQueryOptions(prev => ({ ...prev, [key]: value }));
  }, []);

  const append = useCallback(
    async (message: Omit<UIMessage, "id" | "role" | "createdAt">) => {
      const newUserMessage: UIMessage = {
        id: generateUUID(),
        role: "user",
        ...message,
        createdAt: new Date(),
      };

      const currentQueryOptions: QueryOptions = {
        ...queryOptions,
        filters: queryOptions.filters || "{}",
      };

      // Capture previous messages inside functional update to avoid stale refs
      let messagesBeforeUpdate: UIMessage[] = [];
      setMessages(prev => {
        messagesBeforeUpdate = [...prev];
        return [...prev, newUserMessage];
      });
      setIsLoading(true);

      onChatSubmit?.(newUserMessage.content, currentQueryOptions, messagesBeforeUpdate);

      try {
        console.log(`Sending to ${apiBaseUrl}/query:`, {
          query: newUserMessage.content,
          ...currentQueryOptions,
        });

        // Ensure filters is an object before sending to the API
        let parsedFilters: Record<string, unknown> | undefined;
        if (currentQueryOptions.filters) {
          try {
            parsedFilters =
              typeof currentQueryOptions.filters === "string"
                ? JSON.parse(currentQueryOptions.filters as string)
                : (currentQueryOptions.filters as Record<string, unknown>);
          } catch {
            console.warn("Invalid filters JSON, defaulting to empty object");
            parsedFilters = {};
          }
        }

        const payload = {
          query: newUserMessage.content,
          ...currentQueryOptions,
          filters: parsedFilters ?? {},
          chat_id: chatId,
          stream_response: streamResponse,
          llm_config: currentQueryOptions.llm_config,
        } as Record<string, unknown>;

        if (streamResponse) {
          // Handle streaming response
          const response = await fetch(`${apiBaseUrl}/query`, {
            method: "POST",
            headers: {
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            setMessages(messagesBeforeUpdate);
            throw new Error(errorData.detail || `Query failed: ${response.status} ${response.statusText}`);
          }

          // Process streaming response
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();
          let fullContent = "";
          // Accumulation buffer to reduce React renders
          const FLUSH_MS = 50;
          let buffer = "";
          let lastFlush = Date.now();
          // Helper to (create or) update the assistant message
          const flushMessage = () => {
            if (!assistantMessageId) {
              assistantMessageId = generateUUID();
              const assistantMessage: UIMessage = {
                id: assistantMessageId,
                role: "assistant",
                content: fullContent,
                createdAt: new Date(),
              };
              setMessages(prev => [...prev, assistantMessage]);
            } else {
              const id = assistantMessageId;
              setMessages(prev => prev.map(m => (m.id === id ? { ...m, content: fullContent } : m)));
            }
          };
          let assistantMessageId: string | null = null;

          if (reader) {
            let streamFinished = false;
            try {
              while (true) {
                const { done, value } = await reader.read();
                if (done || streamFinished) break; // stream closed or marked finished

                const chunk = decoder.decode(value);
                const lines = chunk.split("\n");

                for (const line of lines) {
                  if (line.startsWith("data: ")) {
                    try {
                      const data = JSON.parse(line.slice(6));

                      // Handle different event types
                      switch (data.type) {
                        case "assistant":
                          if (data.content) {
                            fullContent += data.content;
                            buffer += data.content;

                            // Always flush on the very first token so the message appears immediately
                            if (!assistantMessageId) {
                              flushMessage();
                              buffer = "";
                              lastFlush = Date.now();
                            } else if (Date.now() - lastFlush >= FLUSH_MS) {
                              flushMessage();
                              buffer = "";
                              lastFlush = Date.now();
                            }
                          }
                          break;

                        case "tool":
                          // Tool execution result - for now just log it
                          // In the future, this could be used to show tool execution status
                          console.log(`Tool executed: ${data.name} -> ${data.content}`);
                          break;

                        case "done":
                          // Streaming is complete, handle sources if provided
                          const sourcesShallow = data.sources ?? [];
                          // ensure final content flushed
                          flushMessage();

                          // Enrich sources in background and attach
                          if (sourcesShallow.length > 0 && assistantMessageId) {
                            try {
                              const enriched = await fetch(`${apiBaseUrl}/batch/chunks`, {
                                method: "POST",
                                headers: {
                                  ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                                  "Content-Type": "application/json",
                                },
                                body: JSON.stringify({
                                  sources: sourcesShallow,
                                  folder_name: queryOptions.folder_name,
                                  use_colpali: true,
                                }),
                              }).then(r => (r.ok ? r.json() : sourcesShallow));

                              setMessages(prev =>
                                prev.map(m =>
                                  m.id === assistantMessageId
                                    ? { ...m, experimental_customData: { sources: enriched } }
                                    : m
                                )
                              );
                            } catch (err) {
                              console.error("Failed to enrich sources: ", err);
                            }
                          }

                          // We received done – stop reading further
                          await reader.cancel();
                          streamFinished = true;
                          break;

                        case "error":
                          // Error occurred
                          throw new Error(data.content || "Unknown error occurred");

                        default:
                          // Legacy format support - handle old format for backward compatibility
                          if (data.content && !data.type) {
                            fullContent += data.content;
                            buffer += data.content;

                            // Always flush on the very first token so the message appears immediately
                            if (!assistantMessageId) {
                              flushMessage();
                              buffer = "";
                              lastFlush = Date.now();
                            } else if (Date.now() - lastFlush >= FLUSH_MS) {
                              flushMessage();
                              buffer = "";
                              lastFlush = Date.now();
                            }
                          } else if (data.done) {
                            // Legacy done format
                            const sourcesShallow = data.sources ?? [];
                            // ensure final content flushed
                            flushMessage();

                            // Enrich sources in background and attach
                            if (sourcesShallow.length > 0 && assistantMessageId) {
                              try {
                                const enriched = await fetch(`${apiBaseUrl}/batch/chunks`, {
                                  method: "POST",
                                  headers: {
                                    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                                    "Content-Type": "application/json",
                                  },
                                  body: JSON.stringify({
                                    sources: sourcesShallow,
                                    folder_name: queryOptions.folder_name,
                                    use_colpali: true,
                                  }),
                                }).then(r => (r.ok ? r.json() : sourcesShallow));

                                setMessages(prev =>
                                  prev.map(m =>
                                    m.id === assistantMessageId
                                      ? { ...m, experimental_customData: { sources: enriched } }
                                      : m
                                  )
                                );
                              } catch (err) {
                                console.error("Failed to enrich sources: ", err);
                              }
                            }

                            // We received done – stop reading further
                            await reader.cancel();
                            streamFinished = true;
                          }
                          break;
                      }
                    } catch (e) {
                      console.warn("Failed to parse streaming data:", line);
                    }
                  }
                }
                if (streamFinished) break;
              }
            } finally {
              // Ensure any remaining buffered content is flushed once the stream ends
              if (fullContent.length > 0) {
                flushMessage();
              }
              reader.releaseLock();
            }
          }
        } else {
          // Handle regular non-streaming response
          const response = await fetch(`${apiBaseUrl}/query`, {
            method: "POST",
            headers: {
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            setMessages(messagesBeforeUpdate);
            throw new Error(errorData.detail || `Query failed: ${response.status} ${response.statusText}`);
          }

          const data = await response.json();
          console.log("Query response:", data);

          const assistantMessage: UIMessage = {
            id: generateUUID(),
            role: "assistant",
            content: data.completion,
            experimental_customData: { sources: data.sources },
            createdAt: new Date(),
          };
          setMessages(prev => [...prev, assistantMessage]);

          // Handle sources for non-streaming responses
          if (data.sources && data.sources.length > 0) {
            try {
              console.log(`Fetching sources from ${apiBaseUrl}/batch/chunks`);
              const sourcesResponse = await fetch(`${apiBaseUrl}/batch/chunks`, {
                method: "POST",
                headers: {
                  ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  sources: data.sources,
                  folder_name: queryOptions.folder_name,
                  use_colpali: true,
                }),
              });

              if (sourcesResponse.ok) {
                const sourcesData = await sourcesResponse.json();
                console.log("Sources data:", sourcesData);

                setMessages(prev => {
                  const updatedMessages = [...prev];
                  const lastMessageIndex = updatedMessages.length - 1;

                  if (lastMessageIndex >= 0 && updatedMessages[lastMessageIndex].role === "assistant") {
                    updatedMessages[lastMessageIndex] = {
                      ...updatedMessages[lastMessageIndex],
                      experimental_customData: {
                        sources: sourcesData,
                      },
                    };
                  }

                  return updatedMessages;
                });
              } else {
                console.error("Error fetching sources:", sourcesResponse.status, sourcesResponse.statusText);
              }
            } catch (err) {
              const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
              console.error("Error fetching full source content:", errorMsg);
            }
          }
        }
      } catch (error) {
        console.error("Chat API error:", error);
        showAlert(error instanceof Error ? error.message : "Failed to get chat response", {
          type: "error",
          title: "Chat Error",
          duration: 5000,
        });
        setIsLoading(false);
      } finally {
        if (!isLoading) {
          /* Only set if it wasn't already set by error block */
        }
        setIsLoading(false);
      }
    },
    [apiBaseUrl, authToken, chatId, queryOptions, onChatSubmit, streamResponse]
  );

  const handleSubmit = useCallback(
    (e?: React.FormEvent<HTMLFormElement>) => {
      e?.preventDefault();
      if (!input.trim() && attachments.length === 0) return;
      append({ content: input });
      setInput("");
      setAttachments([]);
    },
    [input, attachments, append]
  );

  const reload = useCallback(() => {
    console.warn("reload function not implemented");
  }, []);

  const stop = useCallback(() => {
    console.warn("stop function not implemented");
    setIsLoading(false);
  }, []);

  return {
    messages,
    append,
    setMessages,
    isLoading,
    queryOptions,
    setQueryOptions,
    chatId,
    reload,
    stop,
    input,
    setInput,
    handleSubmit,
    attachments,
    setAttachments,
    updateQueryOption,
    status,
  };
}
