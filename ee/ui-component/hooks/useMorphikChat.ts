import { useState, useCallback } from "react";
import type { UIMessage } from "@/components/chat/ChatMessages";
import { showAlert } from "@/components/ui/alert-system";
import { generateUUID } from "@/lib/utils";
import type { QueryOptions } from "@/components/types";

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
  updateQueryOption?: (key: keyof QueryOptions, value: string | number | boolean | undefined) => void;
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
}

export function useMorphikChat({
  chatId,
  apiBaseUrl,
  authToken,
  initialMessages = [],
  initialQueryOptions = {},
  onChatSubmit,
}: UseMorphikChatProps): UseMorphikChatReturn {
  const [messages, setMessages] = useState<UIMessage[]>(initialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: initialQueryOptions.filters ?? "{}",
    k: initialQueryOptions.k ?? 10,
    min_score: initialQueryOptions.min_score ?? 0.7,
    use_reranking: initialQueryOptions.use_reranking ?? false,
    use_colpali: initialQueryOptions.use_colpali ?? true,
    max_tokens: initialQueryOptions.max_tokens ?? 1024,
    temperature: initialQueryOptions.temperature ?? 0.7,
    graph_name: initialQueryOptions.graph_name,
    folder_name: initialQueryOptions.folder_name,
  });

  const status = isLoading ? "loading" : "idle";

  const updateQueryOption = useCallback((key: keyof QueryOptions, value: string | number | boolean | undefined) => {
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

      const messagesBeforeUpdate = [...messages];
      setMessages(prev => [...prev, newUserMessage]);
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
        } as Record<string, unknown>;

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
    [apiBaseUrl, authToken, chatId, messages, queryOptions, onChatSubmit]
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
