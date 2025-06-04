import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { useState, useCallback, useEffect } from "react";
import type { QueryOptions } from "@/components/types";
import { showAlert } from "@/components/ui/alert-system";
import { generateUUID } from "@/lib/utils";

interface MorphikRuntimeProps {
  chatId: string;
  apiBaseUrl: string;
  authToken: string | null;
  initialQueryOptions?: Partial<QueryOptions>;
  streamResponse?: boolean;
  onChatSubmit?: (query: string, options: QueryOptions) => void;
}

export function useMorphikRuntime({
  chatId,
  apiBaseUrl,
  authToken,
  initialQueryOptions = {},
  streamResponse = false,
  onChatSubmit,
}: MorphikRuntimeProps) {
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: initialQueryOptions.filters ?? "{}",
    k: initialQueryOptions.k ?? 5,
    min_score: initialQueryOptions.min_score ?? 0.7,
    use_reranking: initialQueryOptions.use_reranking ?? false,
    use_colpali: initialQueryOptions.use_colpali ?? true,
    max_tokens: initialQueryOptions.max_tokens ?? 1024,
    temperature: initialQueryOptions.temperature ?? 0.3,
    graph_name: initialQueryOptions.graph_name,
    folder_name: initialQueryOptions.folder_name,
  });

  const updateQueryOption = useCallback((key: keyof QueryOptions, value: QueryOptions[keyof QueryOptions]) => {
    setQueryOptions(prev => ({ ...prev, [key]: value }));
  }, []);

  // Create a custom API endpoint for the chat runtime
  const runtime = useChatRuntime({
    api: `/api/morphik-chat?chatId=${chatId}&apiBaseUrl=${encodeURIComponent(apiBaseUrl)}&authToken=${encodeURIComponent(authToken || "")}`,
  });

  // Load existing chat history on mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/chat/${chatId}`, {
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        if (response.ok) {
          const data = await response.json();
          if (data && data.length > 0) {
            // Convert Morphik messages to assistant-ui format
            const messages = data.map((m: any) => ({
              id: generateUUID(),
              role: m.role,
              content: [{ type: "text", text: m.content }],
              createdAt: new Date(m.timestamp),
            }));

            // Initialize runtime with history
            console.log("Loading chat history:", messages);
          }
        }
      } catch (err) {
        console.error("Failed to load chat history", err);
      }
    };

    if (chatId && apiBaseUrl) {
      fetchHistory();
    }
  }, [chatId, apiBaseUrl, authToken]);

  return {
    runtime,
    queryOptions,
    setQueryOptions,
    updateQueryOption,
  };
}
