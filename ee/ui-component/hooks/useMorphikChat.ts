import { useState, useCallback } from 'react';
import type { ChatMessage as MorphikChatMessage, QueryOptions } from '@/components/types';
import { showAlert } from '@/components/ui/alert-system';
import { generateUUID } from '@/lib/utils';
import { UIMessage } from '@/components/chat/ChatMessages';

// Define a simple Attachment type for our purposes
interface Attachment {
  url: string;
  name: string;
  contentType: string;
}

// Map your ChatMessage/Source to UIMessage
function mapMorphikToUIMessage(msg: MorphikChatMessage): UIMessage {
  return {
    id: generateUUID(),
    role: msg.role,
    content: msg.content,
    createdAt: new Date(),
    ...(msg.sources && { experimental_customData: { sources: msg.sources } }),
  };
}

export function useMorphikChat(chatId: string, apiBaseUrl: string, authToken: string | null, initialMessages: MorphikChatMessage[] = []) {
  const [messages, setMessages] = useState<UIMessage[]>(initialMessages.map(mapMorphikToUIMessage));
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  // Query options state (moved from ChatSection)
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    use_colpali: true,
    max_tokens: 500,
    temperature: 0.7,
  });

  // Mapping to useChat's status
  const status = isLoading ? 'streaming' : 'ready';

  const updateQueryOption = <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => {
    setQueryOptions(prev => ({ ...prev, [key]: value }));
  };

  const append = useCallback(async (message: UIMessage | Omit<UIMessage, 'id'>) => {
    // Add user message immediately
    const newUserMessage: UIMessage = {
      id: generateUUID(),
      ...message as Omit<UIMessage, 'id'>,
      createdAt: new Date()
    };
    setMessages(prev => [...prev, newUserMessage]);
    setIsLoading(true);

    // Call your backend
    try {
      // Prepare options
      const options = {
        filters: JSON.parse(queryOptions.filters || '{}'),
        k: queryOptions.k,
        min_score: queryOptions.min_score,
        use_reranking: queryOptions.use_reranking,
        use_colpali: queryOptions.use_colpali,
        max_tokens: queryOptions.max_tokens,
        temperature: queryOptions.temperature,
        graph_name: queryOptions.graph_name,
        folder_name: queryOptions.folder_name,
      };

      console.log(`Sending to ${apiBaseUrl}/query:`, { query: newUserMessage.content, ...options });

      const response = await fetch(`${apiBaseUrl}/query`, {
        method: 'POST',
        headers: {
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {}),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: newUserMessage.content,
          ...options
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Query failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Query response:', data);

      // Add assistant response
      const assistantMessage: UIMessage = {
        id: generateUUID(),
        role: 'assistant',
        content: data.completion,
        experimental_customData: { sources: data.sources },
        createdAt: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);

      // If sources are available, retrieve the full source content
      if (data.sources && data.sources.length > 0) {
        try {
          // Fetch full source details
          console.log(`Fetching sources from ${apiBaseUrl}/batch/chunks`);
          const sourcesResponse = await fetch(`${apiBaseUrl}/batch/chunks`, {
            method: 'POST',
            headers: {
              ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {}),
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              sources: data.sources,
              folder_name: queryOptions.folder_name,
              use_colpali: true,
            })
          });

          if (sourcesResponse.ok) {
            const sourcesData = await sourcesResponse.json();
            console.log('Sources data:', sourcesData);

            // Update the assistantMessage with full source content
            setMessages(prev => {
              const updatedMessages = [...prev];
              const lastMessageIndex = updatedMessages.length - 1;

              if (lastMessageIndex >= 0 && updatedMessages[lastMessageIndex].role === 'assistant') {
                updatedMessages[lastMessageIndex] = {
                  ...updatedMessages[lastMessageIndex],
                  experimental_customData: {
                    sources: sourcesData
                  }
                };
              }

              return updatedMessages;
            });
          } else {
            console.error('Error fetching sources:', sourcesResponse.status, sourcesResponse.statusText);
          }
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
          console.error('Error fetching full source content:', errorMsg);
        }
      }

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      console.error('Chat query error:', errorMsg);
      showAlert(errorMsg, { type: 'error', title: 'Chat Error', duration: 5000 });
      // Add an error message
      setMessages(prev => [...prev, {
        id: generateUUID(),
        role: 'assistant',
        content: `Error: ${errorMsg}`,
        createdAt: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl, authToken, queryOptions]);

  const handleSubmit = useCallback((e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    if (!input.trim() && attachments.length === 0) return;

    append({
      role: 'user',
      content: input,
      createdAt: new Date(),
    });

    // Clear input and attachments
    setInput('');
    setAttachments([]);
  }, [input, attachments, append]);

  // Add other functions as needed
  const reload = useCallback(() => {
    // Implement logic to reload last message if needed
    if (messages.length >= 2) {
      const lastUserMessageIndex = [...messages].reverse().findIndex(m => m.role === 'user');
      if (lastUserMessageIndex !== -1) {
        const actualIndex = messages.length - 1 - lastUserMessageIndex;
        const lastUserMessage = messages[actualIndex];

        // Remove last assistant message and submit the last user message again
        setMessages(prev => prev.slice(0, prev.length - 1));
        append(lastUserMessage);
      }
    }
  }, [messages, append]);

  const stop = useCallback(() => {
    setIsLoading(false);
    // Any additional logic to cancel ongoing requests would go here
  }, []);

  return {
    messages,
    setMessages,
    input,
    setInput,
    isLoading,
    status, // Map to useChat's status for compatibility
    handleSubmit,
    append,
    reload,
    stop,
    attachments,
    setAttachments,
    // Expose query options state and updater
    queryOptions,
    updateQueryOption,
  };
}
