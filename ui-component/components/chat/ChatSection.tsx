"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageSquare } from 'lucide-react';
import { showAlert } from '@/components/ui/alert-system';
import ChatOptionsDialog from './ChatOptionsDialog';
import ChatMessageComponent from './ChatMessage';

import { ChatMessage, QueryOptions, Folder, Source } from '@/components/types';

interface ChatSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
}

const ChatSection: React.FC<ChatSectionProps> = ({ apiBaseUrl, authToken }) => {
  const [chatQuery, setChatQuery] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [showChatAdvanced, setShowChatAdvanced] = useState(false);
  const [availableGraphs, setAvailableGraphs] = useState<string[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    use_colpali: true,
    max_tokens: 500,
    temperature: 0.7
  });

  // Handle URL parameters for folder and filters
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const folderParam = params.get('folder');
      const filtersParam = params.get('filters');
      const documentIdsParam = params.get('document_ids');

      let shouldShowChatOptions = false;

      // Update folder if provided
      if (folderParam) {
        try {
          const folderName = decodeURIComponent(folderParam);
          if (folderName) {
            console.log(`Setting folder from URL parameter: ${folderName}`);
            updateQueryOption('folder_name', folderName);
            shouldShowChatOptions = true;
          }
        } catch (error) {
          console.error('Error parsing folder parameter:', error);
        }
      }

      // Handle document_ids (selected documents) parameter - for backward compatibility
      if (documentIdsParam) {
        try {
          const documentIdsJson = decodeURIComponent(documentIdsParam);
          const documentIds = JSON.parse(documentIdsJson);

          // Create a filter object with external_id filter (correct field name)
          const filtersObj = { external_id: documentIds };
          const validFiltersJson = JSON.stringify(filtersObj);

          console.log(`Setting document_ids filter from URL parameter:`, filtersObj);
          updateQueryOption('filters', validFiltersJson);
          shouldShowChatOptions = true;
        } catch (error) {
          console.error('Error parsing document_ids parameter:', error);
        }
      }
      // Handle general filters parameter
      if (filtersParam) {
        try {
          const filtersJson = decodeURIComponent(filtersParam);
          // Parse the JSON to confirm it's valid
          const filtersObj = JSON.parse(filtersJson);

          console.log(`Setting filters from URL parameter:`, filtersObj);

          // Store the filters directly as a JSON string
          updateQueryOption('filters', filtersJson);
          shouldShowChatOptions = true;

          // Log a more helpful message about what's happening
          if (filtersObj.external_id) {
            console.log(`Chat will filter by ${Array.isArray(filtersObj.external_id) ? filtersObj.external_id.length : 1} document(s)`);
          }
        } catch (error) {
          console.error('Error parsing filters parameter:', error);
        }
      }

      // Only show the chat options panel on initial parameter load
      if (shouldShowChatOptions) {
        setShowChatAdvanced(true);

        // Clear URL parameters after processing them to prevent modal from re-appearing on refresh
        if (window.history.replaceState) {
          const newUrl = window.location.pathname + window.location.hash;
          window.history.replaceState({}, document.title, newUrl);
        }
      }
    }
  }, []);

  // Update query options
  const updateQueryOption = <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => {
    setQueryOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Fetch available graphs for dropdown
  const fetchGraphs = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/graphs`, {
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : ''
        }
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.statusText}`);
      }
      const graphsData = await response.json();
      setAvailableGraphs(graphsData.map((graph: { name: string }) => graph.name));
    } catch (err) {
      console.error('Error fetching available graphs:', err);
    }
  }, [apiBaseUrl, authToken]);

  // Fetch graphs and folders when auth token or API URL changes
  useEffect(() => {
    if (authToken) {
      console.log('ChatSection: Fetching data with new auth token');
      // Clear current messages when auth changes
      setChatMessages([]);
      fetchGraphs();

      // Fetch available folders for dropdown
      const fetchFolders = async () => {
        try {
          const response = await fetch(`${apiBaseUrl}/folders`, {
            headers: {
              'Authorization': authToken ? `Bearer ${authToken}` : ''
            }
          });

          if (!response.ok) {
            throw new Error(`Failed to fetch folders: ${response.statusText}`);
          }

          const foldersData = await response.json();
          setFolders(foldersData);
        } catch (err) {
          console.error('Error fetching folders:', err);
        }
      };

      fetchFolders();
    }
  }, [authToken, apiBaseUrl, fetchGraphs]);

  // Handle chat
  const handleChat = async () => {
    if (!chatQuery.trim()) {
      showAlert('Please enter a message', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    try {
      setLoading(true);

      // Add user message to chat
      const userMessage: ChatMessage = { role: 'user', content: chatQuery };
      setChatMessages(prev => [...prev, userMessage]);

      // Prepare options with graph_name and folder_name if they exist
      const options = {
        filters: JSON.parse(queryOptions.filters || '{}'),
        k: queryOptions.k,
        min_score: queryOptions.min_score,
        use_reranking: queryOptions.use_reranking,
        use_colpali: queryOptions.use_colpali,
        max_tokens: queryOptions.max_tokens,
        temperature: queryOptions.temperature,
        graph_name: queryOptions.graph_name,
        folder_name: queryOptions.folder_name
      };

      const response = await fetch(`${apiBaseUrl}/query`, {
        method: 'POST',
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : '',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: chatQuery,
          ...options
        })
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const data = await response.json();

      // Add assistant response to chat
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.completion,
        sources: data.sources
      };
      setChatMessages(prev => [...prev, assistantMessage]);

      // If sources are available, retrieve the full source content
      if (data.sources && data.sources.length > 0) {
        try {
          // Fetch full source details
          const sourcesResponse = await fetch(`${apiBaseUrl}/batch/chunks`, {
            method: 'POST',
            headers: {
              'Authorization': authToken ? `Bearer ${authToken}` : '',
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

            // Check if we have any image sources
            const imageSources = sourcesData.filter((source: Source) =>
              source.content_type?.startsWith('image/') ||
              (source.content && (
                source.content.startsWith('data:image/png;base64,') ||
                source.content.startsWith('data:image/jpeg;base64,')
              ))
            );
            console.log('Image sources found:', imageSources.length);

            // Update the message with detailed source information
            const updatedMessage = {
              ...assistantMessage,
              sources: sourcesData.map((source: Source) => {
                return {
                  document_id: source.document_id,
                  chunk_number: source.chunk_number,
                  score: source.score,
                  content: source.content,
                  content_type: source.content_type || 'text/plain',
                  filename: source.filename,
                  metadata: source.metadata,
                  download_url: source.download_url
                };
              })
            };

            // Update the message with detailed sources
            setChatMessages(prev => prev.map((msg, idx) =>
              idx === prev.length - 1 ? updatedMessage : msg
            ));
          }
        } catch (err) {
          console.error('Error fetching source details:', err);
          // Continue with basic sources if detailed fetch fails
        }
      }
      setChatQuery(''); // Clear input
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Chat Query Failed',
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle>Chat with Your Documents</CardTitle>
        <CardDescription>
          Ask questions about your documents and get AI-powered answers.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-grow overflow-hidden flex flex-col">
        <ScrollArea className="flex-grow pr-4 mb-4">
          {chatMessages.length > 0 ? (
            <div className="space-y-4">
              {chatMessages.map((message, index) => (
                <ChatMessageComponent
                  key={index}
                  role={message.role}
                  content={message.content}
                  sources={message.sources}
                />
              ))}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <MessageSquare className="mx-auto h-12 w-12 mb-2" />
                <p>Start a conversation about your documents</p>
              </div>
            </div>
          )}
        </ScrollArea>

        <div className="pt-4 border-t">
          <div className="space-y-4">
            <div className="flex gap-2">
              <Textarea
                placeholder="Ask a question..."
                value={chatQuery}
                onChange={(e) => setChatQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleChat();
                  }
                }}
                className="min-h-10"
              />
              <Button onClick={handleChat} disabled={loading}>
                {loading ? 'Sending...' : 'Send'}
              </Button>
            </div>

            <div className="flex justify-between items-center mt-2">
              <p className="text-xs text-muted-foreground">
                Press Enter to send, Shift+Enter for a new line
              </p>

              <ChatOptionsDialog
                showChatAdvanced={showChatAdvanced}
                setShowChatAdvanced={setShowChatAdvanced}
                queryOptions={queryOptions}
                updateQueryOption={updateQueryOption}
                availableGraphs={availableGraphs}
                folders={folders}
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ChatSection;
