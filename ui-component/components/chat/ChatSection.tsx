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

import { ChatMessage, QueryOptions } from '@/components/types';

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
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    use_colpali: true,
    max_tokens: 500,
    temperature: 0.7
  });

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

  // Fetch graphs when auth token or API URL changes
  useEffect(() => {
    if (authToken) {
      console.log('ChatSection: Fetching graphs with new auth token');
      // Clear current messages when auth changes
      setChatMessages([]);
      fetchGraphs();
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
      
      // Prepare options with graph_name if it exists
      const options = {
        filters: JSON.parse(queryOptions.filters || '{}'),
        k: queryOptions.k,
        min_score: queryOptions.min_score,
        use_reranking: queryOptions.use_reranking,
        use_colpali: queryOptions.use_colpali,
        max_tokens: queryOptions.max_tokens,
        temperature: queryOptions.temperature,
        graph_name: queryOptions.graph_name
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
      const assistantMessage: ChatMessage = { role: 'assistant', content: data.completion };
      setChatMessages(prev => [...prev, assistantMessage]);
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
    <Card className="h-[calc(100vh-12rem)] flex flex-col">
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
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ChatSection;