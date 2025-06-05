"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChatMessage } from "@/components/types";
import { generateUUID } from "@/lib/utils";

import { Spin, ArrowUp } from "./icons";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AgentPreviewMessage, AgentUIMessage, ToolCall, DisplayObject, SourceObject } from "./AgentChatMessages";
import { Textarea } from "@/components/ui/textarea";

interface AgentChatSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialMessages?: ChatMessage[];
  isReadonly?: boolean;
  onAgentSubmit?: (query: string) => void;
  chatId?: string;
}

/**
 * AgentChatSection component for interacting with the agent API
 */
const AgentChatSection: React.FC<AgentChatSectionProps> = ({
  apiBaseUrl,
  authToken,
  initialMessages = [],
  isReadonly = false,
  onAgentSubmit,
  chatId,
}) => {
  // State for managing chat
  const [messages, setMessages] = useState<AgentUIMessage[]>(
    initialMessages.map(msg => ({
      id: generateUUID(),
      role: msg.role as "user" | "assistant",
      content: msg.content || "",
      createdAt: new Date(),
    }))
  );
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<"idle" | "submitted" | "completed">("idle");

  // Textarea and scroll refs
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load agent messages from chat history when component mounts or chatId changes
  useEffect(() => {
    const loadAgentHistory = async () => {
      if (chatId && apiBaseUrl && (authToken || apiBaseUrl.includes("localhost"))) {
        try {
          const response = await fetch(`${apiBaseUrl}/chat/${chatId}`, {
            headers: {
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
          });
          if (response.ok) {
            const data = await response.json();
            const agentMessagesFromHistory = data.map((m: any) => {
              const baseMessage = {
                id: generateUUID(),
                role: m.role,
                content: m.content,
                createdAt: new Date(m.timestamp),
              };

              // If this is an assistant message with agent_data, reconstruct experimental_agentData
              if (m.role === "assistant" && m.agent_data) {
                return {
                  ...baseMessage,
                  experimental_agentData: {
                    tool_history: m.agent_data.tool_history || [],
                    displayObjects: m.agent_data.display_objects || [],
                    sources: m.agent_data.sources || [],
                  },
                };
              }

              return baseMessage;
            });
            setMessages(agentMessagesFromHistory);
          }
        } catch (err) {
          console.error("Failed to load agent chat history", err);
        }
      }
    };

    // Only load if we don't have initial messages
    if (initialMessages.length === 0) {
      loadAgentHistory();
    }
  }, [chatId, apiBaseUrl, authToken, initialMessages.length]);

  // Function to handle form submission
  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!input.trim() || status === "submitted" || isReadonly) return;

    const userQuery = input;

    // Create user message
    const userMessage: AgentUIMessage = {
      id: generateUUID(),
      role: "user",
      content: userQuery,
      createdAt: new Date(),
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);

    // Create loading message
    const loadingMessage: AgentUIMessage = {
      id: generateUUID(),
      role: "assistant",
      content: "",
      createdAt: new Date(),
      isLoading: true,
    };

    // Add loading message
    setMessages(prev => [...prev, loadingMessage]);

    // Update status and clear input
    setStatus("submitted");
    setInput("");

    onAgentSubmit?.(userQuery);

    try {
      // Call agent API
      const response = await fetch(`${apiBaseUrl}/agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({
          query: userMessage.content,
          chat_id: chatId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Agent API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Create agent response message with tool history, display objects and sources
      const agentMessage: AgentUIMessage = {
        id: generateUUID(),
        role: "assistant",
        content: data.response,
        createdAt: new Date(),
        experimental_agentData: {
          tool_history: data.tool_history as ToolCall[],
          displayObjects: data.display_objects as DisplayObject[],
          sources: data.sources as SourceObject[],
        },
      };

      // Replace loading message with actual response
      setMessages(prev => prev.map(msg => (msg.isLoading ? agentMessage : msg)));
    } catch (error) {
      console.error("Error submitting to agent API:", error);

      // Create error message
      const errorMessage: AgentUIMessage = {
        id: generateUUID(),
        role: "assistant",
        content: `Error: ${error instanceof Error ? error.message : "Failed to get response from the agent"}`,
        createdAt: new Date(),
      };

      // Replace loading message with error message
      setMessages(prev => prev.map(msg => (msg.isLoading ? errorMessage : msg)));
    } finally {
      setStatus("completed");
    }
  };

  // Textarea height adjustment functions
  const adjustHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight + 2}px`;
    }
  };

  const resetHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value);
    adjustHeight();
  };

  const submitForm = () => {
    handleSubmit();
    resetHeight();
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  // Adjust textarea height on load
  useEffect(() => {
    if (textareaRef.current) {
      adjustHeight();
    }
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className="relative flex h-full w-full flex-col bg-background">
      {/* Messages Area */}
      <div className="relative min-h-0 flex-1">
        <ScrollArea className="h-full" ref={messagesContainerRef}>
          {messages.length === 0 && (
            <div className="flex flex-1 items-center justify-center p-8 text-center">
              <div className="max-w-md space-y-2">
                <h2 className="text-xl font-semibold">Morphik Agent Chat</h2>
                <p className="text-sm text-muted-foreground">Ask a question to the agent to get started.</p>
              </div>
            </div>
          )}

          <div className="flex flex-col pb-[80px] pt-4 md:pb-[120px]">
            {messages.map(message => (
              <AgentPreviewMessage key={message.id} message={message} />
            ))}

            {status === "submitted" && messages.length > 0 && messages[messages.length - 1].role === "user" && (
              <div className="flex h-12 items-center justify-center text-center text-xs text-muted-foreground">
                <Spin className="mr-2 animate-spin" />
                Agent thinking...
              </div>
            )}
          </div>

          <div ref={messagesEndRef} className="min-h-[24px] min-w-[24px] shrink-0" />
        </ScrollArea>
      </div>

      {/* Input Area */}
      <div className="sticky bottom-0 w-full bg-background">
        <div className="mx-auto max-w-4xl px-4 sm:px-6">
          <form
            className="pb-6"
            onSubmit={e => {
              e.preventDefault();
              handleSubmit(e);
            }}
          >
            <div className="relative w-full">
              <div className="pointer-events-none absolute -top-20 left-0 right-0 h-24 bg-gradient-to-t from-background to-transparent" />
              <div className="relative flex items-end">
                <Textarea
                  ref={textareaRef}
                  placeholder="Ask the agent..."
                  value={input}
                  onChange={handleInput}
                  className="max-h-[400px] min-h-[48px] w-full resize-none overflow-hidden pr-16 text-base"
                  rows={1}
                  autoFocus
                  disabled={status === "submitted" || isReadonly}
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      submitForm();
                    }
                  }}
                />
                <Button
                  type="submit"
                  size="icon"
                  className="absolute bottom-2 right-2 h-8 w-8"
                  disabled={!input.trim() || status === "submitted" || isReadonly}
                >
                  <ArrowUp className="h-4 w-4" />
                  <span className="sr-only">Send</span>
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AgentChatSection;
