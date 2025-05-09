"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChatMessage } from "@/components/types";
import { generateUUID } from "@/lib/utils";
import { AgentResponseData } from "@/types/agent-response";

import { Spin, ArrowUp } from "./icons";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AgentPreviewMessage, AgentUIMessage, ToolCall } from "./AgentChatMessages";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface AgentChatSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialMessages?: ChatMessage[];
  isReadonly?: boolean;
  onAgentSubmit?: (query: string) => void;
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
  const [requestRichResponse, setRequestRichResponse] = useState(false);
  const [requestGrounding, setRequestGrounding] = useState(false);

  // Textarea and scroll refs
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Function to handle form submission
  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!input.trim() || status === "submitted" || isReadonly) return;

    const userQuery = input;

    const userMessage: AgentUIMessage = {
      id: generateUUID(),
      role: "user",
      content: userQuery,
      createdAt: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    const loadingMessageId = generateUUID();
    const loadingMessage: AgentUIMessage = {
      id: loadingMessageId,
      role: "assistant",
      content: "", // For loading, content is empty or a placeholder if not using richResponse
      createdAt: new Date(),
      isLoading: true,
    };

    setMessages(prev => [...prev, loadingMessage]);
    setStatus("submitted");
    setInput("");

    onAgentSubmit?.(userQuery);

    try {
      const requestBody: { query: string; rich?: boolean; ground?: boolean } = {
        query: userQuery,
      };
      if (requestRichResponse) {
        requestBody.rich = true;
        // Only send ground if rich is true
        if (requestGrounding) {
          requestBody.ground = true;
        }
      }

      const response = await fetch(`${apiBaseUrl}/agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`Agent API error: ${response.status} - ${errorData.detail || "Unknown error"}`);
      }

      const data = await response.json();

      let agentResponseMessage: AgentUIMessage;

      if (requestRichResponse && data.response && typeof data.response === 'object' && ('mode' in data.response)) {
        // This is a rich response (or a plain response wrapped in the new structure)
        const richData = data.response as AgentResponseData; // Type assertion
        agentResponseMessage = {
          id: loadingMessageId, // Reuse ID to replace loading message
          role: "assistant",
          content: richData.mode === 'plain' ? richData.body : "", // Main text content if plain, or empty if rich (body is in richResponse)
          richResponse: richData, // Store the full rich/plain response object
          createdAt: new Date(),
          experimental_agentData: {
            tool_history: data.tool_history as ToolCall[],
          },
          isLoading: false,
        };
      } else {
        // This is an old-style plain text response or a non-rich response without the mode structure
        agentResponseMessage = {
          id: loadingMessageId, // Reuse ID
          role: "assistant",
          content: typeof data.response === 'string' ? data.response : JSON.stringify(data.response),
          createdAt: new Date(),
          experimental_agentData: {
            tool_history: data.tool_history as ToolCall[],
          },
          isLoading: false,
        };
      }

      setMessages(prev => prev.map(msg => (msg.id === loadingMessageId ? agentResponseMessage : msg)));
    } catch (error) {
      console.error("Error submitting to agent API:", error);
      const errorMessageContent = error instanceof Error ? error.message : "Failed to get response from the agent";
      const errorResponseMessage: AgentUIMessage = {
        id: loadingMessageId, // Reuse ID
        role: "assistant",
        content: `Error: ${errorMessageContent}`,
        createdAt: new Date(),
        isLoading: false,
      };
      setMessages(prev => prev.map(msg => (msg.id === loadingMessageId ? errorResponseMessage : msg)));
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
    // If rich response is turned off, also turn off grounding
    if (!requestRichResponse && requestGrounding) {
      setRequestGrounding(false);
    }
  }, [messages, requestRichResponse, requestGrounding]);

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
          {/* Agent Response Options */}
          {!isReadonly && (
            <div className="mb-2 flex items-center justify-end space-x-4 pr-2 pt-2">
              <div className="flex items-center space-x-2">
                <Switch
                  id="rich-format-switch"
                  checked={requestRichResponse}
                  onCheckedChange={setRequestRichResponse}
                  disabled={status === "submitted"}
                />
                <Label htmlFor="rich-format-switch" className="text-xs">
                  Rich Format
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="ground-answers-switch"
                  checked={requestGrounding}
                  onCheckedChange={setRequestGrounding}
                  disabled={status === "submitted" || !requestRichResponse} // Disable if not rich or if submitted
                />
                <Label htmlFor="ground-answers-switch" className="text-xs">
                  Ground Answers
                </Label>
              </div>
            </div>
          )}

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
