"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useMorphikChat } from "@/hooks/useMorphikChat";
import { Folder } from "@/components/types";
import { generateUUID } from "@/lib/utils";
import type { QueryOptions } from "@/components/types";
import type { UIMessage } from "./ChatMessages";

import { Settings, Spin, ArrowUp, Sparkles } from "./icons";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PreviewMessage } from "./ChatMessages";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { AgentPreviewMessage, AgentUIMessage, ToolCall, DisplayObject, SourceObject } from "./AgentChatMessages";

interface ChatSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialMessages?: UIMessage[];
  isReadonly?: boolean;
  onChatSubmit?: (query: string, options: QueryOptions, initialMessages?: UIMessage[]) => void;
}

/**
 * ChatSection component using Vercel-style UI
 */
const ChatSection: React.FC<ChatSectionProps> = ({
  apiBaseUrl,
  authToken,
  initialMessages = [],
  isReadonly = false,
  onChatSubmit,
}) => {
  // Selected chat ID – start with fresh conversation
  const [chatId, setChatId] = useState<string>(() => generateUUID());

  // Initialize our custom hook
  const { messages, input, setInput, status, handleSubmit, queryOptions, updateQueryOption } = useMorphikChat({
    chatId,
    apiBaseUrl,
    authToken,
    initialMessages,
    onChatSubmit,
  });

  // Helper to safely update options (updateQueryOption may be undefined in readonly mode)
  const safeUpdateOption = useCallback(
    <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => {
      if (updateQueryOption) {
        updateQueryOption(key, value);
      }
    },
    [updateQueryOption]
  );

  // Derive safe option values with sensible defaults to avoid undefined issues in UI
  const safeQueryOptions: Required<Pick<QueryOptions, "k" | "min_score" | "temperature" | "max_tokens">> &
    QueryOptions = {
    k: queryOptions.k ?? 5,
    min_score: queryOptions.min_score ?? 0.7,
    temperature: queryOptions.temperature ?? 0.7,
    max_tokens: queryOptions.max_tokens ?? 1024,
    ...queryOptions,
  };

  // Sidebar collapsed state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // State for settings visibility
  const [showSettings, setShowSettings] = useState(false);
  const [availableGraphs, setAvailableGraphs] = useState<string[]>([]);
  const [loadingGraphs, setLoadingGraphs] = useState(false);
  const [loadingFolders, setLoadingFolders] = useState(false);
  const [folders, setFolders] = useState<Folder[]>([]);

  // Agent mode toggle and state
  const [isAgentMode, setIsAgentMode] = useState(false);
  const [agentMessages, setAgentMessages] = useState<AgentUIMessage[]>([]);
  const [agentStatus, setAgentStatus] = useState<"idle" | "submitted" | "completed">("idle");

  // Fetch available graphs for dropdown
  const fetchGraphs = useCallback(async () => {
    if (!apiBaseUrl) return;

    setLoadingGraphs(true);
    try {
      console.log(`Fetching graphs from: ${apiBaseUrl}/graphs`);
      const response = await fetch(`${apiBaseUrl}/graphs`, {
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.status} ${response.statusText}`);
      }

      const graphsData = await response.json();
      console.log("Graphs data received:", graphsData);

      if (Array.isArray(graphsData)) {
        setAvailableGraphs(graphsData.map((graph: { name: string }) => graph.name));
      } else {
        console.error("Expected array for graphs data but received:", typeof graphsData);
      }
    } catch (err) {
      console.error("Error fetching available graphs:", err);
    } finally {
      setLoadingGraphs(false);
    }
  }, [apiBaseUrl, authToken]);

  // Fetch folders
  const fetchFolders = useCallback(async () => {
    if (!apiBaseUrl) return;

    setLoadingFolders(true);
    try {
      console.log(`Fetching folders from: ${apiBaseUrl}/folders`);
      const response = await fetch(`${apiBaseUrl}/folders`, {
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch folders: ${response.status} ${response.statusText}`);
      }

      const foldersData = await response.json();
      console.log("Folders data received:", foldersData);

      if (Array.isArray(foldersData)) {
        setFolders(foldersData);
      } else {
        console.error("Expected array for folders data but received:", typeof foldersData);
      }
    } catch (err) {
      console.error("Error fetching folders:", err);
    } finally {
      setLoadingFolders(false);
    }
  }, [apiBaseUrl, authToken]);

  // Fetch graphs and folders when component mounts
  useEffect(() => {
    // Define a function to handle data fetching
    const fetchData = async () => {
      if (authToken || apiBaseUrl.includes("localhost")) {
        console.log("ChatSection: Fetching data with auth token:", !!authToken);
        await fetchGraphs();
        await fetchFolders();
      }
    };

    fetchData();
  }, [authToken, apiBaseUrl, fetchGraphs, fetchFolders]);

  // Text area ref and adjustment functions
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  React.useEffect(() => {
    if (textareaRef.current) {
      adjustHeight();
    }
  }, []);

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

  // Submit handler for agent mode – mirrors AgentChatSection logic
  const handleAgentSubmit = async () => {
    if (!input.trim() || agentStatus === "submitted" || isReadonly) return;

    const userQuery = input.trim();

    const userMessage: AgentUIMessage = {
      id: generateUUID(),
      role: "user",
      content: userQuery,
      createdAt: new Date(),
    };

    setAgentMessages(prev => [...prev, userMessage]);

    const loadingMessage: AgentUIMessage = {
      id: generateUUID(),
      role: "assistant",
      content: "",
      createdAt: new Date(),
      isLoading: true,
    };

    setAgentMessages(prev => [...prev, loadingMessage]);
    setAgentStatus("submitted");
    setInput("");

    try {
      const response = await fetch(`${apiBaseUrl}/agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({ query: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error(`Agent API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

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

      setAgentMessages(prev => prev.map(m => (m.isLoading ? agentMessage : m)));
    } catch (error) {
      console.error("Error submitting to agent API:", error);

      const errorMessage: AgentUIMessage = {
        id: generateUUID(),
        role: "assistant",
        content: `Error: ${error instanceof Error ? error.message : "Failed to get response from the agent"}`,
        createdAt: new Date(),
      };

      setAgentMessages(prev => prev.map(m => (m.isLoading ? errorMessage : m)));
    } finally {
      setAgentStatus("completed");
    }
  };

  const submitForm = () => {
    if (isAgentMode) {
      handleAgentSubmit();
    } else {
      handleSubmit();
    }
    resetHeight();
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  // Messages container ref for scrolling
  const messagesContainerRef = React.useRef<HTMLDivElement>(null);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  React.useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, agentMessages]);

  return (
    <div className="relative flex h-full w-full overflow-hidden bg-background">
      {/* Sidebar */}
      <ChatSidebar
        apiBaseUrl={apiBaseUrl}
        authToken={authToken}
        activeChatId={chatId}
        onSelect={id => setChatId(id ?? generateUUID())}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(prev => !prev)}
      />

      {/* Main chat area */}
      <div className="flex h-full flex-1 flex-col">
        {/* Messages Area */}
        <div className="relative min-h-0 flex-1">
          <ScrollArea className="h-full" ref={messagesContainerRef}>
            {(isAgentMode ? agentMessages.length === 0 : messages.length === 0) && (
              <div className="flex flex-1 items-center justify-center p-8 text-center">
                <div className="max-w-md space-y-2">
                  <h2 className="text-xl font-semibold">
                    {isAgentMode ? "Morphik Agent Chat" : "Welcome to Morphik Chat"}
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {isAgentMode
                      ? "Ask a question to the agent to get started."
                      : "Ask a question about your documents to get started."}
                  </p>
                </div>
              </div>
            )}

            <div className="flex flex-col pb-[80px] pt-4 md:pb-[120px]">
              {(isAgentMode ? agentMessages : messages).map(msg =>
                isAgentMode ? (
                  <AgentPreviewMessage key={msg.id} message={msg as AgentUIMessage} />
                ) : (
                  <PreviewMessage key={msg.id} message={msg} />
                )
              )}

              {isAgentMode
                ? agentStatus === "submitted" &&
                  agentMessages.length > 0 &&
                  agentMessages[agentMessages.length - 1].role === "user" && (
                    <div className="flex h-12 items-center justify-center text-center text-xs text-muted-foreground">
                      <Spin className="mr-2 animate-spin" />
                      Agent thinking...
                    </div>
                  )
                : status === "loading" &&
                  messages.length > 0 &&
                  messages[messages.length - 1].role === "user" && (
                    <div className="flex h-12 items-center justify-center text-center text-xs text-muted-foreground">
                      <Spin className="mr-2 animate-spin" />
                      Thinking...
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
                submitForm();
              }}
            >
              <div className="relative w-full">
                <div className="pointer-events-none absolute -top-20 left-0 right-0 h-24 bg-gradient-to-t from-background to-transparent" />
                <div className="relative flex items-end">
                  <Textarea
                    ref={textareaRef}
                    placeholder="Send a message..."
                    value={input}
                    onChange={handleInput}
                    className="max-h-[400px] min-h-[48px] w-full resize-none overflow-hidden pr-16 text-base"
                    rows={1}
                    autoFocus
                    onKeyDown={event => {
                      if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                        event.preventDefault();
                        const busy = isAgentMode ? agentStatus !== "idle" : status !== "idle";
                        if (busy) {
                          console.log("Please wait for the model to finish its response");
                        } else {
                          submitForm();
                        }
                      }
                    }}
                  />

                  <div className="absolute bottom-2 right-2 flex items-center">
                    <Button
                      onClick={submitForm}
                      size="icon"
                      disabled={input.trim().length === 0 || (isAgentMode ? agentStatus !== "idle" : status !== "idle")}
                      className="flex h-8 w-8 items-center justify-center rounded-full"
                    >
                      {isAgentMode ? (
                        agentStatus === "submitted" ? (
                          <Spin className="h-4 w-4 animate-spin" />
                        ) : (
                          <ArrowUp className="h-4 w-4" />
                        )
                      ) : status === "loading" ? (
                        <Spin className="h-4 w-4 animate-spin" />
                      ) : (
                        <ArrowUp className="h-4 w-4" />
                      )}
                      <span className="sr-only">
                        {isAgentMode
                          ? agentStatus === "submitted"
                            ? "Processing"
                            : "Send message"
                          : status === "loading"
                            ? "Processing"
                            : "Send message"}
                      </span>
                    </Button>
                  </div>
                </div>
              </div>

              {/* Settings & Agent Buttons */}
              {!isReadonly && (
                <div className="mt-4 flex justify-center gap-2">
                  <Button
                    variant={isAgentMode ? "default" : "outline"}
                    size="sm"
                    className={`group relative overflow-hidden text-xs ${isAgentMode ? "bg-primary hover:bg-primary/90" : "hover:border-primary/50"}`}
                    title="Goes deeper, reasons across documents and may return image-grounded answers"
                    onClick={() => {
                      setIsAgentMode(prev => !prev);
                      setAgentStatus("idle");
                      setShowSettings(false);
                    }}
                  >
                    <span className="flex items-center gap-1.5">
                      {!isAgentMode && <Sparkles className="h-3.5 w-3.5 text-amber-400 dark:text-amber-300" />}
                      <span>{isAgentMode ? "Back to Chat" : "Agent Mode"}</span>
                    </span>
                    {!isAgentMode && (
                      <span className="absolute inset-0 -z-10 translate-x-[-100%] bg-gradient-to-r from-transparent via-primary/10 to-transparent group-hover:animate-shimmer" />
                    )}
                  </Button>
                  {!isAgentMode && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                      onClick={() => {
                        setShowSettings(!showSettings);
                        if (!showSettings && authToken) {
                          fetchGraphs();
                          fetchFolders();
                        }
                      }}
                    >
                      <Settings className="h-3.5 w-3.5" />
                      <span>{showSettings ? "Hide Settings" : "Show Settings"}</span>
                    </Button>
                  )}
                </div>
              )}

              {/* Settings Panel */}
              {showSettings && !isAgentMode && !isReadonly && (
                <div className="mt-4 rounded-xl border bg-muted/30 p-4 duration-300 animate-in fade-in slide-in-from-bottom-2">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-sm font-semibold">Chat Settings</h3>
                    <Button variant="ghost" size="sm" className="text-xs" onClick={() => setShowSettings(false)}>
                      Done
                    </Button>
                  </div>

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    {/* First Column - Core Settings */}
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="use_reranking" className="text-sm">
                            Use Reranking
                          </Label>
                          <Switch
                            id="use_reranking"
                            checked={safeQueryOptions.use_reranking}
                            onCheckedChange={checked => safeUpdateOption("use_reranking", checked)}
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <Label htmlFor="use_colpali" className="text-sm">
                            Use Colpali
                          </Label>
                          <Switch
                            id="use_colpali"
                            checked={safeQueryOptions.use_colpali}
                            onCheckedChange={checked => safeUpdateOption("use_colpali", checked)}
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="graph_name" className="block text-sm">
                          Knowledge Graph
                        </Label>
                        <Select
                          value={safeQueryOptions.graph_name || "__none__"}
                          onValueChange={value =>
                            safeUpdateOption("graph_name", value === "__none__" ? undefined : value)
                          }
                        >
                          <SelectTrigger className="w-full" id="graph_name">
                            <SelectValue placeholder="Select a knowledge graph" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="__none__">None (Standard RAG)</SelectItem>
                            {loadingGraphs ? (
                              <SelectItem value="loading" disabled>
                                Loading graphs...
                              </SelectItem>
                            ) : availableGraphs.length > 0 ? (
                              availableGraphs.map(graphName => (
                                <SelectItem key={graphName} value={graphName}>
                                  {graphName}
                                </SelectItem>
                              ))
                            ) : (
                              <SelectItem value="none_available" disabled>
                                No graphs available
                              </SelectItem>
                            )}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="folder_name" className="block text-sm">
                          Scope to Folder
                        </Label>
                        <Select
                          value={safeQueryOptions.folder_name || "__none__"}
                          onValueChange={value =>
                            safeUpdateOption("folder_name", value === "__none__" ? undefined : value)
                          }
                        >
                          <SelectTrigger className="w-full" id="folder_name">
                            <SelectValue placeholder="Select a folder" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="__none__">All Folders</SelectItem>
                            {loadingFolders ? (
                              <SelectItem value="loading" disabled>
                                Loading folders...
                              </SelectItem>
                            ) : folders.length > 0 ? (
                              folders.map(folder => (
                                <SelectItem key={folder.id || folder.name} value={folder.name}>
                                  {folder.name}
                                </SelectItem>
                              ))
                            ) : (
                              <SelectItem value="none_available" disabled>
                                No folders available
                              </SelectItem>
                            )}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {/* Second Column - Advanced Settings */}
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="query-k" className="flex justify-between text-sm">
                          <span>Results (k)</span>
                          <span className="text-muted-foreground">{safeQueryOptions.k}</span>
                        </Label>
                        <Slider
                          id="query-k"
                          min={1}
                          max={20}
                          step={1}
                          value={[safeQueryOptions.k]}
                          onValueChange={value => safeUpdateOption("k", value[0])}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="query-min-score" className="flex justify-between text-sm">
                          <span>Min Score</span>
                          <span className="text-muted-foreground">{safeQueryOptions.min_score.toFixed(2)}</span>
                        </Label>
                        <Slider
                          id="query-min-score"
                          min={0}
                          max={1}
                          step={0.01}
                          value={[safeQueryOptions.min_score]}
                          onValueChange={value => safeUpdateOption("min_score", value[0])}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="query-temperature" className="flex justify-between text-sm">
                          <span>Temperature</span>
                          <span className="text-muted-foreground">{safeQueryOptions.temperature.toFixed(2)}</span>
                        </Label>
                        <Slider
                          id="query-temperature"
                          min={0}
                          max={2}
                          step={0.01}
                          value={[safeQueryOptions.temperature]}
                          onValueChange={value => safeUpdateOption("temperature", value[0])}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="query-max-tokens" className="flex justify-between text-sm">
                          <span>Max Tokens</span>
                          <span className="text-muted-foreground">{safeQueryOptions.max_tokens}</span>
                        </Label>
                        <Slider
                          id="query-max-tokens"
                          min={1}
                          max={2048}
                          step={1}
                          value={[safeQueryOptions.max_tokens]}
                          onValueChange={value => safeUpdateOption("max_tokens", value[0])}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatSection;
