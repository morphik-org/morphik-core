"use client";

import React, { useState, useEffect, useCallback } from "react";
import { AssistantRuntimeProvider, ThreadPrimitive, ComposerPrimitive, MessagePrimitive } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Folder } from "@/components/types";
import { generateUUID } from "@/lib/utils";
import type { QueryOptions } from "@/components/types";

import { Settings, Sparkles } from "./icons";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MultiSelect } from "@/components/ui/multi-select";
import { Slider } from "@/components/ui/slider";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { AgentPreviewMessage, AgentUIMessage, ToolCall, DisplayObject, SourceObject } from "./AgentChatMessages";

interface AssistantChatSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  isReadonly?: boolean;
  onChatSubmit?: (query: string, options: QueryOptions) => void;
}

// Interface for document API response
interface ApiDocumentResponse {
  external_id?: string;
  id?: string;
  filename?: string;
  name?: string;
}

/**
 * Modern ChatSection component using assistant-ui with Morphik integration
 */
const AssistantChatSection: React.FC<AssistantChatSectionProps> = ({
  apiBaseUrl,
  authToken,
  isReadonly = false,
  onChatSubmit,
}) => {
  // Selected chat ID – start with fresh conversation
  const [chatId, setChatId] = useState<string>(() => generateUUID());

  // State for streaming toggle
  const [streamingEnabled, setStreamingEnabled] = useState(true);

  // Initialize runtime with basic chat functionality
  const runtime = useChatRuntime({
    api: `/api/morphik-chat?chatId=${chatId}&apiBaseUrl=${encodeURIComponent(apiBaseUrl)}&authToken=${encodeURIComponent(authToken || "")}`,
  });

  // Query options state (moved from runtime)
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: "{}",
    k: 5,
    min_score: 0.7,
    use_reranking: false,
    use_colpali: true,
    max_tokens: 1024,
    temperature: 0.3,
  });

  const updateQueryOption = useCallback((key: keyof QueryOptions, value: QueryOptions[keyof QueryOptions]) => {
    setQueryOptions(prev => ({ ...prev, [key]: value }));
  }, []);

  // Helper to safely update options
  const safeUpdateOption = useCallback(
    <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => {
      if (updateQueryOption) {
        updateQueryOption(key, value);
      }
    },
    [updateQueryOption]
  );

  // Helper to update filters with external_id
  const updateDocumentFilter = useCallback(
    (selectedDocumentIds: string[]) => {
      if (updateQueryOption) {
        const currentFilters = queryOptions.filters || {};
        const parsedFilters = typeof currentFilters === "string" ? JSON.parse(currentFilters || "{}") : currentFilters;

        const newFilters = {
          ...parsedFilters,
          external_id: selectedDocumentIds.length > 0 ? selectedDocumentIds : undefined,
        };

        // Remove undefined values
        Object.keys(newFilters).forEach(key => newFilters[key] === undefined && delete newFilters[key]);

        updateQueryOption("filters", newFilters);
      }
    },
    [updateQueryOption, queryOptions.filters]
  );

  // Derive safe option values with sensible defaults
  const safeQueryOptions: Required<Pick<QueryOptions, "k" | "min_score" | "temperature" | "max_tokens">> &
    QueryOptions = {
    k: queryOptions.k ?? 5,
    min_score: queryOptions.min_score ?? 0.7,
    temperature: queryOptions.temperature ?? 0.3,
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
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [documents, setDocuments] = useState<{ id: string; filename: string }[]>([]);

  // Agent mode toggle and state
  const [isAgentMode, setIsAgentMode] = useState(false);
  const [agentMessages, setAgentMessages] = useState<AgentUIMessage[]>([]);
  const [agentStatus, setAgentStatus] = useState<"idle" | "submitted" | "completed">("idle");

  // Fetch available graphs for dropdown
  const fetchGraphs = useCallback(async () => {
    if (!apiBaseUrl) return;

    setLoadingGraphs(true);
    try {
      const response = await fetch(`${apiBaseUrl}/graphs`, {
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.status} ${response.statusText}`);
      }

      const graphsData = await response.json();

      if (Array.isArray(graphsData)) {
        setAvailableGraphs(graphsData.map((graph: { name: string }) => graph.name));
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
      const response = await fetch(`${apiBaseUrl}/folders`, {
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch folders: ${response.status} ${response.statusText}`);
      }

      const foldersData = await response.json();

      if (Array.isArray(foldersData)) {
        setFolders(foldersData);
      }
    } catch (err) {
      console.error("Error fetching folders:", err);
    } finally {
      setLoadingFolders(false);
    }
  }, [apiBaseUrl, authToken]);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    if (!apiBaseUrl) return;

    setLoadingDocuments(true);
    try {
      const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`);
      }

      const documentsData = await response.json();

      if (Array.isArray(documentsData)) {
        const transformedDocs = documentsData
          .map((doc: ApiDocumentResponse) => {
            const id = doc.external_id || doc.id;
            if (!id) return null;

            return {
              id,
              filename: doc.filename || doc.name || `Document ${id}`,
            };
          })
          .filter((doc): doc is { id: string; filename: string } => doc !== null);

        setDocuments(transformedDocs);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
    } finally {
      setLoadingDocuments(false);
    }
  }, [apiBaseUrl, authToken]);

  // Fetch data when component mounts
  useEffect(() => {
    const fetchData = async () => {
      if (authToken || apiBaseUrl.includes("localhost")) {
        await fetchGraphs();
        await fetchFolders();
        await fetchDocuments();
      }
    };

    fetchData();
  }, [authToken, apiBaseUrl, fetchGraphs, fetchFolders, fetchDocuments]);

  // Agent mode submit handler
  const handleAgentSubmit = async (input: string) => {
    if (!input.trim() || agentStatus === "submitted" || isReadonly) return;

    const userMessage: AgentUIMessage = {
      id: generateUUID(),
      role: "user",
      content: input.trim(),
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

    try {
      const response = await fetch(`${apiBaseUrl}/agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({ query: input }),
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

  // Get current selected values
  const getCurrentSelectedFolders = (): string[] => {
    const folderName = safeQueryOptions.folder_name;
    if (!folderName) return [];
    const folders = Array.isArray(folderName) ? folderName : [folderName];
    return folders.filter(f => f !== "__none__");
  };

  const getCurrentSelectedDocuments = (): string[] => {
    const filters = safeQueryOptions.filters || {};
    const parsedFilters = typeof filters === "string" ? JSON.parse(filters || "{}") : filters;
    const externalId = parsedFilters.external_id;
    if (!externalId) return [];
    const documents = Array.isArray(externalId) ? externalId : [externalId];
    return documents.filter(d => d !== "__none__");
  };

  // Custom Composer component for agent mode
  const AgentComposer = () => {
    const [input, setInput] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      if (input.trim()) {
        handleAgentSubmit(input);
        setInput("");
      }
    };

    return (
      <form onSubmit={handleSubmit} className="w-full">
        <div className="relative flex items-end">
          <textarea
            placeholder="Send a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="min-h-[48px] max-h-[400px] w-full resize-none overflow-hidden pr-16 text-base border rounded-lg p-3"
            rows={1}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (agentStatus === "idle" && input.trim()) {
                  handleSubmit(event);
                }
              }
            }}
          />
          <div className="absolute bottom-2 right-2">
            <Button
              type="submit"
              size="icon"
              disabled={input.trim().length === 0 || agentStatus !== "idle"}
              className="flex h-8 w-8 items-center justify-center rounded-full"
            >
              <span className="sr-only">Send message</span>
              →
            </Button>
          </div>
        </div>
      </form>
    );
  };

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
        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadPrimitive.Root className="flex h-full flex-col">
            {/* Controls Row - Folder Selection and Agent Mode */}
            {!isReadonly && (
              <div className="border-b bg-background p-4">
                <div className="flex items-center justify-between gap-4">
                  {/* Left side - Folder and Document Selection (only in chat mode) */}
                  {!isAgentMode && (
                    <div className="flex items-center gap-4">
                      {/* Folder Selection */}
                      <div className="flex items-center gap-2">
                        <Label htmlFor="folder_name" className="whitespace-nowrap text-sm text-muted-foreground">
                          Folder:
                        </Label>
                        <MultiSelect
                          options={[
                            { label: "All Folders", value: "__none__" },
                            ...(loadingFolders ? [{ label: "Loading folders...", value: "loading" }] : []),
                            ...folders.map(folder => ({
                              label: folder.name,
                              value: folder.name,
                            })),
                          ]}
                          selected={getCurrentSelectedFolders()}
                          onChange={(value: string[]) => {
                            const filteredValues = value.filter(v => v !== "__none__");
                            safeUpdateOption("folder_name", filteredValues.length > 0 ? filteredValues : undefined);
                          }}
                          placeholder="All folders"
                          className="w-[200px]"
                        />
                      </div>

                      {/* Document Selection */}
                      <div className="flex items-center gap-2">
                        <Label htmlFor="document_filter" className="whitespace-nowrap text-sm text-muted-foreground">
                          Document:
                        </Label>
                        <MultiSelect
                          options={[
                            { label: "All Documents", value: "__none__" },
                            ...(loadingDocuments ? [{ label: "Loading documents...", value: "loading" }] : []),
                            ...documents.map(doc => ({
                              label: doc.filename,
                              value: doc.id,
                            })),
                          ]}
                          selected={getCurrentSelectedDocuments()}
                          onChange={(value: string[]) => {
                            const filteredValues = value.filter(v => v !== "__none__");
                            updateDocumentFilter(filteredValues);
                          }}
                          placeholder="All documents"
                          className="w-[220px]"
                        />
                      </div>
                    </div>
                  )}

                  {/* Right side - Agent Mode and Settings */}
                  <div className={`flex items-center gap-2 ${isAgentMode ? "ml-auto" : ""}`}>
                    <Button
                      variant={isAgentMode ? "default" : "outline"}
                      size="sm"
                      className="text-xs font-medium"
                      title="Goes deeper, reasons across documents and may return image-grounded answers"
                      onClick={() => {
                        setIsAgentMode(prev => !prev);
                        setAgentStatus("idle");
                        setShowSettings(false);
                      }}
                    >
                      <span className="flex items-center gap-1.5">
                        {!isAgentMode && <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400" />}
                        <span>{isAgentMode ? "Chat Mode" : "Agent Mode"}</span>
                      </span>
                    </Button>
                    {!isAgentMode && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-1 text-xs font-medium"
                        onClick={() => {
                          setShowSettings(!showSettings);
                          if (!showSettings && authToken) {
                            fetchGraphs();
                            fetchFolders();
                            fetchDocuments();
                          }
                        }}
                      >
                        <Settings className="h-3.5 w-3.5" />
                        <span>{showSettings ? "Hide" : "Settings"}</span>
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Messages Area */}
            {isAgentMode ? (
              <div className="flex-1 overflow-hidden">
                <div className="h-full overflow-y-auto p-4">
                  {agentMessages.length === 0 ? (
                    <div className="flex flex-1 items-center justify-center p-8 text-center">
                      <div className="max-w-md space-y-2">
                        <h2 className="text-xl font-semibold">Morphik Agent Chat</h2>
                        <p className="text-sm text-muted-foreground">
                          Ask a question to the agent to get started.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {agentMessages.map(msg => (
                        <AgentPreviewMessage key={msg.id} message={msg} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <ThreadPrimitive.Viewport className="flex-1 overflow-hidden">
                <div className="h-full overflow-y-auto p-4">
                  <ThreadPrimitive.Messages />
                </div>
              </ThreadPrimitive.Viewport>
            )}

            {/* Input Area */}
            <div className="border-t bg-background p-4">
              {/* Settings Panel */}
              {showSettings && !isAgentMode && !isReadonly && (
                <div className="mb-4 rounded-xl border bg-muted/30 p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-sm font-semibold">Advanced Settings</h3>
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
                        <div className="flex items-center justify-between">
                          <Label htmlFor="streaming_enabled" className="text-sm">
                            Streaming Response
                          </Label>
                          <Switch
                            id="streaming_enabled"
                            checked={streamingEnabled}
                            onCheckedChange={setStreamingEnabled}
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

              {/* Input Composer */}
              {isAgentMode ? (
                <AgentComposer />
              ) : (
                <ComposerPrimitive.Root className="w-full">
                  <div className="relative flex items-end">
                    <ComposerPrimitive.Input className="min-h-[48px] max-h-[400px] w-full resize-none overflow-hidden pr-16 text-base border rounded-lg p-3" />
                    <ComposerPrimitive.Send asChild>
                      <Button size="icon" className="absolute bottom-2 right-2 h-8 w-8 rounded-full">
                        →
                      </Button>
                    </ComposerPrimitive.Send>
                  </div>
                </ComposerPrimitive.Root>
              )}
            </div>
          </ThreadPrimitive.Root>
        </AssistantRuntimeProvider>
      </div>
    </div>
  );
};

export default AssistantChatSection;
