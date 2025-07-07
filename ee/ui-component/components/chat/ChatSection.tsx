"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useMorphikChat, clearChatCache } from "@/hooks/useMorphikChat";
import { generateUUID } from "@/lib/utils";
import type { QueryOptions } from "@/components/types";
import type { UIMessage } from "./ChatMessages";
import { FolderSummary } from "@/components/types";

import { Settings, Spin, ArrowUp, Sparkles } from "./icons";
import { Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { DocumentSelector } from "@/components/ui/document-selector";
import { PreviewMessage } from "./ChatMessages";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { AgentPreviewMessage, AgentUIMessage, DisplayObject, SourceObject, ToolCall } from "./AgentChatMessages";
import { ModelSelector } from "./ModelSelector";

interface ChatSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialMessages?: UIMessage[];
  isReadonly?: boolean;
  onChatSubmit?: (query: string, options: QueryOptions, initialMessages?: UIMessage[]) => void;
}

// Define an interface for the items coming from the chat history API
// This should be identical or similar to the one in AgentChatSection.tsx
interface ChatHistoryAPIItem {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  agent_data?: {
    tool_history?: ToolCall[];
    display_objects?: DisplayObject[];
    sources?: SourceObject[];
  };
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

  // State for streaming toggle
  const [streamingEnabled, setStreamingEnabled] = useState(true);

  // Initialize our custom hook
  const { messages, input, setInput, status, handleSubmit, queryOptions, updateQueryOption } = useMorphikChat({
    chatId,
    apiBaseUrl,
    authToken,
    initialMessages,
    onChatSubmit,
    streamResponse: streamingEnabled,
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

  // Derive safe option values with sensible defaults to avoid undefined issues in UI
  const safeQueryOptions: Required<Pick<QueryOptions, "k" | "min_score" | "temperature" | "max_tokens" | "padding">> &
    QueryOptions = {
    k: queryOptions.k ?? 5,
    min_score: queryOptions.min_score ?? 0.7,
    temperature: queryOptions.temperature ?? 0.3,
    max_tokens: queryOptions.max_tokens ?? 1024,
    padding: queryOptions.padding ?? 0,
    ...queryOptions,
  };

  // Sidebar collapsed state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // State for settings visibility
  const [showSettings, setShowSettings] = useState(false);
  const [availableGraphs, setAvailableGraphs] = useState<string[]>([]);
  const [loadingGraphs, setLoadingGraphs] = useState(false);
  const [loadingFolders, setLoadingFolders] = useState(false);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [documents, setDocuments] = useState<
    {
      id: string;
      filename: string;
      folder_name?: string;
      content_type?: string;
      metadata?: Record<string, unknown>;
      system_metadata?: unknown;
    }[]
  >([]);

  // Model selection state
  const [selectedModel, setSelectedModel] = useState<string>("");

  // Agent mode toggle and state
  const [isAgentMode, setIsAgentMode] = useState(false);
  const [agentMessages, setAgentMessages] = useState<AgentUIMessage[]>([]);
  const [agentStatus, setAgentStatus] = useState<"idle" | "submitted" | "completed">("idle");

  // Load agent messages from chat history when switching to agent mode
  useEffect(() => {
    const loadAgentHistory = async () => {
      if (isAgentMode && chatId && apiBaseUrl && (authToken || apiBaseUrl.includes("localhost"))) {
        try {
          const response = await fetch(`${apiBaseUrl}/chat/${chatId}`, {
            headers: {
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
          });
          if (response.ok) {
            const data: ChatHistoryAPIItem[] = await response.json(); // Typed data
            const agentMessagesFromHistory = data.map((m: ChatHistoryAPIItem): AgentUIMessage => {
              // Replaced any with ChatHistoryAPIItem, map to AgentUIMessage
              const baseMessage: AgentUIMessage = {
                id: generateUUID(),
                role: m.role,
                content: m.content,
                createdAt: new Date(m.timestamp),
              };

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
            setAgentMessages(agentMessagesFromHistory);
          }
        } catch (err) {
          console.error("Failed to load agent chat history", err);
        }
      } else if (!isAgentMode) {
        // Clear agent messages when switching back to regular chat mode
        setAgentMessages([]);
      }
    };

    loadAgentHistory();
  }, [isAgentMode, chatId, apiBaseUrl, authToken]);

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
      console.log(`Fetching folders from: ${apiBaseUrl}/folders/summary`);
      const response = await fetch(`${apiBaseUrl}/folders/summary`, {
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

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    if (!apiBaseUrl) return;

    setLoadingDocuments(true);
    try {
      console.log(`Fetching documents from: ${apiBaseUrl}/documents`);
      const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({}), // Empty body to fetch all docs
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`);
      }

      const documentsData = await response.json();
      console.log("Documents data received:", documentsData);

      if (Array.isArray(documentsData)) {
        // Transform documents to the format we need (id, filename, and folder info)
        const transformedDocs = documentsData
          .map((doc: unknown) => {
            const docObj = doc as Record<string, unknown>;
            const id = (docObj.external_id as string) || (docObj.id as string);
            if (!id) return null; // Skip documents without valid IDs

            return {
              id,
              filename: (docObj.filename as string) || (docObj.name as string) || `Document ${id}`,
              folder_name:
                (docObj.folder_name as string) ||
                ((docObj.system_metadata as Record<string, unknown>)?.folder_name as string),
              content_type: docObj.content_type as string,
              metadata: docObj.metadata as Record<string, unknown>,
              system_metadata: docObj.system_metadata,
            };
          })
          .filter(doc => doc !== null) as {
          id: string;
          filename: string;
          folder_name?: string;
          content_type?: string;
          metadata?: Record<string, unknown>;
          system_metadata?: unknown;
        }[];

        setDocuments(transformedDocs);
      } else {
        console.error("Expected array for documents data but received:", typeof documentsData);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
    } finally {
      setLoadingDocuments(false);
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
        await fetchDocuments();
      }
    };

    fetchData();
  }, [authToken, apiBaseUrl, fetchGraphs, fetchFolders, fetchDocuments]);

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
        body: JSON.stringify({
          query: userMessage.content,
          chat_id: chatId,
        }),
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

  // Handle model selection change
  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);

    // Handle default model - clear llm_config to use server default
    if (modelId === "default") {
      safeUpdateOption("llm_config", undefined);
      return;
    }

    // Check if this is a custom model
    if (modelId.startsWith("custom_")) {
      const savedModels = localStorage.getItem("morphik_custom_models");
      if (savedModels) {
        try {
          const customModels = JSON.parse(savedModels);
          const customModel = customModels.find((m: { id: string }) => `custom_${m.id}` === modelId);

          if (customModel) {
            // Use the custom model's config directly
            safeUpdateOption("llm_config", customModel.config);
            return;
          }
        } catch (err) {
          console.error("Failed to parse custom models:", err);
        }
      }
    }

    // Get API keys from localStorage
    const savedConfig = localStorage.getItem("morphik_api_keys");
    if (savedConfig) {
      try {
        const config = JSON.parse(savedConfig);

        // Build model_config based on selected model and saved API keys
        const modelConfig: Record<string, unknown> = { model: modelId };

        // Determine provider from model ID
        if (modelId.startsWith("gpt")) {
          if (config.openai?.apiKey) {
            modelConfig.api_key = config.openai.apiKey;
            if (config.openai.baseUrl) {
              modelConfig.base_url = config.openai.baseUrl;
            }
          }
        } else if (modelId.startsWith("claude")) {
          if (config.anthropic?.apiKey) {
            modelConfig.api_key = config.anthropic.apiKey;
            if (config.anthropic.baseUrl) {
              modelConfig.base_url = config.anthropic.baseUrl;
            }
          }
        } else if (modelId.startsWith("gemini/")) {
          if (config.google?.apiKey) {
            modelConfig.api_key = config.google.apiKey;
          }
        } else if (modelId.startsWith("groq/")) {
          if (config.groq?.apiKey) {
            modelConfig.api_key = config.groq.apiKey;
          }
        } else if (modelId.startsWith("deepseek/")) {
          if (config.deepseek?.apiKey) {
            modelConfig.api_key = config.deepseek.apiKey;
          }
        }

        safeUpdateOption("llm_config", modelConfig);
      } catch (err) {
        console.error("Failed to parse API keys:", err);
      }
    }
  };

  return (
    <div className="relative flex h-full w-full overflow-hidden bg-background">
      {/* Sidebar */}
      <ChatSidebar
        apiBaseUrl={apiBaseUrl}
        authToken={authToken}
        activeChatId={chatId}
        onSelect={id => {
          // Clear chat cache when switching to ensure fresh data
          clearChatCache(chatId, apiBaseUrl);
          setChatId(id ?? generateUUID());
        }}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(prev => !prev)}
      />

      {/* Main chat area */}
      <div className="flex h-full flex-1 flex-col">
        {/* Conditional layout based on whether there are messages */}
        {(isAgentMode ? agentMessages.length === 0 : messages.length === 0) ? (
          /* Empty state - centered layout with controls */
          <div className="flex h-full flex-1 flex-col items-center justify-center transition-all duration-700 ease-out">
            <div className="mb-12 flex flex-col items-center justify-center text-center">
              <div className="mb-4 flex items-center gap-3">
                <h1 className="text-4xl font-light text-foreground">How can I help?</h1>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                      >
                        <Info className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-sm text-center">
                      <p className="text-sm">
                        {isAgentMode
                          ? "Agent Mode: Advanced AI with reasoning, tool use, and multimodal capabilities. May take longer but provides deeper insights."
                          : "Chat Mode: Fast document search and retrieval. Perfect for quick questions and finding specific information."}
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <p className="max-w-2xl text-lg text-muted-foreground">
                Ask me anything about your documents. I&apos;ll search through them and provide you with relevant
                information.
              </p>
            </div>

            {/* Centered input area for empty state */}
            <div className="w-full max-w-4xl px-4">
              {/* Controls Row - Folder Selection and Agent Mode */}
              {!isReadonly && (
                <div className="border-b border-border/50 pb-3 pt-3">
                  <div className="flex items-center justify-between gap-4">
                    {/* Left side - Document and Folder Selection (only in chat mode) */}
                    {!isAgentMode && (
                      <div className="mr-4 flex-1">
                        <DocumentSelector
                          documents={documents}
                          folders={folders.map(folder => ({
                            name: folder.name,
                            doc_count: folder.doc_count || 0,
                          }))}
                          selectedDocuments={getCurrentSelectedDocuments()}
                          selectedFolders={getCurrentSelectedFolders()}
                          onDocumentSelectionChange={(selectedDocumentIds: string[]) => {
                            updateDocumentFilter(selectedDocumentIds);
                          }}
                          onFolderSelectionChange={(selectedFolderNames: string[]) => {
                            safeUpdateOption(
                              "folder_name",
                              selectedFolderNames.length > 0 ? selectedFolderNames : undefined
                            );
                          }}
                          loading={loadingDocuments || loadingFolders}
                          placeholder="Select documents and folders"
                          className="w-full"
                        />
                      </div>
                    )}

                    {/* Right side - Agent Mode and Settings */}
                    <div className={`flex items-center gap-2 ${isAgentMode ? "ml-auto" : ""}`}>
                      <Button
                        variant={isAgentMode ? "default" : "outline"}
                        size="sm"
                        className="text-xs font-medium transition-all hover:border-primary/50"
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
                          className="flex items-center gap-1 text-xs font-medium transition-all hover:border-primary/50"
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

              {/* Input Form for centered state */}
              <form onSubmit={isAgentMode ? handleAgentSubmit : handleSubmit} className="relative space-y-4 py-4">
                <div className="relative">
                  <Textarea
                    ref={textareaRef}
                    placeholder={isReadonly ? "Chat is read-only" : "Ask a question..."}
                    value={input}
                    onChange={handleInput}
                    onKeyDown={e => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        if (isAgentMode) {
                          handleAgentSubmit();
                        } else {
                          handleSubmit();
                        }
                      }
                    }}
                    disabled={isReadonly || (isAgentMode ? agentStatus === "submitted" : status === "loading")}
                    className="min-h-[60px] resize-none pr-12 text-base"
                    style={{ height: "auto" }}
                  />

                  <Button
                    type="submit"
                    disabled={
                      !input.trim() || isReadonly || (isAgentMode ? agentStatus === "submitted" : status === "loading")
                    }
                    size="sm"
                    className="absolute bottom-2 right-2 h-8 w-8 p-0"
                  >
                    {isAgentMode && agentStatus === "submitted" ? (
                      <Spin className="h-4 w-4 animate-spin" />
                    ) : status === "loading" ? (
                      <Spin className="h-4 w-4 animate-spin" />
                    ) : (
                      <ArrowUp className="h-4 w-4" />
                    )}
                  </Button>
                </div>

                {/* Model Selector - below input (always reserve space) */}
                <div className="mt-2 flex min-h-[32px] items-center justify-between px-2">
                  {!isAgentMode && (
                    <ModelSelector
                      apiBaseUrl={apiBaseUrl}
                      authToken={authToken}
                      selectedModel={selectedModel || "default"}
                      onModelChange={handleModelChange}
                      onRequestApiKey={() => {
                        // Navigate to settings page with API keys tab
                        window.location.href = "?section=settings";
                      }}
                    />
                  )}
                </div>

                {/* Settings Panel */}
                {showSettings && !isAgentMode && !isReadonly && (
                  <div className="mt-4 rounded-lg border border-border/50 bg-muted/20 p-4 shadow-sm duration-300 animate-in fade-in slide-in-from-bottom-2">
                    <div className="mb-4 flex items-center justify-between">
                      <h3 className="text-sm font-semibold">Advanced Settings</h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 text-xs hover:bg-muted/50"
                        onClick={() => setShowSettings(false)}
                      >
                        Done
                      </Button>
                    </div>

                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                      {/* First Column - Core Settings */}
                      <div className="space-y-4">
                        <div className="space-y-3">
                          <div className="flex items-center justify-between rounded-lg bg-background/50 p-3">
                            <Label htmlFor="use_reranking" className="text-sm font-medium">
                              Use Reranking
                            </Label>
                            <Switch
                              id="use_reranking"
                              checked={safeQueryOptions.use_reranking}
                              onCheckedChange={checked => safeUpdateOption("use_reranking", checked)}
                            />
                          </div>
                          <div className="flex items-center justify-between rounded-lg bg-background/50 p-3">
                            <Label htmlFor="use_colpali" className="text-sm font-medium">
                              Use Colpali
                            </Label>
                            <Switch
                              id="use_colpali"
                              checked={safeQueryOptions.use_colpali}
                              onCheckedChange={checked => safeUpdateOption("use_colpali", checked)}
                            />
                          </div>
                          {safeQueryOptions.use_colpali && (
                            <div className="space-y-2 rounded-lg bg-background/50 p-3">
                              <Label htmlFor="query-padding" className="flex justify-between text-sm font-medium">
                                <span>Padding</span>
                                <span className="text-muted-foreground">{safeQueryOptions.padding || 0}</span>
                              </Label>
                              <Slider
                                id="query-padding"
                                min={0}
                                max={10}
                                step={1}
                                value={[safeQueryOptions.padding || 0]}
                                onValueChange={value => safeUpdateOption("padding", value[0])}
                                className="w-full"
                              />
                              <p className="text-xs text-muted-foreground">
                                Additional pages to retrieve before and after matched pages
                              </p>
                            </div>
                          )}
                          <div className="flex items-center justify-between rounded-lg bg-background/50 p-3">
                            <Label htmlFor="streaming_enabled" className="text-sm font-medium">
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
                          <Label htmlFor="graph_name" className="block text-sm font-medium">
                            Knowledge Graph
                          </Label>
                          <Select
                            value={safeQueryOptions.graph_name || "__none__"}
                            onValueChange={value =>
                              safeUpdateOption("graph_name", value === "__none__" ? undefined : value)
                            }
                          >
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select a graph..." />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="__none__">No Graph</SelectItem>
                              {availableGraphs.map(graph => (
                                <SelectItem key={graph} value={graph}>
                                  {graph}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      {/* Second Column - Query Parameters */}
                      <div className="space-y-4">
                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-k" className="flex justify-between text-sm font-medium">
                            <span>Top K Results</span>
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
                          <p className="text-xs text-muted-foreground">Number of document chunks to retrieve</p>
                        </div>

                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-min-score" className="flex justify-between text-sm font-medium">
                            <span>Min Score</span>
                            <span className="text-muted-foreground">{safeQueryOptions.min_score}</span>
                          </Label>
                          <Slider
                            id="query-min-score"
                            min={0}
                            max={1}
                            step={0.1}
                            value={[safeQueryOptions.min_score]}
                            onValueChange={value => safeUpdateOption("min_score", value[0])}
                            className="w-full"
                          />
                          <p className="text-xs text-muted-foreground">Minimum similarity score for results</p>
                        </div>

                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-temperature" className="flex justify-between text-sm font-medium">
                            <span>Temperature</span>
                            <span className="text-muted-foreground">{safeQueryOptions.temperature}</span>
                          </Label>
                          <Slider
                            id="query-temperature"
                            min={0}
                            max={2}
                            step={0.1}
                            value={[safeQueryOptions.temperature]}
                            onValueChange={value => safeUpdateOption("temperature", value[0])}
                            className="w-full"
                          />
                          <p className="text-xs text-muted-foreground">Controls randomness in responses</p>
                        </div>

                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-max-tokens" className="flex justify-between text-sm font-medium">
                            <span>Max Tokens</span>
                            <span className="text-muted-foreground">{safeQueryOptions.max_tokens}</span>
                          </Label>
                          <Slider
                            id="query-max-tokens"
                            min={100}
                            max={4000}
                            step={100}
                            value={[safeQueryOptions.max_tokens]}
                            onValueChange={value => safeUpdateOption("max_tokens", value[0])}
                            className="w-full"
                          />
                          <p className="text-xs text-muted-foreground">Maximum length of the response</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </form>
            </div>
          </div>
        ) : (
          /* Messages present - normal layout */
          <div className="relative min-h-0 flex-1 transition-all duration-700 ease-out">
            <ScrollArea className="h-full" ref={messagesContainerRef}>
              <div className="mx-auto flex max-w-4xl flex-col pb-32 pt-8">
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
                      <div className="flex h-12 items-center justify-start pl-4 text-start text-sm text-muted-foreground">
                        <Spin className="mr-2 h-4 w-4 animate-spin" />
                        <span>Agent thinking...</span>
                      </div>
                    )
                  : status === "loading" &&
                    messages.length > 0 &&
                    messages[messages.length - 1].role === "user" && (
                      <div className="flex h-12 items-center justify-start pl-4 text-start text-sm text-muted-foreground">
                        <Spin className="mr-2 h-4 w-4 animate-spin" />
                        <span>Thinking...</span>
                      </div>
                    )}
              </div>

              <div ref={messagesEndRef} className="min-h-[24px] min-w-[24px] shrink-0" />
            </ScrollArea>
          </div>
        )}

        {/* Input Area - only shown when there are messages */}
        {(isAgentMode ? agentMessages.length > 0 : messages.length > 0) && (
          <div className="sticky bottom-0 w-full bg-background transition-all duration-700 ease-out">
            <div className="mx-auto max-w-4xl px-4">
              {/* Controls Row - Folder Selection and Agent Mode */}
              {!isReadonly && (
                <div className="border-b border-border/50 pb-3 pt-3">
                  <div className="flex items-center justify-between gap-4">
                    {/* Left side - Document and Folder Selection (only in chat mode) */}
                    {!isAgentMode && (
                      <div className="mr-4 flex-1">
                        <DocumentSelector
                          documents={documents}
                          folders={folders.map(folder => ({
                            name: folder.name,
                            doc_count: folder.doc_count || 0,
                          }))}
                          selectedDocuments={getCurrentSelectedDocuments()}
                          selectedFolders={getCurrentSelectedFolders()}
                          onDocumentSelectionChange={(selectedDocumentIds: string[]) => {
                            updateDocumentFilter(selectedDocumentIds);
                          }}
                          onFolderSelectionChange={(selectedFolderNames: string[]) => {
                            safeUpdateOption(
                              "folder_name",
                              selectedFolderNames.length > 0 ? selectedFolderNames : undefined
                            );
                          }}
                          loading={loadingDocuments || loadingFolders}
                          placeholder="Select documents and folders"
                          className="w-full"
                        />
                      </div>
                    )}

                    {/* Right side - Agent Mode and Settings */}
                    <div className={`flex items-center gap-2 ${isAgentMode ? "ml-auto" : ""}`}>
                      <Button
                        variant={isAgentMode ? "default" : "outline"}
                        size="sm"
                        className="text-xs font-medium transition-all hover:border-primary/50"
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
                          className="flex items-center gap-1 text-xs font-medium transition-all hover:border-primary/50"
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

              <form
                className="pb-6 pt-4"
                onSubmit={e => {
                  e.preventDefault();
                  submitForm();
                }}
              >
                <div className="relative w-full">
                  <div className="relative flex items-end">
                    <Textarea
                      ref={textareaRef}
                      placeholder="Send a message..."
                      value={input}
                      onChange={handleInput}
                      className="max-h-[400px] min-h-[52px] w-full resize-none overflow-hidden rounded-lg border border-border bg-background pr-14 text-base transition-colors focus:border-primary"
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
                        disabled={
                          input.trim().length === 0 || (isAgentMode ? agentStatus !== "idle" : status !== "idle")
                        }
                        className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground transition-all hover:bg-primary/90 disabled:opacity-50"
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

                {/* Model Selector - below input (always reserve space) */}
                <div className="mt-2 flex min-h-[32px] items-center justify-between px-2">
                  {!isAgentMode && (
                    <ModelSelector
                      apiBaseUrl={apiBaseUrl}
                      authToken={authToken}
                      selectedModel={selectedModel || "default"}
                      onModelChange={handleModelChange}
                      onRequestApiKey={() => {
                        // Navigate to settings page with API keys tab
                        window.location.href = "?section=settings";
                      }}
                    />
                  )}
                </div>

                {/* Settings Panel */}
                {showSettings && !isAgentMode && !isReadonly && (
                  <div className="mt-4 rounded-lg border border-border/50 bg-muted/20 p-4 shadow-sm duration-300 animate-in fade-in slide-in-from-bottom-2">
                    <div className="mb-4 flex items-center justify-between">
                      <h3 className="text-sm font-semibold">Advanced Settings</h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 text-xs hover:bg-muted/50"
                        onClick={() => setShowSettings(false)}
                      >
                        Done
                      </Button>
                    </div>

                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                      {/* First Column - Core Settings */}
                      <div className="space-y-4">
                        <div className="space-y-3">
                          <div className="flex items-center justify-between rounded-lg bg-background/50 p-3">
                            <Label htmlFor="use_reranking" className="text-sm font-medium">
                              Use Reranking
                            </Label>
                            <Switch
                              id="use_reranking"
                              checked={safeQueryOptions.use_reranking}
                              onCheckedChange={checked => safeUpdateOption("use_reranking", checked)}
                            />
                          </div>
                          <div className="flex items-center justify-between rounded-lg bg-background/50 p-3">
                            <Label htmlFor="use_colpali" className="text-sm font-medium">
                              Use Colpali
                            </Label>
                            <Switch
                              id="use_colpali"
                              checked={safeQueryOptions.use_colpali}
                              onCheckedChange={checked => safeUpdateOption("use_colpali", checked)}
                            />
                          </div>
                          {safeQueryOptions.use_colpali && (
                            <div className="space-y-2 rounded-lg bg-background/50 p-3">
                              <Label htmlFor="query-padding" className="flex justify-between text-sm font-medium">
                                <span>Padding</span>
                                <span className="text-muted-foreground">{safeQueryOptions.padding || 0}</span>
                              </Label>
                              <Slider
                                id="query-padding"
                                min={0}
                                max={10}
                                step={1}
                                value={[safeQueryOptions.padding || 0]}
                                onValueChange={value => safeUpdateOption("padding", value[0])}
                                className="w-full"
                              />
                              <p className="text-xs text-muted-foreground">
                                Additional pages to retrieve before and after matched pages
                              </p>
                            </div>
                          )}
                          <div className="flex items-center justify-between rounded-lg bg-background/50 p-3">
                            <Label htmlFor="streaming_enabled" className="text-sm font-medium">
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
                          <Label htmlFor="graph_name" className="block text-sm font-medium">
                            Knowledge Graph
                          </Label>
                          <Select
                            value={safeQueryOptions.graph_name || "__none__"}
                            onValueChange={value =>
                              safeUpdateOption("graph_name", value === "__none__" ? undefined : value)
                            }
                          >
                            <SelectTrigger
                              className="w-full border-border/50 bg-background/50 shadow-sm transition-colors hover:border-primary/50"
                              id="graph_name"
                            >
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
                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-k" className="flex justify-between text-sm font-medium">
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

                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-min-score" className="flex justify-between text-sm font-medium">
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

                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-temperature" className="flex justify-between text-sm font-medium">
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

                        <div className="space-y-2 rounded-lg bg-background/50 p-3">
                          <Label htmlFor="query-max-tokens" className="flex justify-between text-sm font-medium">
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
        )}
      </div>
    </div>
  );
};

export default ChatSection;
