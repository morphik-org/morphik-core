"use client";

import React, { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import {
  ZoomIn,
  ZoomOut,
  RotateCw,
  ChevronLeft,
  ChevronRight,
  FileText,
  Maximize2,
  User,
  Cpu,
  MessageSquare,
  X,
  GripVertical,
  Send,
  FolderOpen,
  Clock,
  CheckCircle,
  AlertCircle,
  Plus,
} from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { usePDFChatSessions } from "@/hooks/useChatSessions";
import { usePDFSession } from "@/components/pdf/PDFAPIService";

// Configure PDF.js worker - use CDN for reliability
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

interface PDFViewerProps {
  apiBaseUrl?: string;
  authToken?: string | null;
  initialDocumentId?: string; // Add prop to load a specific document on initialization
}

interface PDFState {
  file: File | null;
  currentPage: number;
  totalPages: number;
  scale: number;
  rotation: number;
  pdfDataUrl: string | null;
  controlMode: "manual" | "api"; // New mode toggle
  documentName?: string; // Add document name for selected documents
  documentId?: string; // Add document ID for selected documents
}

interface ZoomBounds {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  timestamp: Date;
  // For assistant messages with tool calls
  tool_calls?: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
  // For tool response messages
  tool_call_id?: string;
  name?: string;
  // For tool messages with additional data
  metadata?: Record<string, unknown>;
  current_frame?: string;
  args?: Record<string, unknown>;
}

interface AgentData {
  display_objects?: unknown[];
  tool_history?: unknown[];
  sources?: unknown[];
}

interface ApiChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  timestamp: string;
  agent_data?: AgentData;
  // For assistant messages with tool calls
  tool_calls?: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
  // For tool response messages
  tool_call_id?: string;
  name?: string;
  // For tool messages with additional data
  metadata?: Record<string, unknown>;
  current_frame?: string;
  args?: Record<string, unknown>;
}

interface PDFDocument {
  id: string;
  filename: string;
  download_url: string;
  created_at?: string;
  folder_name?: string;
  status: string;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function PDFViewer({ apiBaseUrl, authToken, initialDocumentId }: PDFViewerProps) {
  // Get session information from PDF API service context (optional)
  const pdfSession = usePDFSession() || {
    sessionId: `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    userId: "anonymous",
  };

  const [pdfState, setPdfState] = useState<PDFState>({
    file: null,
    currentPage: 1,
    totalPages: 0,
    scale: 1.0,
    rotation: 0,
    pdfDataUrl: null,
    controlMode: "manual", // Default to manual control
  });

  const [, setIsLoading] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [zoomBounds, setZoomBounds] = useState<ZoomBounds>({});
  const pdfContainerRef = useRef<HTMLDivElement>(null);

  // Chat-related state
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatWidth, setChatWidth] = useState(400);
  const [isResizing, setIsResizing] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Tool detail modal state
  const [selectedToolMessage, setSelectedToolMessage] = useState<ChatMessage | null>(null);
  const [isToolDetailOpen, setIsToolDetailOpen] = useState(false);

  // Tool execution state tracking
  const [executingTools, setExecutingTools] = useState<Map<string, ChatMessage>>(new Map());
  console.log("executingTools", executingTools);

  const chatScrollRef = useRef<HTMLDivElement>(null);
  const resizeRef = useRef<HTMLDivElement>(null);

  // Use the new PDF chat sessions hook
  const { currentChatId, createNewSession } = usePDFChatSessions({
    apiBaseUrl: apiBaseUrl || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000",
    authToken: authToken || null,
    documentName: pdfState.documentId || pdfState.documentName || pdfState.file?.name,
  });

  // Document selection state
  const [availableDocuments, setAvailableDocuments] = useState<PDFDocument[]>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [isDocumentSelectorOpen, setIsDocumentSelectorOpen] = useState(false);

  // Memoize PDF options to prevent unnecessary reloads
  const pdfOptions = useMemo(
    () => ({
      cMapUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/cmaps/`,
      cMapPacked: true,
      standardFontDataUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/standard_fonts/`,
    }),
    []
  );

  // Handle chat resize functionality
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const newWidth = window.innerWidth - e.clientX;
      const minWidth = 300;
      const maxWidth = Math.min(800, window.innerWidth * 0.6);

      setChatWidth(Math.max(minWidth, Math.min(maxWidth, newWidth)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    if (isResizing) {
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  // Auto-scroll chat to bottom when new messages are added
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // Handle chat message submission
  const handleChatSubmit = useCallback(async () => {
    if (!chatInput.trim() || isChatLoading || !currentChatId) return;

    // Automatically switch to API mode when chat is submitted
    setPdfState(prev => ({ ...prev, controlMode: "api" }));

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: chatInput.trim(),
      timestamp: new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput("");
    setIsChatLoading(true);

    try {
      // Use the consistent chat ID
      const chatId = currentChatId;

      // Make API call to our document chat endpoint
      const response = await fetch(
        `${apiBaseUrl || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/document/chat/${chatId}/complete`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authToken && { Authorization: `Bearer ${authToken}` }),
          },
          body: JSON.stringify({
            message: userMessage.content,
            document_id: pdfState.documentId || pdfState.file?.name, // Use document ID for selected documents, filename for uploaded files
            session_id: pdfSession?.sessionId, // Include session ID for PDF viewer scoping
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body reader available");
      }

      let currentAssistantMessage: ChatMessage | null = null;
      let assistantContent = "";
      let messageIdCounter = 0;

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              // Handle different event types
              switch (data.type) {
                case "assistant":
                  if (data.tool_calls && data.tool_calls.length > 0) {
                    // Assistant message with tool calls - create message with tool calls info
                    const assistantMessage: ChatMessage = {
                      id: `assistant-${Date.now()}-${messageIdCounter++}`,
                      role: "assistant",
                      content:
                        data.content || "I'll help you with that. Let me use some tools to analyze the document.",
                      tool_calls: data.tool_calls,
                      timestamp: new Date(),
                    };

                    setChatMessages(prev => [...prev, assistantMessage]);
                    currentAssistantMessage = assistantMessage;
                    assistantContent = data.content || "";
                  } else if (data.content) {
                    // Regular assistant content - either create new message or update existing
                    if (!currentAssistantMessage) {
                      // Create new assistant message
                      const assistantMessage: ChatMessage = {
                        id: `assistant-${Date.now()}-${messageIdCounter++}`,
                        role: "assistant",
                        content: data.content,
                        timestamp: new Date(),
                      };

                      setChatMessages(prev => [...prev, assistantMessage]);
                      currentAssistantMessage = assistantMessage;
                      assistantContent = data.content;
                    } else {
                      // Update existing assistant message
                      assistantContent += data.content;
                      const messageId = currentAssistantMessage.id;
                      setChatMessages(prev =>
                        prev.map(msg => (msg.id === messageId ? { ...msg, content: assistantContent } : msg))
                      );
                    }
                  }
                  break;

                case "tool_start":
                  // Tool execution started - create loading message
                  if (data.name && data.id) {
                    const toolLoadingMessage: ChatMessage = {
                      id: `tool-loading-${data.id}`,
                      role: "tool",
                      content: "Executing...",
                      name: data.name,
                      timestamp: new Date(),
                      metadata: { status: "executing" },
                      current_frame: undefined,
                      args: data.args || {},
                    };

                    // Add to executing tools map
                    setExecutingTools(prev => new Map(prev.set(data.id, toolLoadingMessage)));

                    // Add to chat messages
                    setChatMessages(prev => [...prev, toolLoadingMessage]);
                  }
                  break;

                case "tool_complete":
                  // Tool execution completed - update the loading message
                  if (data.name && data.id && data.content) {
                    const completedToolMessage: ChatMessage = {
                      id: `tool-${data.id}`,
                      role: "tool",
                      content: data.content,
                      name: data.name,
                      timestamp: new Date(),
                      metadata: { ...data.metadata, status: "completed" },
                      current_frame: data.current_frame,
                      args: data.args || {},
                    };

                    // Remove from executing tools
                    setExecutingTools(prev => {
                      const newMap = new Map(prev);
                      newMap.delete(data.id);
                      return newMap;
                    });

                    // Update the chat message
                    setChatMessages(prev =>
                      prev.map(msg => (msg.id === `tool-loading-${data.id}` ? completedToolMessage : msg))
                    );

                    // Reset current assistant message so next assistant content creates new message
                    currentAssistantMessage = null;
                    assistantContent = "";
                  }
                  break;

                case "tool":
                  // Legacy tool execution result - create tool message with metadata
                  if (data.name && data.content) {
                    const toolMessage: ChatMessage = {
                      id: `tool-${Date.now()}-${messageIdCounter++}`,
                      role: "tool",
                      content: data.content,
                      name: data.name,
                      timestamp: new Date(),
                      metadata: data.metadata || {},
                      current_frame: data.current_frame,
                      args: data.args || {},
                    };

                    setChatMessages(prev => [...prev, toolMessage]);

                    // Reset current assistant message so next assistant content creates new message
                    currentAssistantMessage = null;
                    assistantContent = "";
                  }
                  break;

                case "done":
                  // Streaming complete
                  setIsChatLoading(false);
                  return;

                case "error":
                  // Error occurred
                  throw new Error(data.content || "Unknown error occurred");

                default:
                  // Legacy format support - handle old format for backward compatibility
                  if (data.content && !data.type) {
                    if (!currentAssistantMessage) {
                      const assistantMessage: ChatMessage = {
                        id: `assistant-${Date.now()}-${messageIdCounter++}`,
                        role: "assistant",
                        content: data.content,
                        timestamp: new Date(),
                      };

                      setChatMessages(prev => [...prev, assistantMessage]);
                      currentAssistantMessage = assistantMessage;
                      assistantContent = data.content;
                    } else {
                      assistantContent += data.content;
                      const messageId = currentAssistantMessage.id;
                      setChatMessages(prev =>
                        prev.map(msg => (msg.id === messageId ? { ...msg, content: assistantContent } : msg))
                      );
                    }
                  } else if (data.tool_call && data.result) {
                    // Legacy tool format
                    const toolMessage: ChatMessage = {
                      id: `tool-${Date.now()}-${messageIdCounter++}`,
                      role: "tool",
                      content: data.result,
                      name: data.tool_call,
                      timestamp: new Date(),
                      metadata: {},
                      current_frame: undefined,
                      args: {},
                    };

                    setChatMessages(prev => [...prev, toolMessage]);
                    currentAssistantMessage = null;
                    assistantContent = "";
                  } else if (data.done) {
                    // Legacy done format
                    setIsChatLoading(false);
                    return;
                  } else if (data.error) {
                    // Legacy error format
                    throw new Error(data.error);
                  }
                  break;
              }
            } catch (parseError) {
              // Ignore parsing errors for incomplete JSON
              console.debug("JSON parse error (likely incomplete):", parseError);
            }
          }
        }
      }

      setIsChatLoading(false);
    } catch (error) {
      console.error("Error in chat submission:", error);

      // Add error message to chat
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : "Unknown error"}. Please try again.`,
        timestamp: new Date(),
      };

      setChatMessages(prev => [...prev, errorMessage]);
      setIsChatLoading(false);
    }
  }, [
    chatInput,
    isChatLoading,
    apiBaseUrl,
    authToken,
    pdfState.file,
    pdfState.documentId,
    currentChatId,
    pdfSession?.sessionId,
  ]);

  // Load chat messages for the current chat session
  const loadChatMessages = useCallback(
    async (chatId: string, forceReload = false) => {
      if (!apiBaseUrl || !chatId) return;

      // Don't reload if we're currently loading or if we already have messages and it's not a forced reload
      if (isChatLoading || (!forceReload && chatMessages.length > 0)) return;

      try {
        const response = await fetch(`${apiBaseUrl}/document/chat/${chatId}`, {
          headers: {
            ...(authToken && { Authorization: `Bearer ${authToken}` }),
          },
        });

        if (response.ok) {
          const history: ApiChatMessage[] = await response.json();
          // Only set messages if we actually have history
          if (history && history.length > 0) {
            const formattedMessages: ChatMessage[] = history.map((msg: ApiChatMessage) => ({
              id: `${msg.role}-${msg.timestamp}`,
              role: msg.role,
              content: msg.content,
              timestamp: new Date(msg.timestamp),
              tool_calls: msg.tool_calls,
              tool_call_id: msg.tool_call_id,
              name: msg.name,
              metadata: msg.metadata,
              current_frame: msg.current_frame,
              args: msg.args,
            }));
            setChatMessages(formattedMessages);
          } else {
            // If no history exists, start with empty messages
            setChatMessages([]);
          }
        } else {
          // If no history exists, start with empty messages
          setChatMessages([]);
        }
      } catch (error) {
        console.error("Error loading chat messages:", error);
        setChatMessages([]);
      }
    },
    [apiBaseUrl, authToken, isChatLoading, chatMessages.length]
  );

  // Load chat messages when currentChatId changes
  useEffect(() => {
    if (currentChatId) {
      // Load messages for the current session
      loadChatMessages(currentChatId, true);
    } else {
      // No session, clear messages
      setChatMessages([]);
    }
  }, [currentChatId, loadChatMessages]);

  // Handle PDF load success
  const onDocumentLoadSuccess = useCallback(
    ({ numPages }: { numPages: number }) => {
      console.log("PDF document loaded successfully with", numPages, "pages");
      console.log("Current PDF state:", pdfState);
      console.log("PDF document loaded successfully with", numPages, "pages");
      console.log("Current PDF state:", pdfState);
      setPdfState(prev => ({
        ...prev,
        totalPages: numPages,
        currentPage: 1,
      }));
      setIsLoading(false);
      console.log("PDF loading state set to false");
    },
    [pdfState]
  );

  // Handle PDF load error
  const onDocumentLoadError = useCallback(
    (error: Error) => {
      console.error("Error loading PDF:", error);
      console.error("PDF.js worker src:", pdfjs.GlobalWorkerOptions.workerSrc);
      console.error("PDF file URL:", pdfState.pdfDataUrl);
      console.error("PDF file object:", pdfState.file);
      console.error("PDF state:", pdfState);

      // Additional debugging for common PDF.js issues
      if (error.message.includes("Invalid PDF")) {
        console.error("PDF appears to be corrupted or invalid");
      } else if (error.message.includes("worker")) {
        console.error("PDF.js worker issue - check network connectivity");
      } else if (error.message.includes("fetch")) {
        console.error("Network issue loading PDF - check CORS and URL accessibility");
      }

      setIsLoading(false);
    },
    [pdfState]
  );

  // PDF Controls
  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= pdfState.totalPages) {
        setPdfState(prev => ({ ...prev, currentPage: page }));
      }
    },
    [pdfState.totalPages]
  );

  const nextPage = useCallback(() => {
    goToPage(pdfState.currentPage + 1);
  }, [pdfState.currentPage, goToPage]);

  const prevPage = useCallback(() => {
    goToPage(pdfState.currentPage - 1);
  }, [pdfState.currentPage, goToPage]);

  const zoomIn = useCallback(() => {
    setPdfState(prev => ({ ...prev, scale: Math.min(prev.scale * 1.2, 3.0) }));
  }, []);

  const zoomOut = useCallback(() => {
    setPdfState(prev => ({ ...prev, scale: Math.max(prev.scale / 1.2, 0.5) }));
  }, []);

  const rotate = useCallback(() => {
    setPdfState(prev => ({ ...prev, rotation: (prev.rotation + 90) % 360 }));
  }, []);

  const resetZoom = useCallback(() => {
    setPdfState(prev => ({ ...prev, scale: 1.0 }));
  }, []);

  // Mode toggle functions
  const toggleControlMode = useCallback(() => {
    setPdfState(prev => ({
      ...prev,
      controlMode: prev.controlMode === "manual" ? "api" : "manual",
    }));
  }, []);

  // Zoom to specific bounds (0-1000 relative coordinates)
  const zoomToY = useCallback((bounds: { top: number; bottom: number }) => {
    const container = pdfContainerRef.current;
    if (!container) return;

    console.log("zoomToY called with bounds:", bounds);

    // Convert 0-1000 bounds to relative (0-1) coordinates
    const relativeTop = bounds.top / 1000;
    const relativeBottom = bounds.bottom / 1000;
    const relativeHeight = relativeBottom - relativeTop;
    const relativeCenter = (relativeTop + relativeBottom) / 2;

    console.log("Relative coords:", { relativeTop, relativeBottom, relativeHeight, relativeCenter });

    // Base PDF width that we use for scaling
    const basePdfWidth = 600;

    // Calculate scale to fit the bounds height in the container
    const containerHeight = container.clientHeight - 32; // Account for padding
    const aspectRatio = 842 / 595; // Standard A4 aspect ratio
    const basePdfHeight = basePdfWidth * aspectRatio;
    const boundsHeightPixels = relativeHeight * basePdfHeight;
    const newScale = Math.min(3.0, Math.max(0.5, (containerHeight / boundsHeightPixels) * 0.9)); // 0.9 for some padding

    console.log("Scale calculation:", { containerHeight, basePdfHeight, boundsHeightPixels, newScale });

    setPdfState(prev => ({ ...prev, scale: newScale }));

    // Find the scroll container and scroll to the bounds
    setTimeout(() => {
      const scrollContainers = [
        container.closest("[data-radix-scroll-area-viewport]"),
        container.closest(".scroll-area-viewport"),
        container.closest('[role="region"]'),
        document.querySelector("[data-radix-scroll-area-viewport]"),
      ].filter(Boolean);

      console.log("Found scroll containers:", scrollContainers.length);

      if (scrollContainers.length > 0) {
        const scrollArea = scrollContainers[0] as HTMLElement;

        // Calculate the position of the bounds within the scaled page
        const scaledPageHeight = basePdfHeight * newScale;
        const boundsTopPixels = relativeTop * scaledPageHeight;
        const boundsCenterPixels = relativeCenter * scaledPageHeight;

        // Account for padding
        const containerPadding = 16; // p-4 = 16px

        // Center the bounds in the viewport
        const targetScrollTop =
          boundsTopPixels + containerPadding - scrollArea.clientHeight / 2 + (boundsHeightPixels * newScale) / 2;

        console.log("Scroll calculation:", {
          scaledPageHeight,
          boundsTopPixels,
          boundsCenterPixels,
          targetScrollTop,
          scrollAreaHeight: scrollArea.clientHeight,
        });

        scrollArea.scrollTop = Math.max(0, targetScrollTop);
        console.log("New scroll top:", scrollArea.scrollTop);
      } else {
        console.warn("No scroll container found");
      }
    }, 200); // Give time for the scale to be applied

    setZoomBounds(prev => ({
      ...prev,
      y: relativeTop * basePdfHeight,
      height: relativeHeight * basePdfHeight,
    }));
  }, []);

  const zoomToX = useCallback((bounds: { left: number; right: number }) => {
    const container = pdfContainerRef.current;
    if (!container) return;

    console.log("zoomToX called with bounds:", bounds);

    // Convert 0-1000 bounds to relative (0-1) coordinates
    const relativeLeft = bounds.left / 1000;
    const relativeRight = bounds.right / 1000;
    const relativeWidth = relativeRight - relativeLeft;
    const relativeCenter = (relativeLeft + relativeRight) / 2;

    console.log("Relative coords:", { relativeLeft, relativeRight, relativeWidth, relativeCenter });

    // Base PDF width that we use for scaling
    const basePdfWidth = 600;

    // Calculate scale to fit the bounds width in the container
    const containerWidth = container.clientWidth - 32; // Account for padding
    const boundsWidthPixels = relativeWidth * basePdfWidth;
    const newScale = Math.min(3.0, Math.max(0.5, (containerWidth / boundsWidthPixels) * 0.9)); // 0.9 for some padding

    console.log("Scale calculation:", { containerWidth, basePdfWidth, boundsWidthPixels, newScale });

    setPdfState(prev => ({ ...prev, scale: newScale }));

    // Find the scroll container and scroll to the bounds
    setTimeout(() => {
      const scrollContainers = [
        container.closest("[data-radix-scroll-area-viewport]"),
        container.closest(".scroll-area-viewport"),
        container.closest('[role="region"]'),
        document.querySelector("[data-radix-scroll-area-viewport]"),
      ].filter(Boolean);

      console.log("Found scroll containers:", scrollContainers.length);

      if (scrollContainers.length > 0) {
        const scrollArea = scrollContainers[0] as HTMLElement;

        // Wait for the PDF to be rendered at the new scale
        setTimeout(() => {
          // Get the PDF page element after scaling
          const pdfPage = container.querySelector(".react-pdf__Page") as HTMLElement;
          if (!pdfPage) {
            console.warn("PDF page not found");
            return;
          }

          // Get the actual position of the PDF page
          const pageRect = pdfPage.getBoundingClientRect();
          const scrollAreaRect = scrollArea.getBoundingClientRect();

          // Calculate the horizontal offset of the PDF page relative to scroll area
          const pageOffsetLeft = pageRect.left - scrollAreaRect.left + scrollArea.scrollLeft;

          // Calculate the position of the bounds within the page
          const scaledPageWidth = basePdfWidth * newScale;
          const boundsLeftPixels = relativeLeft * scaledPageWidth;
          const boundsCenterPixels = relativeCenter * scaledPageWidth;

          // Calculate target scroll position to center the bounds
          const targetScrollLeft = pageOffsetLeft + boundsCenterPixels - scrollArea.clientWidth / 2;

          console.log("Scroll calculation:", {
            scaledPageWidth,
            boundsLeftPixels,
            boundsCenterPixels,
            pageOffsetLeft,
            targetScrollLeft,
            scrollAreaWidth: scrollArea.clientWidth,
            pageRect,
            scrollAreaRect,
          });

          scrollArea.scrollLeft = Math.max(0, targetScrollLeft);
          console.log("New scroll left:", scrollArea.scrollLeft);
        }, 100); // Additional delay to ensure PDF is rendered
      } else {
        console.warn("No scroll container found");
      }
    }, 200); // Give time for the scale to be applied

    setZoomBounds(prev => ({
      ...prev,
      x: relativeLeft * basePdfWidth,
      width: relativeWidth * basePdfWidth,
    }));
  }, []);

  // API endpoint handlers (these will be called by external API requests)
  useEffect(() => {
    if (pdfState.file && pdfState.controlMode === "api") {
      console.log("Registering PDF viewer controls in API mode...");
      // Register global PDF viewer control functions
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).pdfViewerControls = {
        changePage: (page: number) => {
          console.log("PDF viewer changePage called with:", page);
          goToPage(page);
        },
        zoomToY: (bounds: { top: number; bottom: number }) => {
          console.log("PDF viewer zoomToY called with:", bounds);
          zoomToY(bounds);
        },
        zoomToX: (bounds: { left: number; right: number }) => {
          console.log("PDF viewer zoomToX called with:", bounds);
          zoomToX(bounds);
        },
        getCurrentState: () => {
          console.log("PDF viewer getCurrentState called");
          return pdfState;
        },
        getMode: () => pdfState.controlMode,
      };
      console.log("PDF viewer controls registered successfully");
    } else if (pdfState.controlMode === "manual") {
      console.log("Unregistering PDF viewer controls (manual mode)");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (window as any).pdfViewerControls;
    }

    return () => {
      if (pdfState.controlMode === "api") {
        console.log("Unregistering PDF viewer controls");
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (window as any).pdfViewerControls;
      }
    };
  }, [goToPage, zoomToY, zoomToX, pdfState.file, pdfState.controlMode, pdfState]);

  // Fetch available PDF documents
  const fetchAvailableDocuments = useCallback(async () => {
    if (!apiBaseUrl) return;

    setIsLoadingDocuments(true);
    try {
      console.log("Fetching documents from:", `${apiBaseUrl}/documents`);
      const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken && { Authorization: `Bearer ${authToken}` }),
        },
        body: JSON.stringify({}), // Empty body to fetch all documents
      });

      if (response.ok) {
        const allDocuments = await response.json();
        console.log("All documents received:", allDocuments.length);

        // Filter for PDF documents only
        const pdfDocuments: PDFDocument[] = allDocuments
          .filter((doc: { content_type: string }) => doc.content_type === "application/pdf")
          .map(
            (doc: {
              external_id: string;
              filename?: string;
              folder_name?: string;
              system_metadata?: {
                created_at?: string;
                folder_name?: string;
                status?: string;
              };
            }) => ({
              id: doc.external_id,
              filename: doc.filename || `Document ${doc.external_id}`,
              download_url: "", // We'll generate this when needed
              created_at: doc.system_metadata?.created_at,
              folder_name: doc.folder_name ?? doc.system_metadata?.folder_name,
              status: doc.system_metadata?.status || "unknown",
            })
          );

        console.log("PDF documents filtered:", pdfDocuments);
        console.log(
          "PDF document IDs:",
          pdfDocuments.map(d => d.id)
        );
        setAvailableDocuments(pdfDocuments);
      } else {
        console.error("Failed to fetch documents:", response.statusText);
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
    } finally {
      setIsLoadingDocuments(false);
    }
  }, [apiBaseUrl, authToken]);

  // Load selected document from the system
  const handleDocumentSelect = useCallback(
    async (document: PDFDocument) => {
      console.log("Document selected:", document);
      setIsLoading(true);
      setIsDocumentSelectorOpen(false);

      // Reset chat state for new PDF
      setChatMessages([]);

      // Set a timeout to detect if loading takes too long
      const loadingTimeout = setTimeout(() => {
        console.warn("PDF loading is taking longer than expected (30 seconds)");
        console.warn("This might indicate a network issue or corrupted PDF");
      }, 30000);

      try {
        // First, get the download URL for this document
        const downloadUrlEndpoint = `${apiBaseUrl}/documents/${document.id}/download_url`;
        console.log("Fetching download URL from:", downloadUrlEndpoint);

        const downloadUrlResponse = await fetch(downloadUrlEndpoint, {
          headers: {
            ...(authToken && { Authorization: `Bearer ${authToken}` }),
          },
        });

        if (!downloadUrlResponse.ok) {
          console.error("Download URL request failed:", downloadUrlResponse.status, downloadUrlResponse.statusText);
          throw new Error("Failed to get download URL");
        }

        const downloadData = await downloadUrlResponse.json();
        console.log("Download URL response:", downloadData);

        let downloadUrl = downloadData.download_url;

        // Check if it's a local file URL (file://) which browsers can't access
        if (downloadUrl.startsWith("file://")) {
          console.log("Detected file:// URL, switching to direct file endpoint");
          // Use our direct file endpoint instead for local storage
          downloadUrl = `${apiBaseUrl}/documents/${document.id}/file`;
        }

        console.log("Final download URL:", downloadUrl);

        // Use the download URL to load the document
        const response = await fetch(downloadUrl, {
          headers: downloadUrl.includes("s3.amazonaws.com")
            ? {}
            : {
                ...(authToken && { Authorization: `Bearer ${authToken}` }),
              },
        });

        if (!response.ok) {
          console.error("Document download failed:", response.status, response.statusText);
          throw new Error("Failed to download document");
        }

        const blob = await response.blob();
        console.log("Document downloaded successfully, blob size:", blob.size);
        console.log("Blob type:", blob.type);

        // Validate that we have a valid PDF blob
        if (blob.size === 0) {
          throw new Error("Downloaded file is empty");
        }

        if (!blob.type.includes("pdf") && !blob.type.includes("application/octet-stream")) {
          console.warn("Blob type is not PDF:", blob.type, "- proceeding anyway");
        }
        console.log("Blob type:", blob.type);

        // Validate that we have a valid PDF blob
        if (blob.size === 0) {
          throw new Error("Downloaded file is empty");
        }

        if (!blob.type.includes("pdf") && !blob.type.includes("application/octet-stream")) {
          console.warn("Blob type is not PDF:", blob.type, "- proceeding anyway");
        }

        const file = new File([blob], document.filename, { type: "application/pdf" });

        // Create object URL for the PDF
        let pdfDataUrl: string;
        try {
          pdfDataUrl = URL.createObjectURL(blob);
          console.log("Created PDF data URL:", pdfDataUrl);
          console.log("PDF data URL length:", pdfDataUrl.length);
        } catch (urlError) {
          console.error("Failed to create object URL:", urlError);
          throw new Error("Failed to create PDF data URL");
        }

        setPdfState(prev => ({
          ...prev,
          file,
          pdfDataUrl,
          currentPage: 1,
          totalPages: 0, // Will be set in onDocumentLoadSuccess
          scale: 1.0,
          rotation: 0,
          documentName: document.filename,
          documentId: document.id,
        }));

        // Set loading to false after successfully setting up the PDF state
        // Note: onDocumentLoadSuccess will also call setIsLoading(false) when PDF.js finishes loading
        setIsLoading(false);
        clearTimeout(loadingTimeout);

        // Set loading to false after successfully setting up the PDF state
        // Note: onDocumentLoadSuccess will also call setIsLoading(false) when PDF.js finishes loading
        setIsLoading(false);
        clearTimeout(loadingTimeout);
      } catch (error) {
        console.error("Error loading selected document:", error);
        setIsLoading(false);
        clearTimeout(loadingTimeout);
        clearTimeout(loadingTimeout);
      }
    },
    [apiBaseUrl, authToken]
  );

  // Removed openDocumentSelector function since Browse Documents button was removed

  // Load initial document if provided
  useEffect(() => {
    if (initialDocumentId && !pdfState.file) {
      // Find and load the document with the given ID
      fetchAvailableDocuments().then(() => {
        // This will be handled in the next useEffect when availableDocuments is updated
      });
    }
  }, [initialDocumentId, pdfState.file, fetchAvailableDocuments]);

  // Handle loading initial document when availableDocuments is populated
  useEffect(() => {
    if (initialDocumentId && availableDocuments.length > 0 && !pdfState.file) {
      const documentToLoad = availableDocuments.find(doc => doc.id === initialDocumentId);
      if (documentToLoad) {
        handleDocumentSelect(documentToLoad);
      }
    }
  }, [initialDocumentId, availableDocuments, pdfState.file, handleDocumentSelect]);

  // Load documents when component mounts (for the document list)
  useEffect(() => {
    if (!pdfState.file) {
      fetchAvailableDocuments();
    }
  }, [fetchAvailableDocuments, pdfState.file]);

  // Debug PDF state changes
  useEffect(() => {
    console.log("PDF state changed:", pdfState);
    if (pdfState.pdfDataUrl) {
      console.log("PDF data URL is available:", pdfState.pdfDataUrl);
    }
  }, [pdfState]);

  // Test PDF.js worker accessibility
  useEffect(() => {
    const testWorker = async () => {
      try {
        console.log("Testing PDF.js worker accessibility...");
        console.log("Worker URL:", pdfjs.GlobalWorkerOptions.workerSrc);

        // Test if the worker URL is accessible
        const response = await fetch(pdfjs.GlobalWorkerOptions.workerSrc);
        if (response.ok) {
          console.log("PDF.js worker is accessible");
        } else {
          console.error("PDF.js worker is not accessible:", response.status, response.statusText);
        }
      } catch (error) {
        console.error("Error testing PDF.js worker:", error);
      }
    };

    testWorker();
  }, []);

  // Debug PDF state changes
  useEffect(() => {
    console.log("PDF state changed:", pdfState);
    if (pdfState.pdfDataUrl) {
      console.log("PDF data URL is available:", pdfState.pdfDataUrl);
    }
  }, [pdfState]);

  // Test PDF.js worker accessibility
  useEffect(() => {
    const testWorker = async () => {
      try {
        console.log("Testing PDF.js worker accessibility...");
        console.log("Worker URL:", pdfjs.GlobalWorkerOptions.workerSrc);

        // Test if the worker URL is accessible
        const response = await fetch(pdfjs.GlobalWorkerOptions.workerSrc);
        if (response.ok) {
          console.log("PDF.js worker is accessible");
        } else {
          console.error("PDF.js worker is not accessible:", response.status, response.statusText);
        }
      } catch (error) {
        console.error("Error testing PDF.js worker:", error);
      }
    };

    testWorker();
  }, []);

  if (!pdfState.file) {
    return (
      <div className="flex h-screen flex-col bg-white dark:bg-background">
        {/* Clean Header */}
        <div className="border-b bg-white p-4 dark:bg-background">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-muted-foreground" />
              <h2 className="text-lg font-medium text-slate-900 dark:text-slate-100">PDF Viewer</h2>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsChatOpen(!isChatOpen)}
                className={cn(isChatOpen && "bg-accent")}
              >
                <MessageSquare className="mr-2 h-4 w-4" />
                Chat
              </Button>
            </div>
          </div>
        </div>

        {/* Document List Area */}
        <div className="flex min-h-0 flex-1 flex-col p-8">
          <div className="mx-auto flex min-h-0 w-full max-w-4xl flex-1 flex-col">
            <div className="mb-6 text-center">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100">Select a PDF Document</h3>
              <p className="mt-2 text-sm text-muted-foreground">
                Choose from your uploaded PDF documents to view and chat about
              </p>
            </div>

            {isLoadingDocuments ? (
              <div className="flex flex-1 items-center justify-center py-12">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <div className="h-6 w-6 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent"></div>
                  <span>Loading documents...</span>
                </div>
              </div>
            ) : availableDocuments.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center py-12 text-center">
                <FileText className="mb-4 h-16 w-16 text-muted-foreground" />
                <h3 className="mb-2 text-lg font-medium">No PDF documents found</h3>
                <p className="mb-4 text-sm text-muted-foreground">
                  Upload some PDF documents in the Documents section first to view them here.
                </p>
              </div>
            ) : (
              <ScrollArea className="min-h-0 flex-1 px-4">
                <div className="grid gap-4">
                  {availableDocuments.map(doc => (
                    <Card
                      key={doc.id}
                      className="cursor-pointer p-6 transition-colors hover:bg-accent"
                      onClick={() => handleDocumentSelect(doc)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex min-w-0 flex-1 items-start gap-4">
                          <FileText className="mt-1 h-6 w-6 flex-shrink-0 text-muted-foreground" />
                          <div className="min-w-0 flex-1">
                            <h4 className="truncate text-lg font-medium">{doc.filename}</h4>
                            <div className="mt-2 flex items-center gap-6 text-sm text-muted-foreground">
                              {doc.folder_name && (
                                <span className="flex items-center gap-1">
                                  <FolderOpen className="h-4 w-4" />
                                  {doc.folder_name}
                                </span>
                              )}
                              {doc.created_at && (
                                <span className="flex items-center gap-1">
                                  <Clock className="h-4 w-4" />
                                  {new Date(doc.created_at).toLocaleDateString()}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-shrink-0 items-center gap-2">
                          <Badge
                            variant={
                              doc.status === "completed"
                                ? "default"
                                : doc.status === "processing"
                                  ? "secondary"
                                  : "destructive"
                            }
                            className="text-xs"
                          >
                            {doc.status === "completed" && <CheckCircle className="mr-1 h-3 w-3" />}
                            {doc.status === "processing" && <Clock className="mr-1 h-3 w-3" />}
                            {doc.status === "failed" && <AlertCircle className="mr-1 h-3 w-3" />}
                            {doc.status}
                          </Badge>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>
        </div>

        {/* Document Selection Dialog */}
        <Dialog open={isDocumentSelectorOpen} onOpenChange={setIsDocumentSelectorOpen}>
          <DialogContent className="max-h-[80vh] max-w-4xl overflow-hidden">
            <DialogHeader>
              <DialogTitle>Select a PDF Document</DialogTitle>
              <DialogDescription>
                Choose from your previously uploaded PDF documents to load in the viewer.
              </DialogDescription>
            </DialogHeader>

            <div className="flex-1 overflow-hidden">
              {isLoadingDocuments ? (
                <div className="flex items-center justify-center py-8">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent"></div>
                    <span>Loading documents...</span>
                  </div>
                </div>
              ) : availableDocuments.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <FileText className="mb-4 h-12 w-12 text-muted-foreground" />
                  <h3 className="mb-2 text-lg font-medium">No PDF documents found</h3>
                  <p className="mb-4 text-sm text-muted-foreground">
                    Upload some PDF documents first to see them here.
                  </p>
                  <p className="text-sm text-muted-foreground">Go to the Documents section to upload new PDF files.</p>
                </div>
              ) : (
                <ScrollArea className="h-[400px] pr-4">
                  <div className="grid gap-3">
                    {availableDocuments.map(doc => (
                      <Card
                        key={doc.id}
                        className="cursor-pointer p-4 transition-colors hover:bg-accent"
                        onClick={() => handleDocumentSelect(doc)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex min-w-0 flex-1 items-start gap-3">
                            <FileText className="mt-0.5 h-5 w-5 flex-shrink-0 text-muted-foreground" />
                            <div className="min-w-0 flex-1">
                              <h4 className="truncate font-medium">{doc.filename}</h4>
                              <div className="mt-1 flex items-center gap-4 text-sm text-muted-foreground">
                                {doc.folder_name && (
                                  <span className="flex items-center gap-1">
                                    <FolderOpen className="h-3 w-3" />
                                    {doc.folder_name}
                                  </span>
                                )}
                                {doc.created_at && (
                                  <span className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {new Date(doc.created_at).toLocaleDateString()}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="flex flex-shrink-0 items-center gap-2">
                            <Badge
                              variant={
                                doc.status === "completed"
                                  ? "default"
                                  : doc.status === "processing"
                                    ? "secondary"
                                    : "destructive"
                              }
                              className="text-xs"
                            >
                              {doc.status === "completed" && <CheckCircle className="mr-1 h-3 w-3" />}
                              {doc.status === "processing" && <Clock className="mr-1 h-3 w-3" />}
                              {doc.status === "failed" && <AlertCircle className="mr-1 h-3 w-3" />}
                              {doc.status}
                            </Badge>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* Chat Sidebar - Empty State */}
        {isChatOpen && !pdfState.file && (
          <div
            className="fixed right-0 top-0 z-50 h-full border-l bg-background shadow-2xl transition-transform duration-300"
            style={{ width: `${chatWidth}px` }}
          >
            {/* Resize Handle */}
            <div
              ref={resizeRef}
              className="absolute left-0 top-0 h-full w-1 cursor-col-resize bg-border/50 transition-colors hover:bg-border"
              onMouseDown={handleResizeStart}
            >
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transform text-muted-foreground">
                <GripVertical className="h-4 w-4 rotate-90" />
              </div>
            </div>

            <div className="flex h-full flex-col pl-2">
              {/* Chat Header */}
              <div className="flex items-center justify-between border-b p-4">
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold">PDF Chat</h3>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      console.log("+ button clicked (empty state), current chatId:", currentChatId);
                      alert("Please select a PDF document first to start a chat session!");
                    }}
                    title="New Chat Session"
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="icon" onClick={() => setIsChatOpen(false)}>
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Chat Content */}
              <div className="flex flex-1 items-center justify-center p-8">
                <div className="text-center text-muted-foreground">
                  <MessageSquare className="mx-auto mb-4 h-12 w-12" />
                  <p>Select a PDF document to start chatting about its content</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tool Detail Modal */}
        <Dialog open={isToolDetailOpen} onOpenChange={setIsToolDetailOpen}>
          <DialogContent className="max-h-[80vh] max-w-4xl overflow-hidden">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <span className="text-green-600">🔧</span>
                Tool: {selectedToolMessage?.name}
              </DialogTitle>
              <DialogDescription>Tool execution details and results</DialogDescription>
            </DialogHeader>

            {selectedToolMessage && (
              <div className="space-y-4 overflow-y-auto">
                {/* Tool Arguments */}
                {selectedToolMessage.args && Object.keys(selectedToolMessage.args).length > 0 && (
                  <div>
                    <h4 className="mb-2 font-medium">Arguments:</h4>
                    <div className="rounded-lg bg-muted p-3 text-sm">
                      <pre className="whitespace-pre-wrap">{JSON.stringify(selectedToolMessage.args, null, 2)}</pre>
                    </div>
                  </div>
                )}

                {/* Tool Result */}
                <div>
                  <h4 className="mb-2 font-medium">Result:</h4>
                  <div className="rounded-lg border border-green-200 bg-green-50 p-3 text-sm dark:border-green-800 dark:bg-green-950">
                    {selectedToolMessage.content}
                  </div>
                </div>

                {/* Metadata */}
                {selectedToolMessage.metadata && Object.keys(selectedToolMessage.metadata).length > 0 && (
                  <div>
                    <h4 className="mb-2 font-medium">Metadata:</h4>
                    <div className="rounded-lg bg-muted p-3 text-sm">
                      <pre className="whitespace-pre-wrap">{JSON.stringify(selectedToolMessage.metadata, null, 2)}</pre>
                    </div>
                  </div>
                )}

                {/* Current Frame Image */}
                {selectedToolMessage.current_frame && (
                  <div>
                    <h4 className="mb-2 font-medium">Visual Result:</h4>
                    <div className="overflow-hidden rounded-lg border">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={selectedToolMessage.current_frame}
                        alt="Tool result visualization"
                        className="h-auto max-h-96 w-full object-contain"
                      />
                    </div>
                  </div>
                )}

                {/* Timestamp */}
                <div className="text-xs text-muted-foreground">
                  Executed at: {selectedToolMessage.timestamp.toLocaleString()}
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    );
  }

  return (
    <div className="flex h-full bg-white dark:bg-black">
      {/* Main PDF Area */}
      <div
        className="flex flex-1 flex-col transition-all duration-300"
        style={{ marginRight: isChatOpen ? `${chatWidth}px` : "0px" }}
      >
        {/* Clean Header */}
        <div className="border-b border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-black">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-slate-600 dark:text-slate-400" />
              <h2 className="text-lg font-medium text-slate-900 dark:text-slate-100">
                {pdfState.documentName || pdfState.file?.name}
              </h2>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsChatOpen(!isChatOpen)}
                className={cn(isChatOpen && "bg-accent")}
              >
                <MessageSquare className="mr-2 h-4 w-4" />
                Chat
              </Button>
            </div>
          </div>
        </div>

        {/* PDF Display Area */}
        <div className="relative flex-1 overflow-hidden">
          <ScrollArea className="h-full w-full">
            <div
              ref={pdfContainerRef}
              className="flex justify-center p-4 pb-24"
              style={{
                transform: `rotate(${pdfState.rotation}deg)`,
                transformOrigin: "center center",
              }}
            >
              {pdfState.pdfDataUrl && (
                <div className="border border-slate-200 bg-white shadow-lg dark:border-slate-700 dark:bg-zinc-900">
                  <Document
                    file={pdfState.pdfDataUrl}
                    onLoadSuccess={onDocumentLoadSuccess}
                    onLoadError={onDocumentLoadError}
                    options={pdfOptions}
                    loading={
                      <div className="flex h-[800px] w-[600px] items-center justify-center bg-white p-8 text-slate-500 dark:bg-zinc-900 dark:text-slate-400">
                        <div className="text-center">
                          <FileText className="mx-auto mb-4 h-16 w-16 animate-pulse" />
                          <p>Loading PDF...</p>
                        </div>
                      </div>
                    }
                    error={
                      <div className="flex h-[800px] w-[600px] items-center justify-center bg-white p-8 text-red-500 dark:bg-zinc-900 dark:text-red-400">
                        <div className="text-center">
                          <FileText className="mx-auto mb-4 h-16 w-16" />
                          <p>Error loading PDF</p>
                          <p className="mt-2 text-sm">Please try uploading a different file</p>
                        </div>
                      </div>
                    }
                  >
                    <Page
                      pageNumber={pdfState.currentPage}
                      loading={
                        <div className="flex h-[800px] w-[600px] items-center justify-center bg-slate-100 dark:bg-zinc-800">
                          <div className="text-slate-500 dark:text-slate-400">Loading page...</div>
                        </div>
                      }
                      error={
                        <div className="flex h-[800px] w-[600px] items-center justify-center bg-slate-100 dark:bg-zinc-800">
                          <div className="text-red-500 dark:text-red-400">Error loading page</div>
                        </div>
                      }
                      width={600 * pdfState.scale}
                      renderTextLayer={true}
                      renderAnnotationLayer={true}
                    />
                  </Document>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Bottom Floating Control Bar - Fixed to viewport center */}
          <div className="pointer-events-none absolute inset-x-0 bottom-4 z-10 flex justify-center">
            <div className="pointer-events-auto flex items-center gap-4 rounded-lg border border-slate-200 bg-white px-4 py-2 shadow-lg dark:border-slate-700 dark:bg-black">
              {/* Control Mode Toggle */}
              <div
                onClick={toggleControlMode}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                  pdfState.controlMode === "manual"
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                    : "bg-zinc-700 text-white dark:bg-zinc-800"
                )}
              >
                {pdfState.controlMode === "manual" ? <User className="h-4 w-4" /> : <Cpu className="h-4 w-4" />}
                {pdfState.controlMode === "manual" ? "Manual" : "API"}
              </div>

              {/* Page Navigation */}
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={prevPage}
                  disabled={pdfState.currentPage <= 1 || pdfState.controlMode === "api"}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>

                <div className="flex items-center gap-2">
                  <Input
                    type="number"
                    value={pdfState.currentPage}
                    onChange={e => goToPage(parseInt(e.target.value) || 1)}
                    className="w-16 text-center"
                    min={1}
                    max={pdfState.totalPages}
                    disabled={pdfState.controlMode === "api"}
                  />
                  <span className="text-sm text-slate-500">of {pdfState.totalPages}</span>
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={nextPage}
                  disabled={pdfState.currentPage >= pdfState.totalPages || pdfState.controlMode === "api"}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>

              {/* Zoom Controls */}
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={zoomOut} disabled={pdfState.controlMode === "api"}>
                  <ZoomOut className="h-4 w-4" />
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetZoom}
                  disabled={pdfState.controlMode === "api"}
                  className="min-w-16"
                >
                  {Math.round(pdfState.scale * 100)}%
                </Button>

                <Button variant="outline" size="sm" onClick={zoomIn} disabled={pdfState.controlMode === "api"}>
                  <ZoomIn className="h-4 w-4" />
                </Button>
              </div>

              {/* Additional Controls */}
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={rotate} disabled={pdfState.controlMode === "api"}>
                  <RotateCw className="h-4 w-4" />
                </Button>

                <Button variant="outline" size="sm" disabled={pdfState.controlMode === "api"}>
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Sidebar */}
      {isChatOpen && pdfState.file && (
        <div
          className="fixed right-0 top-0 z-50 h-full border-l bg-background shadow-2xl transition-transform duration-300"
          style={{ width: `${chatWidth}px` }}
        >
          {/* Resize Handle */}
          <div
            ref={resizeRef}
            className="absolute left-0 top-0 h-full w-1 cursor-col-resize bg-border/50 transition-colors hover:bg-border"
            onMouseDown={handleResizeStart}
          >
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transform text-muted-foreground">
              <GripVertical className="h-4 w-4 rotate-90" />
            </div>
          </div>

          <div className="flex h-full flex-col pl-2">
            {/* Chat Header */}
            <div className="flex items-center justify-between border-b p-4">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold">PDF Chat</h3>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    console.log("+ button clicked, current chatId:", currentChatId);
                    console.log("Current chatMessages length:", chatMessages.length);
                    // Clear messages first, then create new session
                    setChatMessages([]);
                    const newChatId = createNewSession();
                    console.log("New chat session created:", newChatId);
                  }}
                  title="New Chat Session"
                >
                  <Plus className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => setIsChatOpen(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-hidden">
              <ScrollArea className="h-full" ref={chatScrollRef}>
                <div className="space-y-4 p-4">
                  {chatMessages.length === 0 ? (
                    <div className="mt-8 text-center text-muted-foreground">
                      <MessageSquare className="mx-auto mb-4 h-12 w-12" />
                      <p>Ask questions about the PDF content</p>
                    </div>
                  ) : (
                    chatMessages.map(message => (
                      <div key={message.id} className="space-y-4">
                        {message.role === "user" ? (
                          <div className="w-full">
                            <div className="w-full rounded-lg border border-border/50 bg-muted p-3 text-sm">
                              {message.content}
                            </div>
                          </div>
                        ) : message.role === "system" ? (
                          <div className="w-full">
                            <div className="w-full rounded-lg border border-slate-200 bg-slate-100 p-3 text-sm text-slate-700 dark:border-slate-700 dark:bg-zinc-900 dark:text-slate-300">
                              {message.content}
                            </div>
                          </div>
                        ) : message.role === "tool" ? (
                          <div className="w-full">
                            <div
                              className={`w-full rounded-lg border p-2 text-xs transition-colors ${
                                message.metadata?.status === "executing" || message.content === "Executing..."
                                  ? "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200"
                                  : "cursor-pointer border-green-200 bg-green-50 text-green-800 hover:bg-green-100 dark:border-green-800 dark:bg-green-950 dark:text-green-200 dark:hover:bg-green-900"
                              }`}
                              onClick={() => {
                                if (message.metadata?.status !== "executing" && message.content !== "Executing...") {
                                  setSelectedToolMessage(message);
                                  setIsToolDetailOpen(true);
                                }
                              }}
                              title={
                                message.metadata?.status === "executing" || message.content === "Executing..."
                                  ? "Tool is executing..."
                                  : "Click to view tool details"
                              }
                            >
                              <div className="flex items-center gap-2">
                                <span
                                  className={
                                    message.metadata?.status === "executing" || message.content === "Executing..."
                                      ? "text-amber-600 dark:text-amber-400"
                                      : "text-green-600 dark:text-green-400"
                                  }
                                >
                                  🔧
                                </span>
                                <div className="min-w-0 flex-1">
                                  <span className="font-medium">{message.name}</span>
                                  {message.metadata?.status === "executing" || message.content === "Executing..." ? (
                                    <div className="ml-2 inline-block">
                                      <div className="relative overflow-hidden">
                                        <span className="text-amber-600 dark:text-amber-400">⚡ Executing</span>
                                        <div className="absolute inset-0 -skew-x-12 animate-shimmer bg-gradient-to-r from-transparent via-white/30 to-transparent"></div>
                                      </div>
                                    </div>
                                  ) : (
                                    <span className="ml-1 text-green-600 dark:text-green-400">✓</span>
                                  )}
                                </div>
                                {message.current_frame &&
                                  message.metadata?.status !== "executing" &&
                                  message.content !== "Executing..." && (
                                    <span className="text-green-600 dark:text-green-400">🖼️</span>
                                  )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="w-full text-sm">
                            {/* Show assistant content if present */}
                            {message.content && (
                              <div className="prose prose-sm dark:prose-invert max-w-none text-sm">
                                <ReactMarkdown
                                  components={{
                                    p: ({ children }) => (
                                      <p className="mb-4 text-sm leading-relaxed last:mb-0">{children}</p>
                                    ),
                                    strong: ({ children }) => (
                                      <strong className="text-sm font-semibold">{children}</strong>
                                    ),
                                    ul: ({ children }) => (
                                      <ul className="mb-4 list-disc space-y-1 pl-6 text-sm">{children}</ul>
                                    ),
                                    ol: ({ children }) => (
                                      <ol className="mb-4 list-decimal space-y-1 pl-6 text-sm">{children}</ol>
                                    ),
                                    li: ({ children }) => <li className="text-sm leading-relaxed">{children}</li>,
                                    h1: ({ children }) => <h1 className="mb-3 text-base font-semibold">{children}</h1>,
                                    h2: ({ children }) => <h2 className="mb-2 text-sm font-semibold">{children}</h2>,
                                    h3: ({ children }) => <h3 className="mb-2 text-sm font-semibold">{children}</h3>,
                                    code: ({ children }) => (
                                      <code className="rounded bg-muted px-1 py-0.5 text-xs">{children}</code>
                                    ),
                                  }}
                                >
                                  {message.content}
                                </ReactMarkdown>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  )}

                  {/* Loading Message */}
                  {isChatLoading && (
                    <div className="w-full">
                      <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent"></div>
                        <span>Thinking...</span>
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>

            {/* Chat Input */}
            <div className="border-t p-4">
              <div className="relative">
                <Textarea
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  placeholder={
                    !pdfState.file
                      ? "Load a PDF to start chatting..."
                      : !currentChatId
                        ? "Loading chat..."
                        : "Ask a question about the PDF..."
                  }
                  disabled={!pdfState.file || !currentChatId || isChatLoading}
                  className="max-h-[120px] min-h-[40px] resize-none pr-12"
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (!isChatLoading) {
                        handleChatSubmit();
                      }
                    }
                  }}
                />
                <Button
                  size="icon"
                  onClick={handleChatSubmit}
                  disabled={!chatInput.trim() || isChatLoading || !pdfState.file || !currentChatId}
                  className="absolute bottom-2 right-2 h-8 w-8"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
