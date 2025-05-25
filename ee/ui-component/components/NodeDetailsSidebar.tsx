"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  X,
  Info,
  Hash,
  Tag,
  BarChart3,
  FileText,
  Clock,
  User,
  GripVertical,
  Code,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface NodeObject {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  color: string;
}

interface NodeDetailsSidebarProps {
  node: NodeObject | null;
  isOpen: boolean;
  onClose: () => void;
}

const NodeDetailsSidebar: React.FC<NodeDetailsSidebarProps> = ({ node, isOpen, onClose }) => {
  const [width, setWidth] = useState(384); // Default width in pixels
  const [isResizing, setIsResizing] = useState(false);
  const [showAdditionalInfo, setShowAdditionalInfo] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const resizeRef = useRef<HTMLDivElement>(null);

  // Handle resize functionality
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const newWidth = window.innerWidth - e.clientX;
      const minWidth = 300;
      const maxWidth = Math.min(800, window.innerWidth * 0.6);

      setWidth(Math.max(minWidth, Math.min(maxWidth, newWidth)));
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

  if (!node) return null;

  const props = node.properties || {};

  // Helper function to format property values
  const formatValue = (value: unknown): string => {
    if (value === null || value === undefined) return "N/A";
    if (typeof value === "string") return value;
    if (typeof value === "number") return value.toString();
    if (typeof value === "boolean") return value ? "Yes" : "No";
    return String(value);
  };

  // Helper function to get icon for property type
  const getPropertyIcon = (key: string) => {
    switch (key.toLowerCase()) {
      case "context":
        return <FileText className="h-4 w-4" />;
      case "weight":
      case "pagerank_score":
        return <BarChart3 className="h-4 w-4" />;
      case "human_readable_id":
        return <User className="h-4 w-4" />;
      case "type":
        return <Tag className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  // Get importance score for visual indicator
  const importanceScore = props.pagerank_score ? Number(props.pagerank_score) : 0;

  return (
    <>
      {/* Sidebar */}
      <div
        className={`fixed right-0 top-0 z-50 h-full transform border-l bg-background shadow-2xl transition-transform duration-300 ease-out ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
        style={{ width: `${width}px` }}
        ref={sidebarRef}
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
          {/* Header */}
          <div className="flex items-center justify-between border-b bg-muted/30 p-6">
            <div className="flex items-center space-x-3">
              <div
                className="h-4 w-4 rounded-full shadow-sm"
                style={{ backgroundColor: node.color } as React.CSSProperties}
              />
              <div>
                <h2 className="max-w-64 truncate text-lg font-semibold text-foreground">{node.label}</h2>
                <Badge variant="secondary" className="mt-1">
                  {node.type}
                </Badge>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose} className="transition-colors hover:bg-muted/50">
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Content */}
          <ScrollArea className="flex-1 p-6">
            <div className="space-y-6">
              {/* Context - Most Important Information First */}
              {props.context && typeof props.context === "string" ? (
                <div className="space-y-3">
                  <div className="flex items-center space-x-2 text-sm font-medium text-muted-foreground">
                    <FileText className="h-4 w-4" />
                    <span>Context</span>
                  </div>
                  <div className="rounded-lg border-l-4 border-primary/30 bg-muted/30 p-4">
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">{formatValue(props.context)}</p>
                  </div>
                </div>
              ) : null}

              {/* Importance Score with Tooltip */}
              {importanceScore > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center space-x-2 text-sm font-medium text-muted-foreground">
                    <BarChart3 className="h-4 w-4" />
                    <span>Importance Score</span>
                    <div className="relative">
                      <Info
                        className="h-3 w-3 cursor-help text-muted-foreground/60 transition-colors hover:text-muted-foreground"
                        onMouseEnter={() => setShowTooltip(true)}
                        onMouseLeave={() => setShowTooltip(false)}
                      />
                      {showTooltip && (
                        <div className="absolute bottom-full left-1/2 z-10 mb-2 -translate-x-1/2 transform whitespace-nowrap rounded border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-lg">
                          Calculated via PageRank
                          <div className="absolute left-1/2 top-full h-0 w-0 -translate-x-1/2 transform border-l-2 border-r-2 border-t-2 border-transparent border-t-popover"></div>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>PageRank Score</span>
                      <span className="font-mono">{importanceScore.toFixed(6)}</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-1000 ease-out"
                        style={{ width: `${Math.min(importanceScore * 10000, 100)}%` }}
                      />
                    </div>
                    <div className="text-xs text-muted-foreground">Relative importance in the knowledge graph</div>
                  </div>
                </div>
              )}

              {/* Additional Technical Information Accordion */}
              <div className="space-y-3">
                <button
                  onClick={() => setShowAdditionalInfo(!showAdditionalInfo)}
                  className="flex w-full items-center space-x-2 text-left text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
                >
                  {showAdditionalInfo ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  <Code className="h-4 w-4" />
                  <span>Additional Information</span>
                  <span className="rounded bg-muted px-2 py-0.5 text-xs">
                    {Object.keys(props).filter(key => key !== "context").length} properties
                  </span>
                </button>

                {showAdditionalInfo && (
                  <div className="space-y-4 border-l-2 border-muted pl-6">
                    {/* Node ID */}
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2 text-sm font-medium text-muted-foreground">
                        <Hash className="h-4 w-4" />
                        <span>Node ID</span>
                      </div>
                      <div className="break-all rounded-lg bg-muted/50 p-3 font-mono text-xs">{node.id}</div>
                    </div>

                    {/* All Other Properties */}
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2 text-sm font-medium text-muted-foreground">
                        <Tag className="h-4 w-4" />
                        <span>Properties</span>
                      </div>
                      <div className="space-y-3">
                        {Object.entries(props)
                          .filter(([key]) => key !== "context") // Context is shown separately
                          .sort(([a], [b]) => a.localeCompare(b))
                          .map(([key, value]) => (
                            <div
                              key={key}
                              className="group rounded-lg bg-muted/20 p-3 transition-colors duration-200 hover:bg-muted/40"
                            >
                              <div className="flex items-start space-x-3">
                                <div className="mt-0.5 text-muted-foreground transition-colors group-hover:text-foreground">
                                  {getPropertyIcon(key)}
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="mb-1 text-sm font-medium capitalize text-foreground">
                                    {key.replace(/_/g, " ")}
                                  </div>
                                  <div className="break-words text-sm text-muted-foreground">{formatValue(value)}</div>
                                </div>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Metadata Footer */}
              <div className="border-t pt-4">
                <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>{Object.keys(props).length} properties • Click background to close</span>
                </div>
              </div>
            </div>
          </ScrollArea>
        </div>
      </div>
    </>
  );
};

export default NodeDetailsSidebar;
