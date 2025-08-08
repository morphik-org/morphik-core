"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { AlertCircle, Share2, Plus, Network, Tag, Link, ArrowLeft, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { showAlert } from "@/components/ui/alert-system";
import { MultiSelect } from "@/components/ui/multi-select";
import DeleteConfirmationModal from "@/components/documents/DeleteConfirmationModal";
import { useHeader } from "@/contexts/header-context";

// Dynamically import ForceGraphComponent to avoid SSR issues
const ForceGraphComponent = dynamic(() => import("@/components/ForceGraphComponent"), {
  ssr: false,
});

// Import the NodeDetailsSidebar component
const NodeDetailsSidebar = dynamic(() => import("@/components/NodeDetailsSidebar"), {
  ssr: false,
});

// Define interfaces
interface Graph {
  id: string;
  name: string;
  entities: Entity[];
  relationships: Relationship[];
  metadata: Record<string, unknown>;
  document_ids: string[];
  filters?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  system_metadata?: {
    status?: string;
    workflow_id?: string;
    run_id?: string;
    node_count?: number;
    edge_count?: number;
    [key: string]: unknown;
  };
}

// interface WorkflowStatusResponse {
//   status: "running" | "completed" | "failed";
//   result?: Record<string, unknown>;
//   error?: string;
//   pipeline_stage?: string;
// }

interface GraphStatusResponse {
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  workflow_id?: string;
  run_id?: string;
  pipeline_stage?: string;
  error?: string;
  document_count?: number;
  entity_count?: number;
  relationship_count?: number;
}

interface Entity {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  chunk_sources: Record<string, number[]>;
}

interface Relationship {
  id: string;
  type: string;
  source_id: string;
  target_id: string;
}

interface NodeObject {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  color: string;
}

interface LinkObject {
  source: string;
  target: string;
  type: string;
}

interface GraphSectionProps {
  apiBaseUrl: string;
  onSelectGraph?: (graphName: string | undefined) => void;
  onGraphCreate?: (graphName: string, numDocuments: number) => void;
  onGraphUpdate?: (graphName: string, numAdditionalDocuments: number) => void;
  authToken?: string | null;
  showCreateDialog?: boolean;
  setShowCreateDialog?: (show: boolean) => void;
}

// Map entity types to colors
const entityTypeColors: Record<string, string> = {
  person: "#4f46e5", // Indigo
  organization: "#06b6d4", // Cyan
  location: "#10b981", // Emerald
  date: "#f59e0b", // Amber
  concept: "#8b5cf6", // Violet
  event: "#ec4899", // Pink
  product: "#ef4444", // Red
  default: "#6b7280", // Gray
};

const POLL_INTERVAL_MS = 2000; // 2 seconds

// Interface for document API response
interface ApiDocumentResponse {
  external_id?: string;
  id?: string;
  filename?: string;
  name?: string;
}

const GraphSection: React.FC<GraphSectionProps> = ({
  apiBaseUrl,
  onSelectGraph,
  onGraphCreate,
  onGraphUpdate,
  authToken,
  showCreateDialog: showCreateDialogProp,
  setShowCreateDialog: setShowCreateDialogProp,
}) => {
  // Create auth headers for API requests if auth token is available
  const createHeaders = useCallback(
    (contentType?: string): HeadersInit => {
      const headers: HeadersInit = {};

      if (authToken) {
        headers["Authorization"] = `Bearer ${authToken}`;
      }

      if (contentType) {
        headers["Content-Type"] = contentType;
      }

      return headers;
    },
    [authToken]
  );
  // State variables
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const [selectedGraph, setSelectedGraph] = useState<Graph | null>(null);
  const [graphName, setGraphName] = useState("");
  const [graphDocuments, setGraphDocuments] = useState<string[]>([]);
  const [graphFilters, setGraphFilters] = useState("{}");
  const [additionalDocuments, setAdditionalDocuments] = useState<string[]>([]);
  const [additionalFilters, setAdditionalFilters] = useState("{}");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Delete confirmation state
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [graphToDelete, setGraphToDelete] = useState<string | null>(null);
  const [isDeletingGraph, setIsDeletingGraph] = useState(false);
  const [activeTab, setActiveTab] = useState("list"); // 'list', 'details', 'update', 'visualize' (no longer a tab, but a state)
  const [showCreateDialogLocal, setShowCreateDialogLocal] = useState(false);
  const showCreateDialog = showCreateDialogProp !== undefined ? showCreateDialogProp : showCreateDialogLocal;
  const setShowCreateDialog = setShowCreateDialogProp || setShowCreateDialogLocal;
  const [showNodeLabels, setShowNodeLabels] = useState(true);
  const [showLinkLabels, setShowLinkLabels] = useState(true);
  const [showVisualization, setShowVisualization] = useState(false);
  const [graphDimensions, setGraphDimensions] = useState({ width: 0, height: 0 });
  const [graphData, setGraphData] = useState<{ nodes: NodeObject[]; links: LinkObject[] }>({ nodes: [], links: [] });
  const [loadingVisualization, setLoadingVisualization] = useState(false);
  const [selectedNode, setSelectedNode] = useState<NodeObject | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Document selection state
  const [documents, setDocuments] = useState<{ id: string; filename: string }[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(false);

  // Refs for graph visualization
  const graphContainerRef = useRef<HTMLDivElement>(null);
  // Removed graphInstance ref as it's not needed with the dynamic component

  // Header controls
  const { setCustomBreadcrumbs, setRightContent } = useHeader();

  // set breadcrumbs & button when component mounts
  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Knowledge Graphs" }]);

    const right = (
      <Button variant="default" size="sm" onClick={() => setShowCreateDialog(true)}>
        <Plus className="mr-2 h-4 w-4" /> New Graph
      </Button>
    );
    setRightContent(right);

    return () => {
      setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [setCustomBreadcrumbs, setRightContent, setShowCreateDialog]);

  // Fallback function for local graph data (when API fails or for local graphs)
  const prepareLocalGraphData = useCallback((graph: Graph | null) => {
    if (!graph) return { nodes: [], links: [] };

    const nodes = graph.entities.map(entity => ({
      id: entity.id,
      label: entity.label,
      type: entity.type,
      properties: entity.properties,
      color: entityTypeColors[entity.type.toLowerCase()] || entityTypeColors.default,
    }));

    // Create a Set of all entity IDs for faster lookups
    const nodeIdSet = new Set(graph.entities.map(entity => entity.id));

    // Filter relationships to only include those where both source and target nodes exist
    const links = graph.relationships
      .filter(rel => nodeIdSet.has(rel.source_id) && nodeIdSet.has(rel.target_id))
      .map(rel => ({
        source: rel.source_id,
        target: rel.target_id,
        type: rel.type,
      }));

    return { nodes, links };
  }, []);

  // Prepare data for force-graph
  const prepareGraphData = useCallback(
    async (graph: Graph | null) => {
      if (!graph) return { nodes: [], links: [] };

      try {
        // Fetch visualization data from the API
        const headers = createHeaders();
        const response = await fetch(`${apiBaseUrl}/graph/${encodeURIComponent(graph.name)}/visualization`, {
          headers,
        });

        if (!response.ok) {
          console.error(`Failed to fetch visualization data: ${response.statusText}`);
          // Fallback to local data if API fails
          return prepareLocalGraphData(graph);
        }

        const visualizationData = await response.json();
        return {
          nodes: visualizationData.nodes || [],
          links: visualizationData.links || [],
        };
      } catch (error) {
        console.error("Error fetching visualization data:", error);
        // Fallback to local data if API fails
        return prepareLocalGraphData(graph);
      }
    },
    [apiBaseUrl, createHeaders, prepareLocalGraphData]
  );

  // Load graph data when visualization is shown
  useEffect(() => {
    const loadGraphData = async () => {
      if (!showVisualization || !selectedGraph) return;

      setLoadingVisualization(true);
      try {
        const data = await prepareGraphData(selectedGraph);
        setGraphData(data);
      } catch (error) {
        console.error("Error loading graph data:", error);
        setGraphData({ nodes: [], links: [] });
      } finally {
        setLoadingVisualization(false);
      }
    };

    loadGraphData();
  }, [showVisualization, selectedGraph, prepareGraphData]);

  // Observe graph container size changes
  useEffect(() => {
    if (!showVisualization || !graphContainerRef.current) return;

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        setGraphDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    resizeObserver.observe(graphContainerRef.current);

    // Set initial size
    setGraphDimensions({
      width: graphContainerRef.current.clientWidth,
      height: graphContainerRef.current.clientHeight,
    });

    const currentGraphContainer = graphContainerRef.current; // Store ref value
    return () => {
      if (currentGraphContainer) {
        // Use stored value in cleanup
        resizeObserver.unobserve(currentGraphContainer);
      }
      resizeObserver.disconnect();
    };
  }, [showVisualization]); // Rerun when visualization becomes active/inactive

  // Fetch all graphs
  const fetchGraphs = useCallback(async () => {
    try {
      setLoading(true);
      const headers = createHeaders();
      const response = await fetch(`${apiBaseUrl}/graph`, { headers });

      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.statusText}`);
      }

      const data = await response.json();
      setGraphs(data);
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error fetching graphs: ${error.message}`);
      console.error("Error fetching graphs:", err);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, createHeaders]);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    if (!apiBaseUrl) return;

    setLoadingDocuments(true);
    try {
      console.log(`Fetching documents from: ${apiBaseUrl}/documents`);
      const headers = createHeaders("application/json");
      const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        headers,
        body: JSON.stringify({}), // Empty body to fetch all docs
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`);
      }

      const documentsData = await response.json();
      console.log("Documents data received:", documentsData);

      if (Array.isArray(documentsData)) {
        // Transform documents to the format we need (id and filename)
        const transformedDocs = documentsData
          .map((doc: ApiDocumentResponse) => {
            const id = doc.external_id || doc.id;
            if (!id) return null; // Skip documents without valid IDs

            return {
              id,
              filename: doc.filename || doc.name || `Document ${id}`,
            };
          })
          .filter((doc): doc is { id: string; filename: string } => doc !== null);

        setDocuments(transformedDocs);
      } else {
        console.error("Expected array for documents data but received:", typeof documentsData);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
    } finally {
      setLoadingDocuments(false);
    }
  }, [apiBaseUrl, createHeaders]);

  // Fetch graphs on component mount
  useEffect(() => {
    fetchGraphs();
    // Also fetch documents when component mounts
    if (authToken || apiBaseUrl.includes("localhost")) {
      console.log("GraphSection: Fetching documents with auth token:", !!authToken);
      fetchDocuments();
    }
  }, [fetchGraphs, fetchDocuments, authToken, apiBaseUrl]);

  // Fetch a specific graph
  const fetchGraph = useCallback(
    async (graphName: string) => {
      try {
        setLoading(true);
        setError(null); // Clear previous errors
        const headers = createHeaders();
        const response = await fetch(`${apiBaseUrl}/graph/${encodeURIComponent(graphName)}`, {
          headers,
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch graph: ${response.statusText}`);
        }

        const data = await response.json();
        setSelectedGraph(data);
        setActiveTab("details"); // Set tab to details view

        // Call the callback if provided
        if (onSelectGraph) {
          onSelectGraph(graphName);
        }

        return data;
      } catch (err: unknown) {
        const error = err as Error;
        setError(`Error fetching graph: ${error.message}`);
        console.error("Error fetching graph:", err);
        setSelectedGraph(null); // Reset selected graph on error
        setActiveTab("list"); // Go back to list view on error
        if (onSelectGraph) {
          onSelectGraph(undefined);
        }
        return null;
      } finally {
        setLoading(false);
      }
    },
    [apiBaseUrl, createHeaders, onSelectGraph]
  );

  // Check graph status using the new lightweight endpoint
  const checkGraphStatus = useCallback(
    async (graphName: string): Promise<GraphStatusResponse> => {
      try {
        const headers = createHeaders();
        const url = `${apiBaseUrl}/graph/${encodeURIComponent(graphName)}/status`;
        const response = await fetch(url, {
          method: "GET",
          headers,
        });

        if (!response.ok) {
          throw new Error(`Failed to check graph status: ${response.statusText}`);
        }

        return await response.json();
      } catch (err) {
        console.error("Error checking graph status:", err);
        throw err;
      }
    },
    [apiBaseUrl, createHeaders]
  );

  // // Check workflow status (legacy endpoint, kept for compatibility)
  // const checkWorkflowStatus = useCallback(
  //   async (workflowId: string, runId?: string): Promise<WorkflowStatusResponse> => {
  //     try {
  //       const headers = createHeaders();
  //       const params = new URLSearchParams();
  //       if (runId) {
  //         params.append("run_id", runId);
  //       }

  //       const url = `${apiBaseUrl}/graph/workflow/${encodeURIComponent(workflowId)}/status${params.toString() ? `?${params}` : ""}`;
  //       const response = await fetch(url, {
  //         method: "GET",
  //         headers,
  //       });

  //       if (!response.ok) {
  //         throw new Error(`Failed to check workflow status: ${response.statusText}`);
  //       }

  //       return await response.json();
  //     } catch (err) {
  //       console.error("Error checking workflow status:", err);
  //       throw err;
  //     }
  //   },
  //   [apiBaseUrl, createHeaders]
  // );

  // Handle graph click
  const handleGraphClick = async (graph: Graph) => {
    const fetchedGraph = await fetchGraph(graph.name);
    if (fetchedGraph && fetchedGraph.system_metadata?.status !== "processing") {
      // Automatically show visualization for completed graphs
      setShowVisualization(true);
    }
  };

  // Create a new graph
  const handleCreateGraph = async () => {
    if (!graphName.trim()) {
      setError("Please enter a graph name");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Parse filters
      let parsedFilters = {};
      try {
        parsedFilters = JSON.parse(graphFilters);
      } catch {
        throw new Error("Invalid JSON in filters field");
      }

      const headers = createHeaders("application/json");
      const response = await fetch(`${apiBaseUrl}/graph/create`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          name: graphName,
          filters: Object.keys(parsedFilters).length > 0 ? parsedFilters : undefined,
          documents: graphDocuments.length > 0 ? graphDocuments : undefined,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create graph: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedGraph(data);

      // Check if this is an async operation
      if (data.system_metadata?.workflow_id) {
        // Graph is processing asynchronously
        setActiveTab("details"); // Switch to details tab to show processing state
        showAlert("Your graph is being created in the background. This may take a few minutes.", {
          type: "info",
          title: "Graph Creation Started",
          duration: 5000,
        });
      } else {
        // Legacy synchronous response
        setActiveTab("details"); // Switch to details tab after creation
      }

      // Invoke callback before refresh
      onGraphCreate?.(graphName, graphDocuments.length);

      // Refresh the graphs list
      await fetchGraphs();

      // Reset form
      setGraphName("");
      setGraphDocuments([]);
      setGraphFilters("{}");

      // Close dialog
      setShowCreateDialog(false);
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error creating graph: ${error.message}`);
      console.error("Error creating graph:", err);
      // Keep the dialog open on error so user can fix it
    } finally {
      setLoading(false);
    }
  };

  // Update an existing graph
  const handleUpdateGraph = async () => {
    if (!selectedGraph) {
      setError("No graph selected for update");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Parse additional filters
      let parsedFilters = {};
      try {
        parsedFilters = JSON.parse(additionalFilters);
      } catch {
        throw new Error("Invalid JSON in additional filters field");
      }

      const headers = createHeaders("application/json");
      const response = await fetch(`${apiBaseUrl}/graph/${encodeURIComponent(selectedGraph.name)}/update`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          additional_filters: Object.keys(parsedFilters).length > 0 ? parsedFilters : undefined,
          additional_documents: additionalDocuments.length > 0 ? additionalDocuments : undefined,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to update graph: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedGraph(data);

      // Check if this is an async operation
      if (data.system_metadata?.workflow_id) {
        // Graph update is processing asynchronously
        setActiveTab("details"); // Stay on details tab to show processing state
        showAlert("Your graph is being updated in the background. This may take a few minutes.", {
          type: "info",
          title: "Graph Update Started",
          duration: 5000,
        });
      } else {
        // Legacy synchronous response
        setActiveTab("details"); // Update the selected graph data
      }

      // Invoke callback before refresh
      onGraphUpdate?.(selectedGraph.name, additionalDocuments.length);

      // Refresh the graphs list
      await fetchGraphs();

      // Reset form
      setAdditionalDocuments([]);
      setAdditionalFilters("{}");

      // Only switch back if not async
      if (!data.system_metadata?.workflow_id) {
        setActiveTab("details");
      }
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error updating graph: ${error.message}`);
      console.error("Error updating graph:", err);
      // Keep the update form visible on error
    } finally {
      setLoading(false);
    }
  };

  // Removed useEffect that depended on initializeGraph

  // Poll for processing graphs using the new lightweight status endpoint
  useEffect(() => {
    // Find graphs that are processing
    const processingGraphs = graphs.filter(g => g.system_metadata?.status === "processing");

    if (processingGraphs.length === 0) return; // No need to poll

    const id = setInterval(async () => {
      // Check status for each processing graph using the new endpoint
      const statusChecks = processingGraphs.map(async graph => {
        try {
          const result = await checkGraphStatus(graph.name);

          // If graph status has changed, refresh the graph list
          if (result.status === "completed" || result.status === "failed") {
            await fetchGraphs();
            // If this is the selected graph, refresh it too
            if (selectedGraph?.name === graph.name) {
              await fetchGraph(graph.name);
            }
          } else if (result.status === "processing" && result.pipeline_stage) {
            // Update pipeline stage without full refresh if stage has changed
            const currentStage = graph.system_metadata?.pipeline_stage;
            if (currentStage !== result.pipeline_stage) {
              setGraphs(prevGraphs =>
                prevGraphs.map(g =>
                  g.name === graph.name
                    ? { ...g, system_metadata: { ...g.system_metadata, pipeline_stage: result.pipeline_stage } }
                    : g
                )
              );
              // Update selected graph if it's the one being updated
              if (selectedGraph?.name === graph.name) {
                setSelectedGraph(prev =>
                  prev
                    ? {
                        ...prev,
                        system_metadata: { ...prev.system_metadata, pipeline_stage: result.pipeline_stage },
                      }
                    : prev
                );
              }
            }
          }
        } catch (err) {
          console.error(`Error checking graph status for ${graph.name}:`, err);
        }
      });

      await Promise.all(statusChecks);
    }, POLL_INTERVAL_MS);

    return () => clearInterval(id);
  }, [graphs, selectedGraph, fetchGraphs, fetchGraph, checkGraphStatus]);

  // Handle graph deletion
  const handleDeleteGraph = useCallback(async () => {
    if (!graphToDelete) return;

    setIsDeletingGraph(true);
    try {
      const headers = createHeaders();
      const response = await fetch(`${apiBaseUrl}/graph/${encodeURIComponent(graphToDelete)}`, {
        method: "DELETE",
        headers,
      });

      if (response.ok) {
        // Refresh graphs list
        fetchGraphs();
        // If the deleted graph was selected, clear selection
        if (selectedGraph?.name === graphToDelete) {
          setSelectedGraph(null);
          setActiveTab("list");
          if (onSelectGraph) {
            onSelectGraph(undefined);
          }
        }
        showAlert(`Successfully deleted graph "${graphToDelete}"`, {
          type: "success",
          title: "Graph deleted",
        });
      } else {
        const error = await response.text();
        showAlert(`Failed to delete graph: ${error}`, {
          type: "error",
          title: "Failed to delete graph",
        });
      }
    } catch (error) {
      console.error("Failed to delete graph:", error);
      showAlert("Failed to delete graph. Please try again.", {
        type: "error",
        title: "Error",
      });
    } finally {
      setIsDeletingGraph(false);
      setShowDeleteModal(false);
      setGraphToDelete(null);
    }
  }, [graphToDelete, apiBaseUrl, createHeaders, fetchGraphs, selectedGraph, onSelectGraph]);

  // Conditional rendering based on visualization state
  if (showVisualization && selectedGraph) {
    // Handle node click for sidebar
    const handleNodeClick = (node: NodeObject | null) => {
      setSelectedNode(node);
      setSidebarOpen(!!node);
    };

    // Handle sidebar close
    const handleSidebarClose = () => {
      setSelectedNode(null);
      setSidebarOpen(false);
    };

    return (
      <div className="fixed inset-0 z-50 flex flex-col bg-background">
        {/* Visualization header */}
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="rounded-full hover:bg-muted/50"
              onClick={() => setShowVisualization(false)}
            >
              <ArrowLeft size={18} />
            </Button>
            <div className="flex items-center">
              <Network className="mr-2 h-6 w-6 text-primary" />
              <h2 className="text-lg font-medium">{selectedGraph.name} Visualization</h2>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Tag className="h-4 w-4" />
              <Label htmlFor="show-node-labels" className="cursor-pointer text-sm">
                Nodes
              </Label>
              <Switch id="show-node-labels" checked={showNodeLabels} onCheckedChange={setShowNodeLabels} />
            </div>
            <div className="flex items-center gap-2">
              <Link className="h-4 w-4" />
              <Label htmlFor="show-link-labels" className="cursor-pointer text-sm">
                Relationships
              </Label>
              <Switch id="show-link-labels" checked={showLinkLabels} onCheckedChange={setShowLinkLabels} />
            </div>
          </div>
        </div>

        {/* Graph visualization container */}
        <div ref={graphContainerRef} className="relative flex-1">
          {loadingVisualization ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <div className="mx-auto mb-2 h-8 w-8 animate-spin rounded-full border-b-2 border-primary"></div>
                <p className="text-sm text-muted-foreground">Loading graph visualization...</p>
              </div>
            </div>
          ) : graphDimensions.width > 0 && graphDimensions.height > 0 ? (
            <ForceGraphComponent
              data={graphData}
              width={graphDimensions.width}
              height={graphDimensions.height}
              showNodeLabels={showNodeLabels}
              showLinkLabels={showLinkLabels}
              onNodeClick={handleNodeClick}
            />
          ) : null}
        </div>

        {/* Node Details Sidebar */}
        {selectedNode && <NodeDetailsSidebar node={selectedNode} onClose={handleSidebarClose} isOpen={sidebarOpen} />}
      </div>
    );
  }

  // Default view (List or Details/Update)
  return (
    <div className="flex h-full flex-1 flex-col">
      <div className="flex flex-1 flex-col">
        {/* Graph List View */}
        {activeTab === "list" && (
          <div className="mb-6">
            <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle className="flex items-center">
                    <Plus className="mr-2 h-5 w-5" />
                    Create New Knowledge Graph
                  </DialogTitle>
                  <DialogDescription>
                    Create a knowledge graph from documents in your Morphik collection to enhance your queries.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="graph-name">Graph Name</Label>
                    <Input
                      id="graph-name"
                      placeholder="Enter a unique name for your graph"
                      value={graphName}
                      onChange={e => setGraphName(e.target.value)}
                    />
                    <p className="text-sm text-muted-foreground">
                      Give your graph a descriptive name that helps you identify its purpose.
                    </p>
                  </div>

                  <div className="border-t pt-4">
                    <h3 className="text-md mb-3 font-medium">Document Selection</h3>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="graph-documents">Documents</Label>
                        <MultiSelect
                          options={[
                            { label: "All Documents", value: "__none__" },
                            ...(loadingDocuments ? [{ label: "Loading documents...", value: "loading" }] : []),
                            ...documents.map(doc => ({
                              label: doc.filename,
                              value: doc.id,
                            })),
                          ]}
                          selected={graphDocuments}
                          onChange={(value: string[]) => {
                            const filteredValues = value.filter(v => v !== "__none__");
                            setGraphDocuments(filteredValues);
                          }}
                          placeholder="Select documents for the graph"
                          className="w-full"
                        />
                        <p className="text-xs text-muted-foreground">
                          Select specific documents to include in the graph, or leave empty and use filters below.
                        </p>
                      </div>

                      <div className="relative flex items-center">
                        <div className="flex-grow border-t border-muted"></div>
                        <span className="mx-4 flex-shrink text-xs uppercase text-muted-foreground">Or</span>
                        <div className="flex-grow border-t border-muted"></div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="graph-filters">Metadata Filters (Optional)</Label>
                        <Textarea
                          id="graph-filters"
                          placeholder='{"category": "research", "author": "Jane Doe"}'
                          value={graphFilters}
                          onChange={e => setGraphFilters(e.target.value)}
                          className="min-h-[80px] font-mono"
                        />
                        <p className="text-xs text-muted-foreground">
                          JSON object with metadata filters to select documents.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setShowCreateDialog(false);
                      setError(null); // Clear error when cancelling
                      // Reset form fields on cancel
                      setGraphName("");
                      setGraphDocuments([]);
                      setGraphFilters("{}");
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleCreateGraph} // Removed setShowCreateDialog(false) here, handleCreateGraph does it on success
                    disabled={!graphName || loading}
                  >
                    {loading ? (
                      <div className="mr-2 h-4 w-4 animate-spin rounded-full border-b-2 border-white"></div>
                    ) : null}
                    Create Graph
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center p-8">
            {/* Skeleton Loader for Graph List */}
            <div className="grid w-full grid-cols-2 gap-4 py-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
              {[...Array(12)].map((_, i) => (
                <div key={i} className="flex flex-col items-center rounded-md border border-transparent p-2">
                  <Skeleton className="mb-2 h-12 w-12 rounded-md" />
                  <Skeleton className="h-4 w-20 rounded-md" />
                </div>
              ))}
            </div>
          </div>
        ) : graphs.length === 0 ? (
          <div className="mt-4 rounded-lg border-2 border-dashed p-8 text-center">
            <Network className="mx-auto mb-3 h-12 w-12 text-muted-foreground" />
            <p className="mb-3 text-muted-foreground">No graphs available.</p>
            <Button onClick={() => setShowCreateDialog(true)} variant="default">
              <Plus className="mr-2 h-4 w-4" />
              Create Your First Graph
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4 py-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {graphs.map(graph => (
              <div
                key={graph.id}
                className="group relative flex cursor-pointer flex-col items-center rounded-md border border-transparent p-2 transition-all hover:border-primary/20 hover:bg-primary/5"
                onClick={() => handleGraphClick(graph)}
              >
                {/* Delete button */}
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute -right-1 -top-1 z-10 h-6 w-6 rounded-full bg-background opacity-0 shadow-sm transition-opacity hover:bg-destructive hover:text-destructive-foreground group-hover:opacity-100"
                  onClick={e => {
                    e.stopPropagation();
                    setGraphToDelete(graph.name);
                    setShowDeleteModal(true);
                  }}
                >
                  <X className="h-3 w-3" />
                </Button>
                <div className="mb-2 transition-transform group-hover:scale-110">
                  <Network className="h-12 w-12 text-primary/80 group-hover:text-primary" />
                </div>
                <span className="w-full max-w-[120px] truncate text-center text-sm font-medium transition-colors group-hover:text-primary">
                  {graph.name}
                </span>
                {graph.system_metadata?.status === "processing" && (
                  <Badge
                    variant="secondary"
                    className="mt-1 bg-yellow-400 text-[10px] text-black opacity-90 hover:bg-yellow-400"
                    title={
                      typeof graph.system_metadata?.pipeline_stage === "string"
                        ? graph.system_metadata.pipeline_stage
                        : "Processing"
                    }
                  >
                    {typeof graph.system_metadata?.pipeline_stage === "string"
                      ? graph.system_metadata.pipeline_stage
                      : "Processing"}
                  </Badge>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Graph Details View */}
        {activeTab === "details" && selectedGraph && (
          <div className="flex flex-col space-y-4">
            {selectedGraph.system_metadata?.status === "processing" && (
              <Alert variant="default" className="mb-2">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Graph is processing</AlertTitle>
                <AlertDescription>
                  {typeof selectedGraph.system_metadata?.pipeline_stage === "string"
                    ? `Current stage: ${selectedGraph.system_metadata.pipeline_stage}`
                    : "Entities and relationships are still being extracted."}
                </AlertDescription>
              </Alert>
            )}
            {/* Header with back button */}
            <div className="mb-2 flex items-center justify-between py-2">
              <div className="flex items-center gap-4">
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full hover:bg-muted/50"
                  onClick={() => {
                    setSelectedGraph(null);
                    setActiveTab("list");
                    if (onSelectGraph) {
                      onSelectGraph(undefined);
                    }
                  }}
                >
                  <ArrowLeft size={18} />
                </Button>
                <div className="flex items-center">
                  <Network className="mr-3 h-8 w-8 text-primary" />
                  <h2 className="text-xl font-medium">{selectedGraph.name}</h2>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => setActiveTab("update")}
                  className="flex items-center"
                  disabled={selectedGraph.system_metadata?.status === "processing"}
                >
                  <Plus className="mr-1 h-4 w-4" />
                  Update Graph
                </Button>
                <Button
                  onClick={() => {
                    const nodeCount = (selectedGraph.system_metadata?.node_count as number) || 0;
                    if (nodeCount > 0) {
                      setShowVisualization(true);
                    } else {
                      showAlert("Graph is still preparing. Try again shortly.", {
                        type: "info",
                        title: typeof selectedGraph.system_metadata?.pipeline_stage === "string"
                          ? (selectedGraph.system_metadata?.pipeline_stage as string)
                          : "Preparing graph",
                        duration: 4000,
                      });
                    }
                  }}
                  className="flex items-center"
                  disabled={selectedGraph.system_metadata?.status === "processing"}
                >
                  <Share2 className="mr-1 h-4 w-4" />
                  Visualize
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setGraphToDelete(selectedGraph.name);
                    setShowDeleteModal(true);
                  }}
                  className="flex items-center text-destructive hover:text-destructive"
                >
                  <X className="mr-1 h-4 w-4" />
                  Delete
                </Button>
              </div>
            </div>

            {/* Graph details cards */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-4">
              <div className="rounded-lg bg-muted/50 p-4">
                <h4 className="mb-1 text-sm font-medium text-muted-foreground">Documents</h4>
                <div className="text-2xl font-bold">{selectedGraph.document_ids.length}</div>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <h4 className="mb-1 text-sm font-medium text-muted-foreground">Entities</h4>
                <div className="text-2xl font-bold">
                  {selectedGraph.system_metadata?.status === "processing" ? "…" : selectedGraph.entities.length}
                </div>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <h4 className="mb-1 text-sm font-medium text-muted-foreground">Relationships</h4>
                <div className="text-2xl font-bold">
                  {selectedGraph.system_metadata?.status === "processing" ? "…" : selectedGraph.relationships.length}
                </div>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <h4 className="mb-1 text-sm font-medium text-muted-foreground">Created</h4>
                <div className="text-xl font-semibold">{new Date(selectedGraph.created_at).toLocaleDateString()}</div>
                <div className="text-xs text-muted-foreground">
                  {new Date(selectedGraph.created_at).toLocaleTimeString()}
                </div>
              </div>
            </div>

            {/* Entity and Relationship Type summaries */}
            <div className="mt-4 grid grid-cols-1 gap-6 md:grid-cols-2">
              <div>
                <h4 className="mb-2 text-base font-medium">Entity Types</h4>
                <div className="max-h-60 overflow-y-auto rounded-md border bg-muted/30 p-3">
                  {Object.entries(
                    (selectedGraph.system_metadata?.status === "processing" ? [] : selectedGraph.entities).reduce(
                      (acc, entity) => {
                        acc[entity.type] = (acc[entity.type] || 0) + 1;
                        return acc;
                      },
                      {} as Record<string, number>
                    )
                  )
                    .sort(([, countA], [, countB]) => countB - countA) // Sort by count descending
                    .map(([type, count]) => (
                      <div key={type} className="mb-2 flex items-center justify-between text-sm">
                        <div className="flex items-center">
                          <div
                            className="mr-2 h-3 w-3 flex-shrink-0 rounded-full"
                            style={{
                              backgroundColor: entityTypeColors[type.toLowerCase()] || entityTypeColors.default,
                            }}
                          ></div>
                          <span className="truncate" title={type}>
                            {type}
                          </span>
                        </div>
                        <Badge variant="secondary" className="ml-2 flex-shrink-0">
                          {count}
                        </Badge>
                      </div>
                    ))}
                </div>
              </div>

              <div>
                <h4 className="mb-2 text-base font-medium">Relationship Types</h4>
                <div className="max-h-60 overflow-y-auto rounded-md border bg-muted/30 p-3">
                  {Object.entries(
                    (selectedGraph.system_metadata?.status === "processing" ? [] : selectedGraph.relationships).reduce(
                      (acc, rel) => {
                        acc[rel.type] = (acc[rel.type] || 0) + 1;
                        return acc;
                      },
                      {} as Record<string, number>
                    )
                  )
                    .sort(([, countA], [, countB]) => countB - countA) // Sort by count descending
                    .map(([type, count]) => (
                      <div key={type} className="mb-2 flex items-center justify-between text-sm">
                        <span className="truncate" title={type}>
                          {type}
                        </span>
                        <Badge variant="secondary" className="ml-2 flex-shrink-0">
                          {count}
                        </Badge>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Update Graph View */}
        {activeTab === "update" && selectedGraph && (
          <div className="flex flex-col space-y-4">
            {/* Header with back button */}
            <div className="mb-2 flex items-center justify-between py-2">
              <div className="flex items-center gap-4">
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full hover:bg-muted/50"
                  onClick={() => setActiveTab("details")} // Go back to details
                >
                  <ArrowLeft size={18} />
                </Button>
                <div className="flex items-center">
                  <Network className="mr-3 h-8 w-8 text-primary" />
                  <h2 className="text-xl font-medium">Update: {selectedGraph.name}</h2>
                </div>
              </div>
              {/* No buttons needed on the right side for update view */}
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center text-lg">
                  {" "}
                  {/* Reduced title size */}
                  {/* <Network className="mr-2 h-5 w-5" />  Removed icon from title */}
                  Add More Data to Graph
                </CardTitle>
                <CardDescription>
                  Expand your knowledge graph by adding new documents based on their IDs or metadata filters.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {" "}
                  {/* Increased spacing */}
                  <div className="rounded-lg border bg-muted/50 p-4">
                    <h4 className="mb-2 text-sm font-medium text-muted-foreground">Current Graph Summary</h4>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div>
                        <span className="font-medium">Docs:</span> {selectedGraph.document_ids.length}
                      </div>
                      <div>
                        <span className="font-medium">Entities:</span> {selectedGraph.entities.length}
                      </div>
                      <div>
                        <span className="font-medium">Rels:</span> {selectedGraph.relationships.length}
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="additional-documents">Additional Documents</Label>
                      <MultiSelect
                        options={[
                          { label: "All Documents", value: "__none__" },
                          ...(loadingDocuments ? [{ label: "Loading documents...", value: "loading" }] : []),
                          ...documents.map(doc => ({
                            label: doc.filename,
                            value: doc.id,
                          })),
                        ]}
                        selected={additionalDocuments}
                        onChange={(value: string[]) => {
                          const filteredValues = value.filter(v => v !== "__none__");
                          setAdditionalDocuments(filteredValues);
                        }}
                        placeholder="Select additional documents"
                        className="w-full"
                      />
                      <p className="text-xs text-muted-foreground">
                        Select additional documents to include in the graph.
                      </p>
                    </div>

                    <div className="relative flex items-center">
                      <div className="flex-grow border-t border-muted"></div>
                      <span className="mx-4 flex-shrink text-xs uppercase text-muted-foreground">Or</span>
                      <div className="flex-grow border-t border-muted"></div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="additional-filters">Additional Metadata Filters</Label>
                      <Textarea
                        id="additional-filters"
                        placeholder='{"category": "updates"}'
                        value={additionalFilters}
                        onChange={e => setAdditionalFilters(e.target.value)}
                        className="min-h-[80px] font-mono"
                      />
                      <p className="text-xs text-muted-foreground">
                        Use a JSON object with metadata filters to select additional documents.
                      </p>
                    </div>
                  </div>
                  <Button
                    onClick={handleUpdateGraph}
                    disabled={loading || (additionalDocuments.length === 0 && additionalFilters === "{}")}
                    className="w-full"
                  >
                    {loading ? (
                      <div className="mr-2 h-4 w-4 animate-spin rounded-full border-b-2 border-white"></div>
                    ) : (
                      <Plus className="mr-2 h-4 w-4" />
                    )}
                    Update Knowledge Graph
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={showDeleteModal}
        onClose={() => {
          setShowDeleteModal(false);
          setGraphToDelete(null);
        }}
        onConfirm={handleDeleteGraph}
        itemName={graphToDelete || undefined}
        loading={isDeletingGraph}
      />
    </div>
  );
};

export default GraphSection;
