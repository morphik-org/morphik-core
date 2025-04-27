"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import {
  AlertCircle,
  Share2,
  Plus,
  Network,
  Tag,
  Link,
  ArrowLeft
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';

// Dynamically import ForceGraphComponent to avoid SSR issues
const ForceGraphComponent = dynamic(() => import('@/components/ForceGraphComponent'), { ssr: false });

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

interface GraphSectionProps {
  apiBaseUrl: string;
  onSelectGraph?: (graphName: string | undefined) => void;
  authToken?: string | null;
}

// Map entity types to colors
const entityTypeColors: Record<string, string> = {
  'person': '#4f46e5',   // Indigo
  'organization': '#06b6d4', // Cyan
  'location': '#10b981',  // Emerald
  'date': '#f59e0b',     // Amber
  'concept': '#8b5cf6',   // Violet
  'event': '#ec4899',    // Pink
  'product': '#ef4444',   // Red
  'default': '#6b7280'   // Gray
};

const GraphSection: React.FC<GraphSectionProps> = ({ apiBaseUrl, onSelectGraph, authToken }) => {
  // Create auth headers for API requests if auth token is available
  const createHeaders = useCallback((contentType?: string): HeadersInit => {
    const headers: HeadersInit = {};

    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }

    if (contentType) {
      headers['Content-Type'] = contentType;
    }

    return headers;
  }, [authToken]);
  // State variables
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const [selectedGraph, setSelectedGraph] = useState<Graph | null>(null);
  const [graphName, setGraphName] = useState('');
  const [graphDocuments, setGraphDocuments] = useState<string[]>([]);
  const [graphFilters, setGraphFilters] = useState('{}');
  const [additionalDocuments, setAdditionalDocuments] = useState<string[]>([]);
  const [additionalFilters, setAdditionalFilters] = useState('{}');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('list'); // 'list', 'details', 'update', 'visualize' (no longer a tab, but a state)
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showNodeLabels, setShowNodeLabels] = useState(true);
  const [showLinkLabels, setShowLinkLabels] = useState(true);
  const [showVisualization, setShowVisualization] = useState(false);
  const [graphDimensions, setGraphDimensions] = useState({ width: 0, height: 0 });


  // Refs for graph visualization
  const graphContainerRef = useRef<HTMLDivElement>(null);
  // Removed graphInstance ref as it's not needed with the dynamic component

  // Prepare data for force-graph
  const prepareGraphData = useCallback((graph: Graph | null) => {
    if (!graph) return { nodes: [], links: [] };

    const nodes = graph.entities.map(entity => ({
      id: entity.id,
      label: entity.label,
      type: entity.type,
      properties: entity.properties,
      color: entityTypeColors[entity.type.toLowerCase()] || entityTypeColors.default
    }));

    // Create a Set of all entity IDs for faster lookups
    const nodeIdSet = new Set(graph.entities.map(entity => entity.id));

    // Filter relationships to only include those where both source and target nodes exist
    const links = graph.relationships
      .filter(rel => nodeIdSet.has(rel.source_id) && nodeIdSet.has(rel.target_id))
      .map(rel => ({
        source: rel.source_id,
        target: rel.target_id,
        type: rel.type
      }));

    return { nodes, links };
  }, []);

  // Removed initializeGraph function as it's no longer needed

  // Observe graph container size changes
  useEffect(() => {
    if (!showVisualization || !graphContainerRef.current) return;

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        setGraphDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    });

    resizeObserver.observe(graphContainerRef.current);

    // Set initial size
    setGraphDimensions({
      width: graphContainerRef.current.clientWidth,
      height: graphContainerRef.current.clientHeight
    });

    const currentGraphContainer = graphContainerRef.current; // Store ref value
    return () => {
      if (currentGraphContainer) { // Use stored value in cleanup
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
      const response = await fetch(`${apiBaseUrl}/graphs`, { headers });

      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.statusText}`);
      }

      const data = await response.json();
      setGraphs(data);
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error fetching graphs: ${error.message}`);
      console.error('Error fetching graphs:', err);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, createHeaders]);

  // Fetch graphs on component mount
  useEffect(() => {
    fetchGraphs();
  }, [fetchGraphs]);

  // Fetch a specific graph
  const fetchGraph = async (graphName: string) => {
    try {
      setLoading(true);
      setError(null); // Clear previous errors
      const headers = createHeaders();
      const response = await fetch(
        `${apiBaseUrl}/graph/${encodeURIComponent(graphName)}`,
        { headers }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch graph: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedGraph(data);
      setActiveTab('details'); // Set tab to details view

      // Call the callback if provided
      if (onSelectGraph) {
        onSelectGraph(graphName);
      }

      return data;
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error fetching graph: ${error.message}`);
      console.error('Error fetching graph:', err);
      setSelectedGraph(null); // Reset selected graph on error
      setActiveTab('list'); // Go back to list view on error
      if (onSelectGraph) {
        onSelectGraph(undefined);
      }
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Handle graph click
  const handleGraphClick = (graph: Graph) => {
    fetchGraph(graph.name);
  };

  // Create a new graph
  const handleCreateGraph = async () => {
    if (!graphName.trim()) {
      setError('Please enter a graph name');
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
        throw new Error('Invalid JSON in filters field');
      }

      const headers = createHeaders('application/json');
      const response = await fetch(`${apiBaseUrl}/graph/create`, {
        method: 'POST',
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
      setActiveTab('details'); // Switch to details tab after creation

      // Refresh the graphs list
      await fetchGraphs();

      // Reset form
      setGraphName('');
      setGraphDocuments([]);
      setGraphFilters('{}');

      // Close dialog
      setShowCreateDialog(false);

    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error creating graph: ${error.message}`);
      console.error('Error creating graph:', err);
      // Keep the dialog open on error so user can fix it
    } finally {
      setLoading(false);
    }
  };

  // Update an existing graph
  const handleUpdateGraph = async () => {
    if (!selectedGraph) {
      setError('No graph selected for update');
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
        throw new Error('Invalid JSON in additional filters field');
      }

      const headers = createHeaders('application/json');
      const response = await fetch(`${apiBaseUrl}/graph/${encodeURIComponent(selectedGraph.name)}/update`, {
        method: 'POST',
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
      setSelectedGraph(data); // Update the selected graph data

      // Refresh the graphs list
      await fetchGraphs();

      // Reset form
      setAdditionalDocuments([]);
      setAdditionalFilters('{}');

      // Switch back to details tab
      setActiveTab('details');

    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error updating graph: ${error.message}`);
      console.error('Error updating graph:', err);
      // Keep the update form visible on error
    } finally {
      setLoading(false);
    }
  };

  // Removed useEffect that depended on initializeGraph

  // Conditional rendering based on visualization state
  if (showVisualization && selectedGraph) {
    return (
      <div className="fixed inset-0 bg-background z-50 flex flex-col">
        {/* Visualization header */}
        <div className="flex justify-between items-center p-4 border-b">
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
              <Network className="h-6 w-6 mr-2 text-primary" />
              <h2 className="font-medium text-lg">
                {selectedGraph.name} Visualization
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Tag className="h-4 w-4" />
              <Label htmlFor="show-node-labels" className="text-sm cursor-pointer">Nodes</Label>
              <Switch
                id="show-node-labels"
                checked={showNodeLabels}
                onCheckedChange={setShowNodeLabels}
              />
            </div>
            <div className="flex items-center gap-2">
              <Link className="h-4 w-4" />
              <Label htmlFor="show-link-labels" className="text-sm cursor-pointer">Relationships</Label>
              <Switch
                id="show-link-labels"
                checked={showLinkLabels}
                onCheckedChange={setShowLinkLabels}
              />
            </div>
          </div>
        </div>

        {/* Graph visualization container */}
        <div ref={graphContainerRef} className="flex-1 relative">
          {graphDimensions.width > 0 && graphDimensions.height > 0 && (
            <ForceGraphComponent
              data={prepareGraphData(selectedGraph)}
              width={graphDimensions.width}
              height={graphDimensions.height}
              showNodeLabels={showNodeLabels}
              showLinkLabels={showLinkLabels}
            />
          )}
        </div>
      </div>
    );
  }

  // Default view (List or Details/Update)
  return (
    <div className="flex-1 flex flex-col h-full p-4">
      <div className="flex-1 flex flex-col">
        {/* Graph List View */}
        {activeTab === 'list' && (
          <div className="mb-6">
            <div className="flex justify-end items-center mb-4"> {/* Removed justify-between and empty div */}
              <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Plus className="h-4 w-4 mr-2" /> New Graph
                  </Button>
                </DialogTrigger>
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
                        onChange={(e) => setGraphName(e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">
                        Give your graph a descriptive name that helps you identify its purpose.
                      </p>
                    </div>

                    <div className="border-t pt-4">
                      <h3 className="text-md font-medium mb-3">Document Selection</h3>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="graph-documents">Document IDs (Optional)</Label>
                          <Textarea
                            id="graph-documents"
                            placeholder="Enter document IDs separated by commas"
                            value={graphDocuments.join(', ')}
                            onChange={(e) => setGraphDocuments(e.target.value.split(',').map(id => id.trim()).filter(id => id))}
                            className="min-h-[80px]"
                          />
                          <p className="text-xs text-muted-foreground">
                            Specify document IDs to include in the graph, or leave empty and use filters below.
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="graph-filters">Metadata Filters (Optional)</Label>
                          <Textarea
                            id="graph-filters"
                            placeholder='{"category": "research", "author": "Jane Doe"}'
                            value={graphFilters}
                            onChange={(e) => setGraphFilters(e.target.value)}
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
                    <Button variant="outline" onClick={() => {
                      setShowCreateDialog(false);
                      setError(null); // Clear error when cancelling
                      // Reset form fields on cancel
                      setGraphName('');
                      setGraphDocuments([]);
                      setGraphFilters('{}');
                     }}>
                      Cancel
                    </Button>
                    <Button
                      onClick={handleCreateGraph} // Removed setShowCreateDialog(false) here, handleCreateGraph does it on success
                      disabled={!graphName || loading}
                    >
                      {loading ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      ) : null}
                      Create Graph
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>

            {loading ? (
              <div className="flex justify-center items-center p-8">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
              </div>
            ) : graphs.length === 0 ? (
              <div className="text-center p-8 border-2 border-dashed rounded-lg mt-4">
                <Network className="mx-auto h-12 w-12 mb-3 text-muted-foreground" />
                <p className="text-muted-foreground mb-3">No graphs available.</p>
                <Button onClick={() => setShowCreateDialog(true)} variant="default">
                  <Plus className="mr-2 h-4 w-4" />
                  Create Your First Graph
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4 py-2">
                {graphs.map((graph) => (
                  <div
                    key={graph.id}
                    className="cursor-pointer group flex flex-col items-center p-2 border border-transparent rounded-md hover:border-primary/20 hover:bg-primary/5 transition-all"
                    onClick={() => handleGraphClick(graph)}
                  >
                    <div className="mb-2 group-hover:scale-110 transition-transform">
                      <Network className="h-12 w-12 text-primary/80 group-hover:text-primary" />
                    </div>
                    <span className="text-sm font-medium text-center truncate w-full max-w-[120px] group-hover:text-primary transition-colors">
                      {graph.name}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Graph Details View */}
        {activeTab === 'details' && selectedGraph && (
          <div className="flex flex-col space-y-4">
            {/* Header with back button */}
            <div className="flex justify-between items-center py-2 mb-2">
              <div className="flex items-center gap-4">
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full hover:bg-muted/50"
                  onClick={() => {
                    setSelectedGraph(null);
                    setActiveTab('list');
                    if (onSelectGraph) {
                      onSelectGraph(undefined);
                    }
                  }}
                >
                  <ArrowLeft size={18} />
                </Button>
                <div className="flex items-center">
                  <Network className="h-8 w-8 mr-3 text-primary" />
                  <h2 className="font-medium text-xl">
                    {selectedGraph.name}
                  </h2>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => setActiveTab('update')}
                  className="flex items-center"
                >
                  <Plus className="mr-1 h-4 w-4" />
                  Update Graph
                </Button>
                <Button
                  onClick={() => setShowVisualization(true)}
                  className="flex items-center"
                >
                  <Share2 className="mr-1 h-4 w-4" />
                  Visualize
                </Button>
              </div>
            </div>

            {/* Graph details cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
               <div className="bg-muted/50 p-4 rounded-lg">
                <h4 className="font-medium mb-1 text-sm text-muted-foreground">Documents</h4>
                <div className="text-2xl font-bold">{selectedGraph.document_ids.length}</div>
              </div>

              <div className="bg-muted/50 p-4 rounded-lg">
                <h4 className="font-medium mb-1 text-sm text-muted-foreground">Entities</h4>
                <div className="text-2xl font-bold">{selectedGraph.entities.length}</div>
              </div>

              <div className="bg-muted/50 p-4 rounded-lg">
                <h4 className="font-medium mb-1 text-sm text-muted-foreground">Relationships</h4>
                <div className="text-2xl font-bold">{selectedGraph.relationships.length}</div>
              </div>

              <div className="bg-muted/50 p-4 rounded-lg">
                <h4 className="font-medium mb-1 text-sm text-muted-foreground">Created</h4>
                <div className="text-xl font-semibold">{new Date(selectedGraph.created_at).toLocaleDateString()}</div>
                <div className="text-xs text-muted-foreground">
                  {new Date(selectedGraph.created_at).toLocaleTimeString()}
                </div>
              </div>
            </div>

            {/* Entity and Relationship Type summaries */}
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2 text-base">Entity Types</h4>
                <div className="bg-muted/30 p-3 rounded-md border max-h-60 overflow-y-auto">
                  {Object.entries(
                    selectedGraph.entities.reduce((acc, entity) => {
                      acc[entity.type] = (acc[entity.type] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).sort(([, countA], [, countB]) => countB - countA) // Sort by count descending
                   .map(([type, count]) => (
                    <div key={type} className="flex justify-between items-center mb-2 text-sm">
                      <div className="flex items-center">
                        <div
                          className="w-3 h-3 rounded-full mr-2 flex-shrink-0"
                          style={{ backgroundColor: entityTypeColors[type.toLowerCase()] || entityTypeColors.default }}
                        ></div>
                        <span className="truncate" title={type}>{type}</span>
                      </div>
                      <Badge variant="secondary" className="ml-2 flex-shrink-0">{count}</Badge>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                 <h4 className="font-medium mb-2 text-base">Relationship Types</h4>
                <div className="bg-muted/30 p-3 rounded-md border max-h-60 overflow-y-auto">
                  {Object.entries(
                    selectedGraph.relationships.reduce((acc, rel) => {
                      acc[rel.type] = (acc[rel.type] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).sort(([, countA], [, countB]) => countB - countA) // Sort by count descending
                   .map(([type, count]) => (
                    <div key={type} className="flex justify-between items-center mb-2 text-sm">
                      <span className="truncate" title={type}>{type}</span>
                      <Badge variant="secondary" className="ml-2 flex-shrink-0">{count}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Update Graph View */}
        {activeTab === 'update' && selectedGraph && (
          <div className="flex flex-col space-y-4">
             {/* Header with back button */}
             <div className="flex justify-between items-center py-2 mb-2">
               <div className="flex items-center gap-4">
                 <Button
                   variant="ghost"
                   size="icon"
                   className="rounded-full hover:bg-muted/50"
                   onClick={() => setActiveTab('details')} // Go back to details
                 >
                   <ArrowLeft size={18} />
                 </Button>
                 <div className="flex items-center">
                   <Network className="h-8 w-8 mr-3 text-primary" />
                   <h2 className="font-medium text-xl">
                     Update: {selectedGraph.name}
                   </h2>
                 </div>
               </div>
               {/* No buttons needed on the right side for update view */}
             </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center text-lg"> {/* Reduced title size */}
                  {/* <Network className="mr-2 h-5 w-5" />  Removed icon from title */}
                  Add More Data to Graph
                </CardTitle>
                <CardDescription>
                  Expand your knowledge graph by adding new documents based on their IDs or metadata filters.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6"> {/* Increased spacing */}
                  <div className="bg-muted/50 p-4 rounded-lg border">
                    <h4 className="font-medium mb-2 text-sm text-muted-foreground">Current Graph Summary</h4>
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
                      <Label htmlFor="additional-documents">Additional Document IDs</Label>
                      <Textarea
                        id="additional-documents"
                        placeholder="Enter document IDs separated by commas"
                        value={additionalDocuments.join(', ')}
                        onChange={(e) => setAdditionalDocuments(e.target.value.split(',').map(id => id.trim()).filter(id => id))}
                        className="min-h-[80px]"
                      />
                      <p className="text-xs text-muted-foreground">
                        Specify additional document IDs to include in the graph.
                      </p>
                    </div>

                    <div className="relative flex items-center">
                      <div className="flex-grow border-t border-muted"></div>
                      <span className="flex-shrink mx-4 text-xs text-muted-foreground uppercase">Or</span>
                      <div className="flex-grow border-t border-muted"></div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="additional-filters">Additional Metadata Filters</Label>
                      <Textarea
                        id="additional-filters"
                        placeholder='{"category": "updates"}'
                        value={additionalFilters}
                        onChange={(e) => setAdditionalFilters(e.target.value)}
                        className="min-h-[80px] font-mono"
                      />
                      <p className="text-xs text-muted-foreground">
                        Use a JSON object with metadata filters to select additional documents.
                      </p>
                    </div>
                  </div>

                  <Button
                    onClick={handleUpdateGraph}
                    disabled={loading || (additionalDocuments.length === 0 && additionalFilters === '{}')} // Disable if no input
                    className="w-full"
                  >
                    {loading ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
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
    </div>
  );
};

export default GraphSection;
