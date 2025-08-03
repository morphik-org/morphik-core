"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { DocumentSelector } from "@/components/ui/document-selector";
import { Search, Code2, Eye, Clock, GripVertical, FileText, Terminal } from "lucide-react";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useTheme } from "next-themes";
import { showAlert } from "@/components/ui/alert-system";
import SearchResultCard from "./SearchResultCard";
import SearchResultCardCarousel from "./SearchResultCardCarousel";

import { SearchResult, SearchOptions, FolderSummary, GroupedSearchResponse } from "@/components/types";

interface SearchV2SectionProps {
  apiBaseUrl: string;
  authToken: string | null;
}

const defaultSearchOptions: SearchOptions = {
  filters: "{}",
  k: 5,
  min_score: 0.7,
  use_reranking: false,
  use_colpali: true,
  padding: 0,
  folder_name: undefined,
};

const SearchV2Section: React.FC<SearchV2SectionProps> = ({ apiBaseUrl, authToken }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [groupedResults, setGroupedResults] = useState<GroupedSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>(defaultSearchOptions);
  const [responseTime, setResponseTime] = useState<number | null>(null);
  const [requestPayload, setRequestPayload] = useState<any>(null);
  const [responseData, setResponseData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<"code" | "results">("code");
  const [activeResultsTab, setActiveResultsTab] = useState<"visual" | "json">("visual");
  const [leftPanelWidth, setLeftPanelWidth] = useState(380);
  const [isResizing, setIsResizing] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [selectedFolders, setSelectedFolders] = useState<string[]>([]);
  const [availableDocuments, setAvailableDocuments] = useState<any[]>([]);
  const { theme } = useTheme();

  // Update search options
  const updateSearchOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    setSearchOptions(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  // Handle panel resizing
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = Math.max(300, Math.min(800, e.clientX));
      setLeftPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  // Fetch folders and documents when auth token or API URL changes
  useEffect(() => {
    const fetchFolders = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/folders/summary`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });

        if (response.ok) {
          const folderData = await response.json();
          setFolders(folderData);
        }
      } catch (error) {
        console.error("Error fetching folders:", error);
      }
    };

    const fetchDocuments = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/documents`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });

        if (response.ok) {
          const documentsData = await response.json();
          setAvailableDocuments(documentsData);
        }
      } catch (error) {
        console.error("Error fetching documents:", error);
      }
    };

    if (authToken || apiBaseUrl.includes("localhost")) {
      fetchFolders();
      fetchDocuments();
    }
  }, [authToken, apiBaseUrl]);

  // Update filters when documents are selected (same logic as chat)
  const updateDocumentFilter = useCallback(
    (selectedDocumentIds: string[]) => {
      const currentFilters = searchOptions.filters || "{}";
      const parsedFilters = typeof currentFilters === "string" ? JSON.parse(currentFilters) : currentFilters;

      const newFilters = {
        ...parsedFilters,
        external_id: selectedDocumentIds.length > 0 ? selectedDocumentIds : undefined,
      };

      // Remove undefined values
      Object.keys(newFilters).forEach(key => newFilters[key] === undefined && delete newFilters[key]);

      updateSearchOption("filters", newFilters);
    },
    [updateSearchOption, searchOptions.filters]
  );

  // Update folder name when folders are selected
  const updateFolderFilter = useCallback(
    (selectedFolderNames: string[]) => {
      updateSearchOption(
        "folder_name",
        selectedFolderNames.length > 0 ? selectedFolderNames : undefined
      );
    },
    [updateSearchOption]
  );

  // Handle search
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      showAlert("Please enter a search query", {
        type: "error",
        duration: 3000,
      });
      return;
    }

    const currentSearchOptions: SearchOptions = {
      ...searchOptions,
      filters: searchOptions.filters || "{}",
    };

    try {
      setLoading(true);
      setSearchResults([]);
      setGroupedResults(null);
      setResponseTime(null);

      // Handle filters - convert to object if needed
      let filtersObject = {};
      if (currentSearchOptions.filters) {
        if (typeof currentSearchOptions.filters === "string") {
          filtersObject = JSON.parse(currentSearchOptions.filters);
        } else {
          filtersObject = currentSearchOptions.filters;
        }
      }

      // Prepare request payload
      const payload = {
        query: searchQuery,
        filters: filtersObject,
        folder_name: currentSearchOptions.folder_name,
        k: currentSearchOptions.k,
        min_score: currentSearchOptions.min_score,
        use_reranking: currentSearchOptions.use_reranking,
        use_colpali: currentSearchOptions.use_colpali,
        padding: currentSearchOptions.padding || 0,
      };

      setRequestPayload(payload);

      // Use grouped endpoint when padding is enabled, regular endpoint otherwise
      const shouldUseGroupedEndpoint = (currentSearchOptions.padding || 0) > 0;
      const endpoint = shouldUseGroupedEndpoint ? "/retrieve/chunks/grouped" : "/retrieve/chunks";

      const startTime = performance.now();

      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          Authorization: authToken ? `Bearer ${authToken}` : "",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const endTime = performance.now();
      setResponseTime(endTime - startTime);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `Search failed: ${response.statusText}` }));
        throw new Error(errorData.detail || `Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResponseData(data);

      if (shouldUseGroupedEndpoint) {
        setGroupedResults(data);
        setSearchResults(data.chunks);
      } else {
        setSearchResults(data);
        setGroupedResults(null);
      }

      const resultCount = shouldUseGroupedEndpoint ? data.chunks?.length || 0 : data.length || 0;
      if (resultCount === 0) {
        showAlert("No search results found for the query", {
          type: "info",
          duration: 3000,
        });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      showAlert(errorMsg, {
        type: "error",
        title: "Search Failed",
        duration: 5000,
      });
      setSearchResults([]);
      setResponseData(null);
    } finally {
      setLoading(false);
    }
  };

  const resultCount = searchResults.length;

  // Get current selected values (same pattern as chat)
  const getCurrentSelectedFolders = (): string[] => {
    const folderName = searchOptions.folder_name;
    if (!folderName) return [];
    const folders = Array.isArray(folderName) ? folderName : [folderName];
    return folders.filter(f => f !== "__none__");
  };

  const getCurrentSelectedDocuments = (): string[] => {
    const filters = searchOptions.filters || {};
    const parsedFilters = typeof filters === "string" ? JSON.parse(filters || "{}") : filters;
    const externalId = parsedFilters.external_id;
    if (!externalId) return [];
    const documents = Array.isArray(externalId) ? externalId : [externalId];
    return documents.filter(d => d !== "__none__");
  };

  // Generate request payload for API
  const generateRequestPayload = useCallback(() => {
    const currentFilters = searchOptions.filters || "{}";
    const parsedFilters = typeof currentFilters === "string" ? JSON.parse(currentFilters) : currentFilters;

    const payload: any = {
      query: searchQuery || "Enter your search query here",
      k: searchOptions.k,
      min_score: searchOptions.min_score,
      padding: searchOptions.padding || 0,
    };

    // Add optional fields only if they have values
    if (Object.keys(parsedFilters).length > 0) {
      payload.filters = parsedFilters;
    }

    if (searchOptions.use_reranking !== undefined) {
      payload.use_reranking = searchOptions.use_reranking;
    }

    if (searchOptions.use_colpali !== undefined) {
      payload.use_colpali = searchOptions.use_colpali;
    }

    const selectedFolders = getCurrentSelectedFolders();
    if (selectedFolders.length > 0) {
      payload.folder_name = selectedFolders.length === 1 ? selectedFolders[0] : selectedFolders;
    }

    return payload;
  }, [searchQuery, searchOptions, getCurrentSelectedFolders]);

  // Generate cURL command
  const generateCurlCode = useCallback(() => {
    const payload = generateRequestPayload();
    const headers = authToken ? `-H "Authorization: Bearer YOUR_TOKEN"` : '';

    return `curl -X POST "${apiBaseUrl}/retrieve/chunks" \\
  -H "Content-Type: application/json" \\
  ${headers}${headers ? ' \\' : ''}
  -d '${JSON.stringify(payload, null, 2)}'`;
  }, [apiBaseUrl, authToken, generateRequestPayload]);

  // Generate Python code
  const generatePythonCode = useCallback(() => {
    const payload = generateRequestPayload();

    return `import requests
import json

# API configuration
url = "${apiBaseUrl}/retrieve/chunks"
headers = {
    "Content-Type": "application/json",${authToken ? '\n    "Authorization": "Bearer YOUR_TOKEN",' : ''}
}

# Request payload
payload = ${JSON.stringify(payload, null, 4)}

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Handle the response
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))
else:
    print(f"Error: {response.status_code} - {response.text}")`;
  }, [apiBaseUrl, authToken, generateRequestPayload]);

  // Generate JavaScript code
  const generateJavaScriptCode = useCallback(() => {
    const payload = generateRequestPayload();

    return `// Using fetch API
const response = await fetch("${apiBaseUrl}/retrieve/chunks", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",${authToken ? '\n    "Authorization": "Bearer YOUR_TOKEN",' : ''}
  },
  body: JSON.stringify(${JSON.stringify(payload, null, 2)})
});

const data = await response.json();
console.log(data);`;
  }, [apiBaseUrl, authToken, generateRequestPayload]);

  return (
    <div className="h-[calc(100vh-theme(spacing.16))] bg-background">
      <div className="flex h-full">
        {/* Left Control Panel */}
        <div
          className="bg-background border-r flex-shrink-0"
          style={{ width: leftPanelWidth }}
        >
          <div className="h-full flex flex-col">
            {/* Header */}
            <div className="px-4 py-3 border-b">
              <h3 className="font-medium">Search</h3>
            </div>

            {/* Scrollable Content */}
            <ScrollArea className="flex-1">
              <div className="p-4 space-y-4">
                {/* Query Section */}
                <div className="space-y-3">
                  <Textarea
                    placeholder="Enter your search query..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSearch();
                      }
                    }}
                    className="min-h-20 resize-none"
                  />
                  <Button onClick={handleSearch} disabled={loading} className="w-full">
                    <Search className="mr-2 h-4 w-4" />
                    {loading ? "Searching..." : "Run"}
                  </Button>
                </div>

                {/* Document and Folder Selection */}
                <div className="space-y-2">
                  <Label className="text-xs text-muted-foreground">Documents & Folders</Label>
                  <DocumentSelector
                    documents={availableDocuments.map(doc => ({
                      id: doc.external_id || doc.id,
                      filename: doc.title || doc.filename || doc.name || `Document ${doc.external_id || doc.id}`,
                      folder_name: doc.folder_name || (doc.system_metadata?.folder_name as string),
                      content_type: doc.content_type,
                      metadata: doc.metadata,
                      system_metadata: doc.system_metadata,
                    }))}
                    folders={folders.map(folder => ({
                      name: folder.name,
                      doc_count: folder.document_count || 0,
                    }))}
                    selectedDocuments={getCurrentSelectedDocuments()}
                    selectedFolders={getCurrentSelectedFolders()}
                    onDocumentSelectionChange={(selectedDocumentIds: string[]) => {
                      updateDocumentFilter(selectedDocumentIds);
                    }}
                    onFolderSelectionChange={(selectedFolderNames: string[]) => {
                      updateFolderFilter(selectedFolderNames);
                    }}
                    loading={availableDocuments.length === 0}
                    placeholder="Select documents and folders"
                    className="w-full"
                  />
                </div>

                {/* Single-line Options */}
                <div className="space-y-3">

                  {/* Number of results */}
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-muted-foreground">Results</Label>
                    <Input
                      type="number"
                      min="1"
                      max="50"
                      value={searchOptions.k}
                      onChange={e => updateSearchOption("k", parseInt(e.target.value) || 5)}
                      className="w-20 text-sm"
                    />
                  </div>

                  {/* Min Score */}
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-muted-foreground">Min Score</Label>
                    <Input
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={searchOptions.min_score}
                      onChange={e => updateSearchOption("min_score", parseFloat(e.target.value) || 0.7)}
                      className="w-20 text-sm"
                    />
                  </div>

                  {/* Context Padding */}
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-muted-foreground">Padding</Label>
                    <Input
                      type="number"
                      min="0"
                      max="10"
                      value={searchOptions.padding || 0}
                      onChange={e => updateSearchOption("padding", parseInt(e.target.value) || 0)}
                      className="w-20 text-sm"
                    />
                  </div>

                  {/* Use Reranking */}
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-muted-foreground">Use Reranking</Label>
                    <Switch
                      checked={searchOptions.use_reranking}
                      onCheckedChange={value => updateSearchOption("use_reranking", value)}
                    />
                  </div>

                  {/* Use ColPali */}
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-muted-foreground">Use ColPali</Label>
                    <Switch
                      checked={searchOptions.use_colpali}
                      onCheckedChange={value => updateSearchOption("use_colpali", value)}
                    />
                  </div>
                </div>

                {/* JSON Filters - keeping this multi-line since it's complex */}
                <div className="space-y-2">
                  <Label className="text-sm text-muted-foreground">JSON Filters</Label>
                  <Textarea
                    placeholder="{}"
                    value={searchOptions.filters || "{}"}
                    onChange={e => updateSearchOption("filters", e.target.value)}
                    className="min-h-16 font-mono text-xs"
                  />
                </div>

                {/* API Code Preview */}
                {requestPayload && (
                  <div className="space-y-2">
                    <Label className="text-sm text-muted-foreground">API Request</Label>
                    <div className="rounded-md bg-muted/50 p-3 border">
                      <pre className="text-xs overflow-x-auto whitespace-pre-wrap">
                        <code className="text-muted-foreground">{`fetch("${apiBaseUrl}/retrieve/chunks", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"
  },
  body: JSON.stringify(${JSON.stringify(requestPayload, null, 2)})
})`}</code>
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Resizable Separator */}
        <div
          className="w-1 bg-border hover:bg-muted-foreground/30 cursor-col-resize transition-colors"
          onMouseDown={handleMouseDown}
        />

        {/* Right Output Panel */}
        <div className="flex-1 bg-background">
          <div className="h-full flex flex-col">
            {/* Compact Header */}
            <div className="px-4 py-3 border-b flex items-center justify-between">
              <div className="flex items-center gap-4">
                <h3 className="font-medium">Results ({resultCount})</h3>
                {responseTime && (
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    Response {(responseTime / 1000).toFixed(2)}s
                  </div>
                )}
              </div>
              <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "code" | "results")}>
                <TabsList className="grid w-48 grid-cols-2">
                  <TabsTrigger value="code" className="text-xs">
                    <Terminal className="h-3 w-3 mr-1" />
                    Code
                  </TabsTrigger>
                  <TabsTrigger value="results" className="text-xs">
                    <Eye className="h-3 w-3 mr-1" />
                    Results
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {/* Content */}
            <div className="flex-1 min-h-0">
              <Tabs value={activeTab} className="h-full">
                <TabsContent value="code" className="h-full mt-0">
                  <div className="h-full flex flex-col">
                    {/* Future: Code language tabs will go here */}
                    <div className="border-b px-4 py-2">
                      <div className="text-sm text-muted-foreground">cURL</div>
                    </div>

                    {/* Code content */}
                    <div className="flex-1 overflow-hidden">
                      <ScrollArea className="h-full">
                        <div className="p-4">
                          <SyntaxHighlighter
                            language="bash"
                            style={theme === 'dark' ? vscDarkPlus : vs}
                            className="rounded-lg"
                            showLineNumbers
                            wrapLines
                            customStyle={{
                              backgroundColor: theme === 'dark' ? '#1e1e1e' : '#fafafa',
                              border: theme === 'dark' ? '1px solid #333' : '1px solid #d1d5db',
                              fontSize: '13px',
                              lineHeight: '1.4',
                              color: theme === 'dark' ? '#d4d4d4' : '#374151',
                              borderRadius: '8px',
                              padding: '16px'
                            }}
                          >
                            {generateCurlCode()}
                          </SyntaxHighlighter>
                        </div>
                      </ScrollArea>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="results" className="h-full mt-0">
                  <div className="h-full flex flex-col">
                    {/* Results sub-tabs */}
                    <div className="border-b px-4 py-2">
                      <Tabs value={activeResultsTab} onValueChange={(value) => setActiveResultsTab(value as "visual" | "json")}>
                        <TabsList className="grid w-48 grid-cols-2">
                          <TabsTrigger value="visual" className="text-xs">
                            <Eye className="h-3 w-3 mr-1" />
                            Visual
                          </TabsTrigger>
                          <TabsTrigger value="json" className="text-xs">
                            <Code2 className="h-3 w-3 mr-1" />
                            JSON
                          </TabsTrigger>
                        </TabsList>
                      </Tabs>
                    </div>

                    {/* Results content */}
                    <div className="flex-1 min-h-0">
                      <Tabs value={activeResultsTab} className="h-full">
                        <TabsContent value="visual" className="h-full mt-0 p-4">
                          {searchResults.length > 0 ? (
                            <ScrollArea className="h-full">
                              <div className="space-y-4 pr-4">
                                {groupedResults?.has_padding
                                  ? groupedResults.groups.map(group => (
                                      <SearchResultCardCarousel
                                        key={`${group.main_chunk.document_id}-${group.main_chunk.chunk_number}`}
                                        group={group}
                                      />
                                    ))
                                  : searchResults.map(result => (
                                      <SearchResultCard key={`${result.document_id}-${result.chunk_number}`} result={result} />
                                    ))}
                              </div>
                            </ScrollArea>
                          ) : (
                            <div className="flex h-full items-center justify-center">
                              <div className="text-center max-w-md">
                                <Search className="mx-auto mb-4 h-12 w-12 text-muted-foreground/50" />
                                <h3 className="text-lg font-medium mb-2">No results yet</h3>
                                <p className="text-sm text-muted-foreground">
                                  Run a search to see results here
                                </p>
                              </div>
                            </div>
                          )}
                        </TabsContent>

                        <TabsContent value="json" className="h-full mt-0 p-4">
                          {responseData ? (
                            <ScrollArea className="h-full">
                              <pre className="text-xs bg-muted/30 p-4 rounded-lg overflow-x-auto border">
                                <code>{JSON.stringify(responseData, null, 2)}</code>
                              </pre>
                            </ScrollArea>
                          ) : (
                            <div className="flex h-full items-center justify-center">
                              <div className="text-center max-w-md">
                                <Code2 className="mx-auto mb-4 h-12 w-12 text-muted-foreground/50" />
                                <h3 className="text-lg font-medium mb-2">No response data</h3>
                                <p className="text-sm text-muted-foreground">
                                  Run a search to see the JSON response here
                                </p>
                              </div>
                            </div>
                          )}
                        </TabsContent>
                      </Tabs>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchV2Section;
