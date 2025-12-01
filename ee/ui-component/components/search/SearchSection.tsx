"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { Search, RotateCcw, Book, Copy, Check, Code2, Terminal, ImagePlus, X } from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";
import SearchResultCard from "./SearchResultCard";
import SearchResultCardCarousel from "./SearchResultCardCarousel";

import { SearchResult, SearchOptions, FolderSummary, GroupedSearchResponse } from "@/components/types";

interface SearchSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  onSearchSubmit?: (query: string, options: SearchOptions) => void;
}

const defaultSearchOptions: SearchOptions = {
  filters: "{}",
  k: 10,
  min_score: 0.0,
  use_reranking: false,
  use_colpali: true,
  padding: 0,
  folder_name: undefined,
};

type CodeLanguage = "python" | "javascript" | "curl";
type RightPanelTab = "code" | "output";

const SearchSection: React.FC<SearchSectionProps> = ({ apiBaseUrl, authToken, onSearchSubmit }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [groupedResults, setGroupedResults] = useState<GroupedSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>(defaultSearchOptions);
  const [copied, setCopied] = useState(false);
  const [codeLanguage, setCodeLanguage] = useState<CodeLanguage>("python");
  const [rightPanelTab, setRightPanelTab] = useState<RightPanelTab>("code");
  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [queryImagePreview, setQueryImagePreview] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const updateSearchOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    setSearchOptions(prev => ({ ...prev, [key]: value }));
  };

  // Fetch folders
  useEffect(() => {
    setSearchResults([]);
    setGroupedResults(null);

    const fetchFolders = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/folders/details`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify({
            include_document_count: true,
            include_status_counts: false,
            include_documents: false,
          }),
        });

        if (response.ok) {
          const folderResult = await response.json();
          const folderEntries = Array.isArray(folderResult?.folders) ? folderResult.folders : [];
          const mappedFolders = folderEntries.map((entry: Record<string, unknown>) => {
            const folder = (entry?.folder ?? {}) as Record<string, unknown>;
            const documentInfo = (entry?.document_info ?? {}) as Record<string, unknown>;
            const systemMetadata = (folder.system_metadata ?? {}) as Record<string, unknown>;
            const updatedAt = systemMetadata?.updated_at ?? systemMetadata?.created_at ?? undefined;
            return {
              id: folder.id as string,
              name: folder.name as string,
              description: (folder.description ?? undefined) as string | undefined,
              doc_count: (documentInfo?.document_count ??
                (Array.isArray(folder.document_ids) ? folder.document_ids.length : undefined)) as number | undefined,
              updated_at: typeof updatedAt === "string" ? updatedAt : updatedAt ? String(updatedAt) : undefined,
            };
          });
          setFolders(mappedFolders);
        }
      } catch (error) {
        console.error("Error fetching folders:", error);
      }
    };

    if (authToken || apiBaseUrl.includes("localhost")) {
      fetchFolders();
    }
  }, [authToken, apiBaseUrl]);

  const handleSearch = async () => {
    // Validate: either text query or image must be provided
    if (!searchQuery.trim() && !queryImage) {
      showAlert("Please enter a search query or upload an image", { type: "error", duration: 3000 });
      return;
    }

    // Image queries require ColPali
    if (queryImage && !searchOptions.use_colpali) {
      showAlert("Image search requires Morphik multimodal retrieval (ColPali) to be enabled", {
        type: "error",
        duration: 3000,
      });
      return;
    }

    const currentSearchOptions: SearchOptions = {
      ...searchOptions,
      filters: searchOptions.filters || "{}",
      query_image: queryImage || undefined,
    };
    onSearchSubmit?.(searchQuery, currentSearchOptions);

    try {
      setLoading(true);
      setSearchResults([]);
      setGroupedResults(null);

      let filtersObject = {};
      if (currentSearchOptions.filters) {
        if (typeof currentSearchOptions.filters === "string") {
          filtersObject = JSON.parse(currentSearchOptions.filters);
        } else {
          filtersObject = currentSearchOptions.filters;
        }
      }

      const shouldUseGroupedEndpoint = (currentSearchOptions.padding || 0) > 0;
      const endpoint = shouldUseGroupedEndpoint ? "/retrieve/chunks/grouped" : "/retrieve/chunks";

      // Switch to output tab immediately when search starts
      setRightPanelTab("output");

      // Build request body
      const requestBody: Record<string, unknown> = {
        filters: filtersObject,
        folder_name: currentSearchOptions.folder_name,
        k: currentSearchOptions.k,
        min_score: currentSearchOptions.min_score,
        use_reranking: currentSearchOptions.use_reranking,
        use_colpali: currentSearchOptions.use_colpali,
        padding: currentSearchOptions.padding || 0,
      };

      // Add either text query or image query (mutually exclusive)
      if (queryImage) {
        requestBody.query_image = queryImage;
      } else {
        requestBody.query = searchQuery;
      }

      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          Authorization: authToken ? `Bearer ${authToken}` : "",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `Search failed: ${response.statusText}` }));
        throw new Error(errorData.detail || `Search failed: ${response.statusText}`);
      }

      const data = await response.json();

      if (shouldUseGroupedEndpoint) {
        setGroupedResults(data);
        setSearchResults(data.chunks);
      } else {
        setSearchResults(data);
        setGroupedResults(null);
      }

      const resultCount = shouldUseGroupedEndpoint ? data.chunks?.length || 0 : data.length || 0;
      if (resultCount === 0) {
        showAlert("No search results found for the query", { type: "info", duration: 3000 });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      showAlert(errorMsg, { type: "error", title: "Search Failed", duration: 5000 });
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSearchQuery("");
    setSearchResults([]);
    setGroupedResults(null);
    setQueryImage(null);
    setQueryImagePreview(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("image/")) {
      showAlert("Please select an image file", { type: "error", duration: 3000 });
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      showAlert("Image must be less than 10MB", { type: "error", duration: 3000 });
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      setQueryImage(base64);
      setQueryImagePreview(base64);
      // Auto-enable ColPali when image is uploaded
      if (!searchOptions.use_colpali) {
        updateSearchOption("use_colpali", true);
        showAlert("Morphik multimodal retrieval enabled for image search", { type: "info", duration: 3000 });
      }
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setQueryImage(null);
    setQueryImagePreview(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  };

  const generateCodeSnippet = (lang: CodeLanguage): string => {
    const k = searchOptions.k || 10;
    const minScore = searchOptions.min_score || 0;
    const folder = searchOptions.folder_name;
    const reranking = searchOptions.use_reranking;
    const colpali = searchOptions.use_colpali;
    const padding = searchOptions.padding || 0;
    const hasImage = !!queryImage;
    const useGroupedEndpoint = padding > 0;

    // Escape helpers for different languages
    const escapeForPython = (str: string) => str.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
    const escapeForJS = (str: string) => str.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
    const escapeForJSON = (str: string) => str.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");

    if (lang === "python") {
      const folderParam = folder ? `\n    folder_name="${folder}",` : "";
      const paddingParam = padding > 0 ? `\n    padding=${padding},` : "";
      if (hasImage) {
        return `from morphik import Morphik

client = Morphik()

# Load your query image
with open("query_image.png", "rb") as f:
    import base64
    image_b64 = base64.b64encode(f.read()).decode()

results = client.retrieve_chunks(
    query_image=image_b64,
    k=${k},
    min_score=${minScore},${folderParam}${paddingParam}
    use_reranking=${reranking ? "True" : "False"},
    use_colpali=True  # Required for image search
)

for result in results:
    print(result.content)`;
      }
      const query = escapeForPython(searchQuery || "your search query");
      return `from morphik import Morphik

client = Morphik()

results = client.retrieve_chunks(
    query="${query}",
    k=${k},
    min_score=${minScore},${folderParam}${paddingParam}
    use_reranking=${reranking ? "True" : "False"},
    use_colpali=${colpali ? "True" : "False"}
)

for result in results:
    print(result.content)`;
    }

    if (lang === "javascript") {
      const folderParam = folder ? `\n  folder_name: "${folder}",` : "";
      const paddingParam = padding > 0 ? `\n  padding: ${padding},` : "";
      if (hasImage) {
        return `import Morphik from "morphik";
import fs from "fs";

const client = new Morphik();

// Load your query image
const imageBuffer = fs.readFileSync("query_image.png");
const imageB64 = imageBuffer.toString("base64");

const results = await client.retrieve.chunks.create({
  query_image: imageB64,
  k: ${k},
  min_score: ${minScore},${folderParam}${paddingParam}
  use_reranking: ${reranking},
  use_colpali: true  // Required for image search
});

results.forEach(result => {
  console.log(result.content);
});`;
      }
      const query = escapeForJS(searchQuery || "your search query");
      return `import Morphik from "morphik";

const client = new Morphik();

const results = await client.retrieve.chunks.create({
  query: "${query}",
  k: ${k},
  min_score: ${minScore},${folderParam}${paddingParam}
  use_reranking: ${reranking},
  use_colpali: ${colpali}
});

results.forEach(result => {
  console.log(result.content);
});`;
    }

    // curl
    const endpoint = useGroupedEndpoint ? "/retrieve/chunks/grouped" : "/retrieve/chunks";
    const folderParam = folder ? `\n  "folder_name": "${folder}",` : "";
    const paddingParam = padding > 0 ? `\n  "padding": ${padding},` : "";
    if (hasImage) {
      return `# First, encode your image to base64
# IMAGE_B64=$(base64 -i query_image.png)

curl -X POST "${apiBaseUrl}${endpoint}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
  "query_image": "data:image/png;base64,<IMAGE_B64>",
  "k": ${k},
  "min_score": ${minScore},${folderParam}${paddingParam}
  "use_reranking": ${reranking},
  "use_colpali": true
}'`;
    }
    const query = escapeForJSON(searchQuery || "your search query");
    return `curl -X POST "${apiBaseUrl}${endpoint}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
  "query": "${query}",
  "k": ${k},
  "min_score": ${minScore},${folderParam}${paddingParam}
  "use_reranking": ${reranking},
  "use_colpali": ${colpali}
}'`;
  };

  const handleCopyCode = async () => {
    await navigator.clipboard.writeText(generateCodeSnippet(codeLanguage));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Escape HTML entities to prevent XSS
  const escapeHtml = (str: string): string => {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };

  // Simple syntax highlighting for code
  const highlightCode = (line: string, lang: CodeLanguage): React.ReactNode => {
    // Escape HTML first to prevent XSS
    const escaped = escapeHtml(line);

    if (lang === "python") {
      // Python keywords
      const result = escaped
        .replace(/\b(from|import|for|in|def|class|return|if|else|elif|True|False|None|async|await)\b/g, "<kw>$1</kw>")
        .replace(/&quot;([^&]*)&quot;/g, "<str>&quot;$1&quot;</str>")
        .replace(/&#039;([^&]*)&#039;/g, "<str>&#039;$1&#039;</str>")
        .replace(/\b(\d+\.?\d*)\b/g, "<num>$1</num>")
        .replace(/\b(print|len|range|str|int|float|list|dict)\b(?=\()/g, "<fn>$1</fn>")
        .replace(/\.(\w+)\(/g, ".<method>$1</method>(")
        .replace(/(\w+)\s*=/g, "<var>$1</var> =");

      return (
        <span
          dangerouslySetInnerHTML={{
            __html: result
              .replace(/<kw>/g, '<span class="text-pink-400">')
              .replace(/<\/kw>/g, "</span>")
              .replace(/<str>/g, '<span class="text-green-400">')
              .replace(/<\/str>/g, "</span>")
              .replace(/<num>/g, '<span class="text-orange-400">')
              .replace(/<\/num>/g, "</span>")
              .replace(/<fn>/g, '<span class="text-yellow-300">')
              .replace(/<\/fn>/g, "</span>")
              .replace(/<method>/g, '<span class="text-blue-400">')
              .replace(/<\/method>/g, "</span>")
              .replace(/<var>/g, '<span class="text-zinc-100">')
              .replace(/<\/var>/g, "</span>"),
          }}
        />
      );
    }

    if (lang === "javascript") {
      const result = escaped
        .replace(
          /\b(import|from|const|let|var|function|return|if|else|for|of|in|async|await|new|true|false|null|undefined)\b/g,
          "<kw>$1</kw>"
        )
        .replace(/&quot;([^&]*)&quot;/g, "<str>&quot;$1&quot;</str>")
        .replace(/&#039;([^&]*)&#039;/g, "<str>&#039;$1&#039;</str>")
        .replace(/`([^`]*)`/g, "<str>`$1`</str>")
        .replace(/\b(\d+\.?\d*)\b/g, "<num>$1</num>")
        .replace(/\b(console)\b/g, "<fn>$1</fn>")
        .replace(/\.(\w+)\(/g, ".<method>$1</method>(");

      return (
        <span
          dangerouslySetInnerHTML={{
            __html: result
              .replace(/<kw>/g, '<span class="text-pink-400">')
              .replace(/<\/kw>/g, "</span>")
              .replace(/<str>/g, '<span class="text-green-400">')
              .replace(/<\/str>/g, "</span>")
              .replace(/<num>/g, '<span class="text-orange-400">')
              .replace(/<\/num>/g, "</span>")
              .replace(/<fn>/g, '<span class="text-yellow-300">')
              .replace(/<\/fn>/g, "</span>")
              .replace(/<method>/g, '<span class="text-blue-400">')
              .replace(/<\/method>/g, "</span>"),
          }}
        />
      );
    }

    // curl
    const result = escaped
      .replace(/\b(curl)\b/g, "<kw>$1</kw>")
      .replace(/(-X|-H|-d)\b/g, "<flag>$1</flag>")
      .replace(/&quot;([^&]*)&quot;/g, "<str>&quot;$1&quot;</str>")
      .replace(/&#039;([^&]*)&#039;/g, "<str>&#039;$1&#039;</str>")
      .replace(/\b(\d+\.?\d*)\b/g, "<num>$1</num>")
      .replace(/\b(POST|GET|PUT|DELETE|PATCH)\b/g, "<method>$1</method>");

    return (
      <span
        dangerouslySetInnerHTML={{
          __html: result
            .replace(/<kw>/g, '<span class="text-pink-400">')
            .replace(/<\/kw>/g, "</span>")
            .replace(/<flag>/g, '<span class="text-cyan-400">')
            .replace(/<\/flag>/g, "</span>")
            .replace(/<str>/g, '<span class="text-green-400">')
            .replace(/<\/str>/g, "</span>")
            .replace(/<num>/g, '<span class="text-orange-400">')
            .replace(/<\/num>/g, "</span>")
            .replace(/<method>/g, '<span class="text-yellow-300">')
            .replace(/<\/method>/g, "</span>"),
        }}
      />
    );
  };

  const folderValue = Array.isArray(searchOptions.folder_name)
    ? searchOptions.folder_name[0]
    : searchOptions.folder_name;

  return (
    <ResizablePanelGroup direction="horizontal" className="h-full">
      {/* Main content - Fixed (no scroll) */}
      <ResizablePanel defaultSize={50} minSize={30} maxSize={80}>
        <ScrollArea className="h-full">
          <div className="flex flex-col p-1 pr-4">
            {/* Header */}
            <div className="mb-6 flex items-start justify-between">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">Search</h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  Search your documents and retrieve relevant content
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" asChild>
                  <a
                    href="https://docs.morphik.ai/api-reference/retrieve-chunks"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Book className="mr-2 h-4 w-4" />
                    Docs
                  </a>
                </Button>
              </div>
            </div>

            {/* Query Section */}
            <div className="rounded-lg border bg-card">
              <div className="p-4">
                <Label className="mb-2 block text-sm font-medium">Query</Label>
                <input
                  type="file"
                  ref={imageInputRef}
                  onChange={handleImageUpload}
                  accept="image/*"
                  className="hidden"
                />
                {queryImagePreview ? (
                  <div className="relative inline-block">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={queryImagePreview}
                      alt="Query image"
                      className="max-h-[100px] rounded-lg border object-contain"
                    />
                    <button
                      onClick={handleRemoveImage}
                      className="absolute -right-2 -top-2 rounded-full bg-destructive p-1 text-destructive-foreground hover:bg-destructive/90"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ) : (
                  <Textarea
                    placeholder="Enter your search query, or upload image for image search"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSearch();
                      }
                    }}
                    className="min-h-[100px] resize-none border-0 bg-transparent p-0 text-base focus-visible:ring-0"
                  />
                )}
              </div>
              <div className="flex items-center justify-between border-t bg-muted/30 px-4 py-3">
                <Button variant="outline" size="sm" onClick={() => imageInputRef.current?.click()}>
                  <ImagePlus className="mr-2 h-4 w-4" />
                  {queryImagePreview ? "Change image" : "Upload image"}
                </Button>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm" onClick={handleClear} disabled={!searchQuery && !queryImagePreview}>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Clear
                  </Button>
                  <Button onClick={handleSearch} disabled={loading} size="sm">
                    <Search className="mr-2 h-4 w-4" />
                    {loading ? "Searching..." : "Search"}
                  </Button>
                </div>
              </div>
            </div>

            {/* Options Section */}
            <div className="mt-6 space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {/* Number of Results */}
                <div>
                  <Label className="mb-2 block text-sm">Number of results</Label>
                  <Input
                    type="number"
                    min={1}
                    max={100}
                    value={searchOptions.k || 10}
                    onChange={e => updateSearchOption("k", parseInt(e.target.value) || 10)}
                    placeholder="Default: 10"
                  />
                  <p className="mt-1 text-xs text-muted-foreground">Max: 100</p>
                </div>

                {/* Folder Scope */}
                <div>
                  <Label className="mb-2 block text-sm">Scope to folder</Label>
                  <Select
                    value={folderValue || "__all__"}
                    onValueChange={v => updateSearchOption("folder_name", v === "__all__" ? undefined : v)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="All folders" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__all__">All folders</SelectItem>
                      {folders.map(folder => (
                        <SelectItem key={folder.name} value={folder.name}>
                          {folder.name} {folder.doc_count !== undefined && `(${folder.doc_count})`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Min Score */}
                <div>
                  <Label className="mb-2 block text-sm">Minimum score</Label>
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={searchOptions.min_score || 0}
                    onChange={e => updateSearchOption("min_score", parseFloat(e.target.value) || 0)}
                    placeholder="0.0"
                  />
                  <p className="mt-1 text-xs text-muted-foreground">Filter results below this threshold</p>
                </div>
              </div>

              {/* Metadata Filters */}
              <div>
                <Label className="mb-2 block text-sm">Metadata filters (JSON)</Label>
                <Textarea
                  value={
                    typeof searchOptions.filters === "object"
                      ? JSON.stringify(searchOptions.filters, null, 2)
                      : searchOptions.filters || ""
                  }
                  onChange={e => updateSearchOption("filters", e.target.value)}
                  placeholder='{"category": "reports"}'
                  rows={3}
                  className="font-mono text-sm"
                />
                <p className="mt-1 text-xs text-muted-foreground">Filter results by document metadata</p>
              </div>

              {/* Advanced Settings */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div>
                    <Label className="text-sm">Use reranking</Label>
                    <p className="text-xs text-muted-foreground">Improve result relevance with reranking</p>
                  </div>
                  <Switch
                    checked={searchOptions.use_reranking || false}
                    onCheckedChange={checked => updateSearchOption("use_reranking", checked)}
                  />
                </div>
                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div>
                    <Label className="text-sm">Use ColPali</Label>
                    <p className="text-xs text-muted-foreground">Visual document understanding</p>
                  </div>
                  <Switch
                    checked={!!searchOptions.use_colpali}
                    onCheckedChange={checked => updateSearchOption("use_colpali", checked)}
                  />
                </div>
              </div>

              {searchOptions.use_colpali && (
                <div>
                  <Label className="mb-2 block text-sm">Padding</Label>
                  <Input
                    type="number"
                    min={0}
                    max={10}
                    value={searchOptions.padding || 0}
                    onChange={e => updateSearchOption("padding", parseInt(e.target.value) || 0)}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    Additional pages to retrieve before and after matched pages
                  </p>
                </div>
              )}
            </div>
          </div>
        </ScrollArea>
      </ResizablePanel>

      <ResizableHandle withHandle />

      {/* Code/Output Panel */}
      <ResizablePanel defaultSize={50} minSize={20} maxSize={70}>
        <div className="ml-4 flex h-full flex-col overflow-hidden rounded-lg border bg-zinc-950">
          {/* Top tabs: Code / Output */}
          <div className="flex items-center border-b border-zinc-800">
            <button
              onClick={() => setRightPanelTab("code")}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                rightPanelTab === "code" ? "border-b-2 border-white text-white" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Code2 className="h-4 w-4" />
              Code
            </button>
            <button
              onClick={() => setRightPanelTab("output")}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                rightPanelTab === "output" ? "border-b-2 border-white text-white" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Terminal className="h-4 w-4" />
              Output
            </button>
          </div>

          {rightPanelTab === "code" ? (
            <>
              {/* Language tabs */}
              <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setCodeLanguage("python")}
                    className={`flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      codeLanguage === "python"
                        ? "bg-zinc-800 text-zinc-100"
                        : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300"
                    }`}
                  >
                    <span className="text-yellow-500">&#x1F40D;</span>
                    Python
                  </button>
                  <button
                    onClick={() => setCodeLanguage("javascript")}
                    className={`flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      codeLanguage === "javascript"
                        ? "bg-zinc-800 text-zinc-100"
                        : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300"
                    }`}
                  >
                    <span className="text-yellow-400">JS</span>
                    Javascript
                  </button>
                  <button
                    onClick={() => setCodeLanguage("curl")}
                    className={`flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      codeLanguage === "curl"
                        ? "bg-zinc-800 text-zinc-100"
                        : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300"
                    }`}
                  >
                    <span className="text-zinc-400">&gt;_</span>
                    curl
                  </button>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
                  onClick={handleCopyCode}
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>

              {/* Code with line numbers */}
              <ScrollArea className="flex-1">
                <div className="p-4">
                  <pre className="text-sm">
                    <code className="text-zinc-300">
                      {generateCodeSnippet(codeLanguage)
                        .split("\n")
                        .map((line, i) => (
                          <div key={i} className="flex">
                            <span className="mr-4 inline-block w-5 select-none text-right text-zinc-600">{i + 1}</span>
                            <span className="flex-1">{highlightCode(line, codeLanguage)}</span>
                          </div>
                        ))}
                    </code>
                  </pre>
                </div>
              </ScrollArea>
            </>
          ) : (
            /* Output tab - Visual Results */
            <div className="flex flex-1 flex-col overflow-hidden bg-zinc-900">
              {searchResults.length > 0 ? (
                <>
                  {/* Sticky Results header */}
                  <div className="sticky top-0 z-10 flex items-center justify-between border-b border-zinc-800 bg-zinc-900 px-4 py-3">
                    <h3 className="text-sm font-medium text-zinc-100">Results ({searchResults.length})</h3>
                  </div>
                  <ScrollArea className="flex-1">
                    <div className="space-y-3 p-4">
                      {groupedResults?.has_padding
                        ? groupedResults.groups.map(group => (
                            <SearchResultCardCarousel
                              key={`${group.main_chunk.document_id}-${group.main_chunk.chunk_number}`}
                              group={group}
                              darkMode
                            />
                          ))
                        : searchResults.map(result => (
                            <SearchResultCard
                              key={`${result.document_id}-${result.chunk_number}`}
                              result={result}
                              darkMode
                            />
                          ))}
                    </div>
                  </ScrollArea>
                </>
              ) : (
                <div className="flex flex-1 flex-col items-center justify-center py-12 text-center">
                  <Terminal className="mb-3 h-10 w-10 text-zinc-600" />
                  <p className="text-sm text-zinc-400">No output yet</p>
                  <p className="mt-1 text-xs text-zinc-500">Run a search to see results here</p>
                </div>
              )}
            </div>
          )}
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

export default SearchSection;
