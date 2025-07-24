"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { listConnectorFiles, ConnectorFile } from "@/lib/connectorsApi";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Folder as FolderIcon,
  File as FileIcon,
  ChevronRight,
  Home,
  UploadCloud,
  Loader2,
  AlertCircle,
} from "lucide-react";

const DEFAULT_PAGE_SIZE = 20;

interface FileBrowserProps {
  connectorType: string;
  apiBaseUrl: string;
  authToken: string | null;
  onFileIngest: (fileId: string, fileName: string, connectorType: string) => void; // Added connectorType
  onRepositoryIngest?: (repoPath: string, connectorType: string) => void; // New prop for repository ingestion
  // onPathChange?: (newPath: string | null, newPathName: string | null) => void; // Optional: if parent needs to know
}

interface PathCrumb {
  id: string | null;
  name: string;
}

export function FileBrowser({
  connectorType,
  apiBaseUrl,
  authToken,
  onFileIngest,
  onRepositoryIngest,
}: FileBrowserProps) {
  const [files, setFiles] = useState<ConnectorFile[]>([]);
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [pathHistory, setPathHistory] = useState<PathCrumb[]>([{ id: null, name: "Root" }]);
  const [nextPageToken, setNextPageToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState<Record<string, boolean>>({});
  const [ingestingRepos, setIngestingRepos] = useState<Record<string, boolean>>({});

  const currentFolderName = useMemo(() => {
    return pathHistory.length > 0 ? pathHistory[pathHistory.length - 1].name : "Root";
  }, [pathHistory]);

  const fetchFilesData = useCallback(
    async (pathId: string | null, pageToken?: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await listConnectorFiles(
          apiBaseUrl,
          connectorType,
          authToken,
          pathId, // API function handles 'root' case if pathId is 'root' or null
          pageToken,
          undefined,
          DEFAULT_PAGE_SIZE
        );
        setFiles(prevFiles => (pageToken ? [...prevFiles, ...result.files] : result.files));
        setNextPageToken(result.next_page_token || null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred.");
        setFiles([]); // Clear files on error
        setNextPageToken(null);
      } finally {
        setIsLoading(false);
      }
    },
    [apiBaseUrl, connectorType, authToken]
  );

  useEffect(() => {
    // Fetch files when currentPath changes or (debounced) searchTerm changes
    // For simplicity, direct searchTerm usage here. Debounce recommended for production.
    fetchFilesData(currentPath, undefined);
  }, [currentPath, fetchFilesData]);

  const handleFolderClick = (folder: ConnectorFile) => {
    if (!folder.is_folder) return;
    const newPathId = folder.id; // folder.id should be string for a navigable folder from GDrive
    setPathHistory(prev => [...prev, { id: newPathId, name: folder.name }]);
    setCurrentPath(newPathId);
    setFiles([]); // Clear old files before new fetch
    setNextPageToken(null);
  };

  const handleBreadcrumbClick = (crumbIndex: number) => {
    const newPathHistory = pathHistory.slice(0, crumbIndex + 1);
    setPathHistory(newPathHistory);
    const newCurrentPath = newPathHistory[newPathHistory.length - 1].id;
    setCurrentPath(newCurrentPath);
    setFiles([]);
    setNextPageToken(null);
  };

  const handleLoadMore = () => {
    if (nextPageToken) {
      fetchFilesData(currentPath, nextPageToken);
    }
  };

  const handleIngestFile = async (file: ConnectorFile) => {
    if (file.is_folder) return;
    setIsIngesting(prev => ({ ...prev, [file.id]: true }));
    try {
      onFileIngest(file.id, file.name, connectorType);
      // Optionally, provide some feedback here or rely on parent for notifications
      // For example, after successful call to onFileIngest, you might want to clear the ingesting state
      // or the parent component might re-render or navigate away.
    } catch (ingestError) {
      setError(ingestError instanceof Error ? ingestError.message : "Failed to start ingestion.");
      // Reset ingesting state for this specific file on error
      setIsIngesting(prev => ({ ...prev, [file.id]: false }));
    }
  };

  const handleIngestRepository = async (file: ConnectorFile) => {
    if (!file.is_folder || connectorType !== "github") return;

    // For GitHub, repository ID is in format "owner/repo"
    const repoPath = file.id;
    setIngestingRepos(prev => ({ ...prev, [repoPath]: true }));
    setError(null);

    try {
      if (onRepositoryIngest) {
        onRepositoryIngest(repoPath, connectorType);
      } else {
        // Fallback or direct call if parent doesn't handle it
        throw new Error("Repository ingestion not supported");
      }
      // You can add a success notification here if needed
    } catch (ingestError) {
      const errorMessage = ingestError instanceof Error ? ingestError.message : "Failed to start repository ingestion.";
      setError(errorMessage);
      console.error("Repository ingestion error:", ingestError);
    } finally {
      // We might want to keep the loading state until parent confirms completion
      // For now, let's just reset it. Parent can manage this via props if needed.
      setIngestingRepos(prev => ({ ...prev, [repoPath]: false }));
    }
  };

  // Helper function to determine if we're at repository root level for GitHub
  const isAtRepositoryLevel = () => {
    return connectorType === "github" && currentPath === null;
  };

  // --- RENDER LOGIC ---
  if (isLoading && files.length === 0) {
    // Initial load
    return (
      <div className="flex items-center justify-center p-10">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-2">Loading files...</span>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col space-y-4 rounded-lg border p-4">
      {/* Header: Breadcrumbs and Search */}
      <div className="flex flex-col items-center justify-between gap-2 border-b pb-2 sm:flex-row">
        <div className="flex items-center space-x-1 overflow-x-auto whitespace-nowrap">
          {pathHistory.map((crumb, index) => (
            <React.Fragment key={crumb.id ?? `root-crumb-${index}`}>
              {index > 0 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
              <Button
                variant="link"
                className={`p-1 text-sm ${index === pathHistory.length - 1 ? "font-semibold text-primary" : "text-muted-foreground"}`}
                onClick={() => handleBreadcrumbClick(index)}
                disabled={isLoading}
              >
                {index === 0 ? <Home className="mr-1 h-4 w-4" /> : null}
                {crumb.name}
              </Button>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {!isLoading && error && (
        <div className="flex items-center space-x-2 rounded-md bg-red-50 p-3 text-red-600">
          <AlertCircle className="h-5 w-5" />
          <span>Error: {error}</span>
          <Button variant="outline" size="sm" onClick={() => fetchFilesData(currentPath, undefined)}>
            Try Again
          </Button>
        </div>
      )}

      {/* File Table */}
      <div className="flex-grow overflow-y-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[50px]"></TableHead> {/* Icon */}
              <TableHead>Name</TableHead>
              <TableHead className="hidden md:table-cell">Last Modified</TableHead>
              <TableHead className="hidden sm:table-cell">Size</TableHead>
              <TableHead className="w-[100px] text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {!isLoading && files.length === 0 && !error && (
              <TableRow>
                <TableCell colSpan={5} className="h-24 text-center text-muted-foreground">
                  No files or folders found in {currentFolderName}.
                </TableCell>
              </TableRow>
            )}
            {files.map(file => (
              <TableRow key={file.id}>
                <TableCell>
                  {file.is_folder ? (
                    <FolderIcon className="h-5 w-5 text-blue-500" />
                  ) : (
                    <FileIcon className="h-5 w-5 text-gray-500" />
                  )}
                </TableCell>
                <TableCell
                  className={`font-medium ${file.is_folder ? "cursor-pointer hover:underline" : ""}`}
                  onClick={() => file.is_folder && handleFolderClick(file)}
                >
                  {file.name}
                </TableCell>
                <TableCell className="hidden text-sm text-muted-foreground md:table-cell">
                  {file.modified_date ? new Date(file.modified_date).toLocaleDateString() : "N/A"}
                </TableCell>
                <TableCell className="hidden text-sm text-muted-foreground sm:table-cell">
                  {file.is_folder ? "-" : file.size ? (file.size / 1024).toFixed(1) + " KB" : "N/A"}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex justify-end gap-2">
                    {!file.is_folder && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleIngestFile(file)}
                        disabled={isIngesting[file.id] || isLoading}
                      >
                        {isIngesting[file.id] ? (
                          <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                        ) : (
                          <UploadCloud className="mr-1 h-4 w-4" />
                        )}
                        Ingest
                      </Button>
                    )}
                    {file.is_folder && isAtRepositoryLevel() && onRepositoryIngest && (
                      <Button
                        variant="default"
                        size="sm"
                        onClick={() => handleIngestRepository(file)}
                        disabled={ingestingRepos[file.id] || isLoading}
                      >
                        {ingestingRepos[file.id] ? (
                          <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                        ) : (
                          <UploadCloud className="mr-1 h-4 w-4" />
                        )}
                        Ingest Repo
                      </Button>
                    )}
                  </div>
                </TableCell>
              </TableRow>
            ))}
            {/* In-table loading indicator for pagination, less intrusive */}
            {isLoading && files.length > 0 && (
              <TableRow>
                <TableCell colSpan={5} className="text-center">
                  <Loader2 className="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination/Load More */}
      {!isLoading && nextPageToken && (
        <div className="flex justify-center pt-4">
          <Button variant="outline" onClick={handleLoadMore} disabled={isLoading}>
            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Load More
          </Button>
        </div>
      )}
    </div>
  );
}
