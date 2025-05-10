'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  listConnectorFiles,
  ConnectorFile,
  // ingestConnectorFile, // Parent will call this via onFileIngest
} from '@/lib/connectorsApi';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Folder as FolderIcon,
  File as FileIcon,
  ChevronRight,
  Home,
  UploadCloud,
  Loader2,
  AlertCircle,
  Search,
} from 'lucide-react';
// import { useDebounce } from '@/hooks/useDebounce'; // Assuming you have a debounce hook

const DEFAULT_PAGE_SIZE = 20;

interface FileBrowserProps {
  connectorType: string;
  apiBaseUrl: string;
  onFileIngest: (fileId: string, fileName: string, connectorType: string) => void; // Added connectorType
  initialPath?: string | null; // Allow null for root
  // onPathChange?: (newPath: string | null, newPathName: string | null) => void; // Optional: if parent needs to know
}

interface PathCrumb {
  id: string | null;
  name: string;
}

export function FileBrowser({
  connectorType,
  apiBaseUrl,
  onFileIngest,
  initialPath = null, // Default to null for root
}: FileBrowserProps) {
  const [files, setFiles] = useState<ConnectorFile[]>([]);
  const [currentPath, setCurrentPath] = useState<string | null>(initialPath);
  const [pathHistory, setPathHistory] = useState<PathCrumb[]>([{ id: initialPath, name: 'Root' }]);
  const [nextPageToken, setNextPageToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');
  // const debouncedSearchTerm = useDebounce(searchTerm, 500); // Example of debouncing
  const [isIngesting, setIsIngesting] = useState<Record<string, boolean>>({});

  const currentFolderName = useMemo(() => {
    return pathHistory.length > 0 ? pathHistory[pathHistory.length - 1].name : 'Root';
  }, [pathHistory]);


  const fetchFilesData = useCallback(async (pathId: string | null, pageToken?: string, searchQuery?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await listConnectorFiles(
        apiBaseUrl,
        connectorType,
        pathId, // API function handles 'root' case if pathId is 'root' or null
        pageToken,
        searchQuery,
        DEFAULT_PAGE_SIZE
      );
      setFiles(prevFiles => pageToken ? [...prevFiles, ...result.files] : result.files);
      setNextPageToken(result.next_page_token || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
      setFiles([]); // Clear files on error
      setNextPageToken(null);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl, connectorType]);

  useEffect(() => {
    // Fetch files when currentPath changes or (debounced) searchTerm changes
    // For simplicity, direct searchTerm usage here. Debounce recommended for production.
    fetchFilesData(currentPath, undefined, searchTerm || undefined);
  }, [currentPath, searchTerm, fetchFilesData]);


  const handleFolderClick = (folder: ConnectorFile) => {
    if (!folder.is_folder) return;
    const newPathId = folder.id; // folder.id should be string for a navigable folder from GDrive
    setPathHistory(prev => [...prev, { id: newPathId, name: folder.name }]);
    setCurrentPath(newPathId);
    setFiles([]); // Clear old files before new fetch
    setNextPageToken(null);
    setSearchTerm(''); // Reset search when navigating folders
  };

  const handleBreadcrumbClick = (crumbIndex: number) => {
    const newPathHistory = pathHistory.slice(0, crumbIndex + 1);
    setPathHistory(newPathHistory);
    const newCurrentPath = newPathHistory[newPathHistory.length - 1].id;
    setCurrentPath(newCurrentPath);
    setFiles([]);
    setNextPageToken(null);
    setSearchTerm('');
  };

  const handleLoadMore = () => {
    if (nextPageToken) {
      fetchFilesData(currentPath, nextPageToken, searchTerm || undefined);
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
      setError(ingestError instanceof Error ? ingestError.message : 'Failed to start ingestion.');
      // Reset ingesting state for this specific file on error
      setIsIngesting(prev => ({ ...prev, [file.id]: false }));
    }
  };

  const handleSearchInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    // If not using debounced search in useEffect, you might trigger search here or on submit
  };

  const handleSearchSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // fetchFilesData will be called by useEffect due to searchTerm change
    // If not using debounced/useEffect search, call explicitly:
    // setCurrentPath(null); // Reset to root or decide search scope
    // setPathHistory([{ id: null, name: 'Root' }]);
    // fetchFilesData(null, undefined, searchTerm);
     // No need to call fetchFilesData explicitly if useEffect handles searchTerm changes
  };


  // --- RENDER LOGIC ---
  if (isLoading && files.length === 0) { // Initial load
    return (
      <div className="flex items-center justify-center p-10">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-2">Loading files...</span>
      </div>
    );
  }

  return (
    <div className="p-4 border rounded-lg space-y-4">
      {/* Header: Breadcrumbs and Search */}
      <div className="flex flex-col sm:flex-row justify-between items-center gap-2 pb-2 border-b">
        <div className="flex items-center space-x-1 overflow-x-auto whitespace-nowrap">
          {pathHistory.map((crumb, index) => (
            <React.Fragment key={crumb.id ?? `root-crumb-${index}`}>
              {index > 0 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
              <Button
                variant="link"
                className={`p-1 text-sm ${index === pathHistory.length - 1 ? 'font-semibold text-primary' : 'text-muted-foreground'}`}
                onClick={() => handleBreadcrumbClick(index)}
                disabled={isLoading}
              >
                {index === 0 ? <Home className="h-4 w-4 mr-1" /> : null}
                {crumb.name}
              </Button>
            </React.Fragment>
          ))}
        </div>
        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2">
          <Input
            type="search"
            placeholder="Search files..."
            value={searchTerm}
            onChange={handleSearchInputChange}
            className="h-9 max-w-xs"
            disabled={isLoading}
          />
           {/* <Button type="submit" variant="outline" size="icon" disabled={isLoading}>
            <Search className="h-4 w-4" />
          </Button> */}
        </form>
      </div>

      {/* Error Display */}
      {!isLoading && error && (
        <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-md">
          <AlertCircle className="h-5 w-5" />
          <span>Error: {error}</span>
          <Button variant="outline" size="sm" onClick={() => fetchFilesData(currentPath, undefined, searchTerm || undefined)}>Try Again</Button>
        </div>
      )}

      {/* File Table */}
      <div className="rounded-md border max-h-[500px] overflow-y-auto">
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
            {files.map((file) => (
              <TableRow key={file.id}>
                <TableCell>
                  {file.is_folder ? (
                    <FolderIcon className="h-5 w-5 text-blue-500" />
                  ) : (
                    <FileIcon className="h-5 w-5 text-gray-500" />
                  )}
                </TableCell>
                <TableCell
                  className={`font-medium ${file.is_folder ? 'cursor-pointer hover:underline' : ''}`}
                  onClick={() => file.is_folder && handleFolderClick(file)}
                >
                  {file.name}
                </TableCell>
                <TableCell className="hidden md:table-cell text-sm text-muted-foreground">
                  {file.modified_date ? new Date(file.modified_date).toLocaleDateString() : 'N/A'}
                </TableCell>
                <TableCell className="hidden sm:table-cell text-sm text-muted-foreground">
                  {file.is_folder ? '-' : (file.size ? (file.size / 1024).toFixed(1) + ' KB' : 'N/A')}
                </TableCell>
                <TableCell className="text-right">
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
                </TableCell>
              </TableRow>
            ))}
            {/* In-table loading indicator for pagination, less intrusive */}
            {isLoading && files.length > 0 && (
                <TableRow>
                    <TableCell colSpan={5} className="text-center">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground mx-auto" />
                    </TableCell>
                </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination/Load More */}
      {!isLoading && nextPageToken && (
        <div className="pt-4 flex justify-center">
          <Button variant="outline" onClick={handleLoadMore} disabled={isLoading}>
            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Load More
          </Button>
        </div>
      )}
    </div>
  );
}
