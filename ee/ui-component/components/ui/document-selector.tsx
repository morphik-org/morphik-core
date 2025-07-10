"use client";

import React, { useState, useCallback, useMemo } from "react";
import { ChevronUp, Search, Folder, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DocumentSelectorDocument {
  id: string;
  filename: string;
  folder_name?: string;
  content_type?: string;
  metadata?: Record<string, unknown>;
  system_metadata?: unknown;
}

interface DocumentSelectorProps {
  documents: DocumentSelectorDocument[];
  folders: Array<{ name: string; doc_count: number }>;
  selectedDocuments: string[];
  selectedFolders: string[];
  onDocumentSelectionChange: (documentIds: string[]) => void;
  onFolderSelectionChange: (folderNames: string[]) => void;
  loading?: boolean;
  placeholder?: string;
  className?: string;
}

export function DocumentSelector({
  documents,
  folders,
  selectedDocuments,
  selectedFolders,
  onDocumentSelectionChange,
  onFolderSelectionChange,
  loading = false,
  placeholder = "Select documents and folders",
  className,
}: DocumentSelectorProps) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Group documents by folder
  const groupedDocuments = useMemo(() => {
    const grouped: Record<string, DocumentSelectorDocument[]> = {};

    documents.forEach(doc => {
      const folderName = doc.folder_name || "Unorganized";
      if (!grouped[folderName]) {
        grouped[folderName] = [];
      }
      grouped[folderName].push(doc);
    });

    return grouped;
  }, [documents]);

  // Filter documents and folders based on search query
  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) {
      return { folders, documents, groupedDocuments };
    }

    const query = searchQuery.toLowerCase();

    // Filter folders
    const filteredFolders = folders.filter(folder => folder.name.toLowerCase().includes(query));

    // Filter documents
    const filteredDocuments = documents.filter(
      doc =>
        doc.filename.toLowerCase().includes(query) || (doc.folder_name && doc.folder_name.toLowerCase().includes(query))
    );

    // Regroup filtered documents
    const filteredGroupedDocuments: Record<string, DocumentSelectorDocument[]> = {};
    filteredDocuments.forEach(doc => {
      const folderName = doc.folder_name || "Unorganized";
      if (!filteredGroupedDocuments[folderName]) {
        filteredGroupedDocuments[folderName] = [];
      }
      filteredGroupedDocuments[folderName].push(doc);
    });

    return {
      folders: filteredFolders,
      documents: filteredDocuments,
      groupedDocuments: filteredGroupedDocuments,
    };
  }, [folders, documents, groupedDocuments, searchQuery]);

  // Handle folder selection
  const handleFolderToggle = useCallback(
    (folderName: string) => {
      const newSelectedFolders = selectedFolders.includes(folderName)
        ? selectedFolders.filter(name => name !== folderName)
        : [...selectedFolders, folderName];

      onFolderSelectionChange(newSelectedFolders);

      // When a folder is selected, also select/deselect all its documents
      const folderDocuments = groupedDocuments[folderName] || [];
      const folderDocumentIds = folderDocuments.map(doc => doc.id);

      if (selectedFolders.includes(folderName)) {
        // Deselecting folder - remove all its documents from selection
        const newSelectedDocuments = selectedDocuments.filter(id => !folderDocumentIds.includes(id));
        onDocumentSelectionChange(newSelectedDocuments);
      } else {
        // Selecting folder - add all its documents to selection
        const newSelectedDocuments = [
          ...selectedDocuments.filter(id => !folderDocumentIds.includes(id)),
          ...folderDocumentIds,
        ];
        onDocumentSelectionChange(newSelectedDocuments);
      }
    },
    [selectedFolders, selectedDocuments, onFolderSelectionChange, onDocumentSelectionChange, groupedDocuments]
  );

  // Handle document selection
  const handleDocumentToggle = useCallback(
    (documentId: string) => {
      const newSelectedDocuments = selectedDocuments.includes(documentId)
        ? selectedDocuments.filter(id => id !== documentId)
        : [...selectedDocuments, documentId];

      onDocumentSelectionChange(newSelectedDocuments);
    },
    [selectedDocuments, onDocumentSelectionChange]
  );

  // Handle "Select All" toggle
  const handleSelectAllToggle = useCallback(() => {
    const allDocumentIds = documents.map(doc => doc.id);
    const allFolderNames = folders.map(folder => folder.name);

    const isAllSelected = selectedDocuments.length === allDocumentIds.length;

    if (isAllSelected) {
      onDocumentSelectionChange([]);
      onFolderSelectionChange([]);
    } else {
      onDocumentSelectionChange(allDocumentIds);
      onFolderSelectionChange(allFolderNames);
    }
  }, [documents, folders, selectedDocuments, onDocumentSelectionChange, onFolderSelectionChange]);

  // Get display text for the trigger button
  const getDisplayText = useCallback(() => {
    const totalSelected = selectedDocuments.length + selectedFolders.length;

    if (totalSelected === 0) {
      return placeholder;
    }

    if (totalSelected === 1) {
      if (selectedFolders.length === 1) {
        return `${selectedFolders[0]} (folder)`;
      }
      const selectedDoc = documents.find(doc => doc.id === selectedDocuments[0]);
      return selectedDoc ? selectedDoc.filename : "1 document";
    }

    return `${totalSelected} items selected`;
  }, [selectedDocuments, selectedFolders, documents, placeholder]);

  // Check if all items are selected
  const isAllSelected = useMemo(() => {
    return selectedDocuments.length === documents.length && selectedFolders.length === folders.length;
  }, [selectedDocuments, selectedFolders, documents, folders]);

  // Check if some items are selected (for indeterminate state)
  const isSomeSelected = useMemo(() => {
    return selectedDocuments.length > 0 || selectedFolders.length > 0;
  }, [selectedDocuments, selectedFolders]);

  // Type guard for system metadata
  const getSystemMetadata = (metadata: unknown) => {
    if (metadata && typeof metadata === "object" && metadata !== null) {
      return metadata as { file_size?: number; created_at?: string };
    }
    return undefined;
  };

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  return (
    <div className={className}>
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="h-auto w-full justify-between px-3 py-2"
          >
            <div className="flex min-w-0 flex-1 items-center gap-2">
              <span className="truncate text-left">{getDisplayText()}</span>
              {isSomeSelected && (
                <div className="ml-2 flex items-center gap-1">
                  {selectedFolders.map(folder => (
                    <Badge key={folder} variant="secondary" className="text-xs">
                      <Folder className="mr-1 h-3 w-3" />
                      {folder}
                    </Badge>
                  ))}
                  {selectedDocuments.length > 0 && (
                    <Badge variant="secondary" className="text-xs">
                      {selectedDocuments.length} docs
                    </Badge>
                  )}
                </div>
              )}
            </div>
            <ChevronUp className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          className="w-[var(--radix-dropdown-menu-trigger-width)] min-w-[500px] p-0 duration-200 animate-in slide-in-from-bottom-4"
          align="start"
          side="top"
          sideOffset={8}
        >
          <div className="flex h-[400px] flex-col">
            {/* Header with search and select all */}
            <div className="border-b p-3">
              <div className="mb-2 flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search documents and folders..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    className="pl-8"
                  />
                </div>
                <Button variant="ghost" size="sm" onClick={handleSelectAllToggle} className="whitespace-nowrap">
                  {isAllSelected ? "Deselect All" : "Select All"}
                </Button>
              </div>
            </div>

            {/* Content area */}
            <ScrollArea className="flex-1">
              <div className="p-2">
                {loading ? (
                  <div className="py-8 text-center text-muted-foreground">Loading documents...</div>
                ) : (
                  <div className="space-y-2">
                    {/* Folders */}
                    {filteredItems.folders.map(folder => {
                      const isSelected = selectedFolders.includes(folder.name);
                      const folderDocuments = filteredItems.groupedDocuments[folder.name] || [];

                      return (
                        <div key={folder.name} className="space-y-1">
                          <div
                            className={cn(
                              "flex cursor-pointer items-center gap-2 rounded-md p-2 hover:bg-muted/50",
                              isSelected && "bg-muted"
                            )}
                            onClick={() => handleFolderToggle(folder.name)}
                          >
                            <Checkbox checked={isSelected} onChange={() => handleFolderToggle(folder.name)} />
                            <Folder className="h-4 w-4 text-blue-500" />
                            <div className="min-w-0 flex-1">
                              <div className="truncate font-medium">{folder.name}</div>
                              <div className="text-xs text-muted-foreground">{folder.doc_count} documents</div>
                            </div>
                          </div>

                          {/* Documents in this folder */}
                          {folderDocuments.map(doc => {
                            const isDocSelected = selectedDocuments.includes(doc.id);

                            return (
                              <div
                                key={doc.id}
                                className={cn(
                                  "ml-6 flex cursor-pointer items-center gap-2 rounded-md p-2 hover:bg-muted/50",
                                  isDocSelected && "bg-muted"
                                )}
                                onClick={() => handleDocumentToggle(doc.id)}
                              >
                                <Checkbox checked={isDocSelected} onChange={() => handleDocumentToggle(doc.id)} />
                                <FileText className="h-4 w-4 text-gray-500" />
                                <div className="min-w-0 flex-1">
                                  <div className="truncate text-sm font-medium">{doc.filename}</div>
                                  <div className="text-xs text-muted-foreground">
                                    {getSystemMetadata(doc.system_metadata)?.file_size &&
                                      formatFileSize(getSystemMetadata(doc.system_metadata)!.file_size!)}
                                    {getSystemMetadata(doc.system_metadata)?.created_at && (
                                      <span className="ml-2">
                                        {new Date(
                                          getSystemMetadata(doc.system_metadata)!.created_at!
                                        ).toLocaleDateString()}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      );
                    })}

                    {/* Unorganized documents */}
                    {filteredItems.groupedDocuments["Unorganized"] && (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 p-2 text-sm font-medium text-muted-foreground">
                          <FileText className="h-4 w-4" />
                          Unorganized Documents
                        </div>
                        {filteredItems.groupedDocuments["Unorganized"].map(doc => {
                          const isDocSelected = selectedDocuments.includes(doc.id);

                          return (
                            <div
                              key={doc.id}
                              className={cn(
                                "ml-6 flex cursor-pointer items-center gap-2 rounded-md p-2 hover:bg-muted/50",
                                isDocSelected && "bg-muted"
                              )}
                              onClick={() => handleDocumentToggle(doc.id)}
                            >
                              <Checkbox checked={isDocSelected} onChange={() => handleDocumentToggle(doc.id)} />
                              <FileText className="h-4 w-4 text-gray-500" />
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-sm font-medium">{doc.filename}</div>
                                <div className="text-xs text-muted-foreground">
                                  {getSystemMetadata(doc.system_metadata)?.file_size &&
                                    formatFileSize(getSystemMetadata(doc.system_metadata)!.file_size!)}
                                  {getSystemMetadata(doc.system_metadata)?.created_at && (
                                    <span className="ml-2">
                                      {new Date(
                                        getSystemMetadata(doc.system_metadata)!.created_at!
                                      ).toLocaleDateString()}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Empty state */}
                    {filteredItems.folders.length === 0 && filteredItems.documents.length === 0 && (
                      <div className="py-8 text-center text-muted-foreground">
                        {searchQuery ? "No documents or folders found" : "No documents available"}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Footer with selection summary */}
            {isSomeSelected && (
              <div className="border-t bg-muted/20 p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    {selectedFolders.length} folders, {selectedDocuments.length} documents selected
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      onDocumentSelectionChange([]);
                      onFolderSelectionChange([]);
                    }}
                  >
                    Clear Selection
                  </Button>
                </div>
              </div>
            )}
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
