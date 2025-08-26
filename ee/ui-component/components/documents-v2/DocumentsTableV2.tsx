"use client";

import React, { useState, useEffect } from "react";
import { useDocumentsV2, FileSystemItem } from "./DocumentsProviderV2";
import { DocumentRowV2 } from "./DocumentRowV2";
import { Checkbox } from "../ui/checkbox";
import { Loader2 } from "lucide-react";
import { Table, TableBody, TableHead, TableHeader, TableRow, TableCell } from "../ui/table";
import { Skeleton } from "../ui/skeleton";

export function DocumentsTableV2() {
  const {
    tableItems,
    selectedIds,
    setSelectedIds,
    setSelectedDocument,
    setCurrentFolder,
    isLoading,
    loadingStates,
    onFolderClick,
    onDocumentClick,
  } = useDocumentsV2();

  // Track if we're on the client to avoid hydration issues with Checkbox
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Handle select all
  const handleSelectAll = (checked: boolean | "indeterminate") => {
    if (checked === true) {
      // Select all items
      const allIds = new Set<string>();
      tableItems.forEach(item => {
        const id = item.type === "folder" ? item.data.id : item.data.external_id;
        allIds.add(id);
      });
      setSelectedIds(allIds);
    } else {
      // Deselect all
      setSelectedIds(new Set());
    }
  };

  // Toggle selection for a single item
  const toggleSelection = (id: string) => {
    const newSelection = new Set(selectedIds);
    if (newSelection.has(id)) {
      newSelection.delete(id);
    } else {
      newSelection.add(id);
    }
    setSelectedIds(newSelection);
  };

  // Handle item click
  const handleItemClick = (item: FileSystemItem) => {
    if (item.type === "folder") {
      // Navigate into folder
      // Backend expects folder_name, which is the folder's name, not ID
      setCurrentFolder(item.data.name);
      if (onFolderClick) {
        onFolderClick(item.data.name);
      }
    } else {
      // Select document for detail view
      setSelectedDocument(item.data);
      if (onDocumentClick) {
        onDocumentClick(item.data.filename || item.data.external_id);
      }
    }
  };

  // Handle double click
  const handleItemDoubleClick = (item: FileSystemItem) => {
    if (item.type === "folder") {
      // Already handled by single click for folders
      return;
    } else {
      // Open document detail modal
      setSelectedDocument(item.data);
      // Trigger modal open
      const event = new CustomEvent("openDocumentDetail", { detail: item.data });
      window.dispatchEvent(event);
    }
  };

  // Determine checkbox state
  const allSelected = tableItems.length > 0 && selectedIds.size === tableItems.length;
  const someSelected = selectedIds.size > 0 && selectedIds.size < tableItems.length;

  // Don't render anything until client-side to avoid hydration issues
  if (!isClient) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Show skeleton loader when loading
  if (isLoading) {
    return (
      <div className="w-full">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">
                <Skeleton className="h-4 w-4" />
              </TableHead>
              <TableHead>Name</TableHead>
              <TableHead className="w-32">Type</TableHead>
              <TableHead className="w-40">Modified</TableHead>
              <TableHead className="w-32">Status</TableHead>
              <TableHead className="w-20 text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[...Array(5)].map((_, i) => (
              <TableRow key={i}>
                <TableCell className="w-12">
                  <Skeleton className="h-4 w-4" />
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Skeleton className="h-5 w-5" />
                    <Skeleton className="h-4 w-48" />
                  </div>
                </TableCell>
                <TableCell className="w-32">
                  <Skeleton className="h-4 w-16" />
                </TableCell>
                <TableCell className="w-40">
                  <Skeleton className="h-4 w-24" />
                </TableCell>
                <TableCell className="w-32">
                  <Skeleton className="h-4 w-20" />
                </TableCell>
                <TableCell className="w-20">
                  <Skeleton className="ml-auto h-8 w-8" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  }

  if (tableItems.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center text-muted-foreground">
        <p className="text-lg font-medium">No documents or folders</p>
        <p className="mt-2 text-sm">Upload documents or create folders to get started</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">
              <Checkbox checked={allSelected} onCheckedChange={handleSelectAll} aria-label="Select all items" />
            </TableHead>
            <TableHead>Name</TableHead>
            <TableHead className="w-32">Type</TableHead>
            <TableHead className="w-40">Modified</TableHead>
            <TableHead className="w-32">Status</TableHead>
            <TableHead className="w-20 text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {tableItems.map(item => {
            const id = item.type === "folder" ? item.data.id : item.data.external_id;
            const processingStatus = item.type === "document" ? loadingStates.get(item.data.external_id) : undefined;

            return (
              <DocumentRowV2
                key={id}
                item={item}
                selected={selectedIds.has(id)}
                onSelect={() => toggleSelection(id)}
                onClick={() => handleItemClick(item)}
                onDoubleClick={() => handleItemDoubleClick(item)}
                processingStatus={processingStatus}
              />
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
