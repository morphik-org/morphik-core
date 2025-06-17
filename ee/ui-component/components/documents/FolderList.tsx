"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { PlusCircle, ArrowLeft, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { FolderSummary } from "@/components/types";
import { useRouter, usePathname } from "next/navigation";
import Image from "next/image";

interface FolderListProps {
  folders: FolderSummary[];
  selectedFolder: string | null;
  setSelectedFolder: (folderName: string | null) => void;
  apiBaseUrl: string;
  authToken: string | null;
  refreshFolders: () => void;
  loading: boolean;
  refreshAction?: () => void;
  selectedDocuments?: string[];
  handleDeleteMultipleDocuments?: () => void;
  showUploadDialog?: boolean;
  setShowUploadDialog?: (show: boolean) => void;
  uploadDialogComponent?: React.ReactNode;
  onFolderCreate?: (folderName: string) => void;
}

const FolderList: React.FC<FolderListProps> = ({
  folders,
  selectedFolder,
  setSelectedFolder,
  apiBaseUrl,
  authToken,
  refreshFolders,
  loading,
  refreshAction,
  selectedDocuments = [],
  handleDeleteMultipleDocuments,
  uploadDialogComponent,
  onFolderCreate,
}) => {
  const router = useRouter();
  const pathname = usePathname();
  const [showNewFolderDialog, setShowNewFolderDialog] = React.useState(false);
  const [newFolderName, setNewFolderName] = React.useState("");
  const [newFolderDescription, setNewFolderDescription] = React.useState("");
  const [isCreatingFolder, setIsCreatingFolder] = React.useState(false);

  // Function to update both state and URL
  const updateSelectedFolder = (folderName: string | null) => {
    setSelectedFolder(folderName);

    // Update URL to reflect the selected folder
    if (folderName) {
      router.push(`${pathname}?folder=${encodeURIComponent(folderName)}`);
    } else {
      router.push(pathname);
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;

    setIsCreatingFolder(true);

    try {
      console.log(`Creating folder: ${newFolderName}`);

      const response = await fetch(`${apiBaseUrl}/folders`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({
          name: newFolderName.trim(),
          description: newFolderDescription.trim() || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to create folder: ${response.statusText}`);
      }

      // Get the created folder data
      const folderData = await response.json();
      console.log(`Created folder with ID: ${folderData.id} and name: ${folderData.name}`);

      // Close dialog and reset form
      setShowNewFolderDialog(false);
      setNewFolderName("");
      setNewFolderDescription("");

      // Refresh folder list - use a fresh fetch
      await refreshFolders();

      // Auto-select this newly created folder so user can immediately add files to it
      // This ensures we start with a clean empty folder view
      updateSelectedFolder(folderData.name);

      console.log(`handleCreateFolder: Calling onFolderCreate with '${folderData.name}'`);
      onFolderCreate?.(folderData.name);
    } catch (error) {
      console.error("Error creating folder:", error);
    } finally {
      setIsCreatingFolder(false);
    }
  };

  // If we're viewing a specific folder or all documents, show back button and folder title
  if (selectedFolder !== null) {
    return (
      <div className="mb-4">
        <div className="flex items-center justify-between py-2">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="rounded-full hover:bg-muted/50"
              onClick={() => updateSelectedFolder(null)}
            >
              <ArrowLeft size={18} />
            </Button>
            <div className="flex items-center">
              {selectedFolder === "all" ? (
                <span className="mr-3 text-3xl" aria-hidden="true">
                  📄
                </span>
              ) : (
                <Image src="/icons/folder-icon.png" alt="Folder" width={32} height={32} className="mr-3" />
              )}
              <h2 className="text-xl font-medium">{selectedFolder === "all" ? "All Documents" : selectedFolder}</h2>
            </div>

            {/* Show action buttons if documents are selected */}
            {selectedDocuments && selectedDocuments.length > 0 && (
              <div className="ml-4 flex gap-2">
                {/* Delete button */}
                {handleDeleteMultipleDocuments && (
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleDeleteMultipleDocuments}
                    className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
                    title={`Delete ${selectedDocuments.length} selected document${selectedDocuments.length > 1 ? "s" : ""}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {refreshAction && (
              <Button variant="outline" onClick={refreshAction} className="flex items-center" title="Refresh documents">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="mr-1"
                >
                  <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                  <path d="M21 3v5h-5"></path>
                  <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                  <path d="M8 16H3v5"></path>
                </svg>
                Refresh
              </Button>
            )}

            {/* Upload dialog component */}
            {uploadDialogComponent}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6">
      <div className="mb-4 flex items-center justify-between">
        <Dialog open={showNewFolderDialog} onOpenChange={setShowNewFolderDialog}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <PlusCircle className="mr-2 h-4 w-4" /> New Folder
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Folder</DialogTitle>
              <DialogDescription>Create a new folder to organize your documents.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div>
                <Label htmlFor="folderName">Folder Name</Label>
                <Input
                  id="folderName"
                  value={newFolderName}
                  onChange={e => setNewFolderName(e.target.value)}
                  placeholder="Enter folder name"
                />
              </div>
              <div>
                <Label htmlFor="folderDescription">Description (Optional)</Label>
                <Textarea
                  id="folderDescription"
                  value={newFolderDescription}
                  onChange={e => setNewFolderDescription(e.target.value)}
                  placeholder="Enter folder description"
                  rows={3}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="ghost" onClick={() => setShowNewFolderDialog(false)} disabled={isCreatingFolder}>
                Cancel
              </Button>
              <Button onClick={handleCreateFolder} disabled={!newFolderName.trim() || isCreatingFolder}>
                {isCreatingFolder ? "Creating..." : "Create Folder"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <div className="flex items-center gap-2">
          {refreshAction && (
            <Button variant="outline" onClick={refreshAction} className="flex items-center" title="Refresh folders">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mr-1"
              >
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                <path d="M21 3v5h-5"></path>
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                <path d="M8 16H3v5"></path>
              </svg>
              Refresh
            </Button>
          )}
          {uploadDialogComponent}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 py-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        <div className="group flex cursor-pointer flex-col items-center" onClick={() => updateSelectedFolder("all")}>
          <div className="mb-2 flex h-16 w-16 items-center justify-center transition-transform group-hover:scale-110">
            <span className="text-4xl" aria-hidden="true">
              📄
            </span>
          </div>
          <span className="text-center text-sm font-medium transition-colors group-hover:text-primary">
            All Documents
          </span>
        </div>

        {folders.map(folder => (
          <div
            key={folder.name}
            className="group flex cursor-pointer flex-col items-center"
            onClick={() => updateSelectedFolder(folder.name)}
          >
            <div className="mb-2 flex h-16 w-16 items-center justify-center transition-transform group-hover:scale-110">
              <Image src="/icons/folder-icon.png" alt="Folder" width={64} height={64} className="object-contain" />
            </div>
            <span className="w-full max-w-[100px] truncate text-center text-sm font-medium transition-colors group-hover:text-primary">
              {folder.name}
            </span>
          </div>
        ))}
      </div>

      {folders.length === 0 && !loading && (
        <div className="mt-4 flex flex-col items-center justify-center p-8">
          <Image src="/icons/folder-icon.png" alt="Folder" width={80} height={80} className="mb-3 opacity-50" />
          <p className="text-sm text-muted-foreground">No folders yet. Create one to organize your documents.</p>
        </div>
      )}

      {loading && folders.length === 0 && (
        <div className="mt-4 flex items-center justify-center p-8">
          <div className="flex items-center space-x-2">
            <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-primary"></div>
            <p className="text-sm text-muted-foreground">Loading folders...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FolderList;
