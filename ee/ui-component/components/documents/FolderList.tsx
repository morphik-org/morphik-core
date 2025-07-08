"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { PlusCircle, X } from "lucide-react";
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
import { FolderSummary, Document } from "@/components/types";
import Image from "next/image";
import DeleteConfirmationModal from "@/components/documents/DeleteConfirmationModal";
import WorkflowDialogs from "./shared/WorkflowDialogs";
import { EmptyFolders } from "./shared/EmptyStates";
import { useWorkflowManagement, useFolderNavigation, useDeleteConfirmation } from "./shared/CommonHooks";

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
  unorganizedDocuments?: Document[];
  onDocumentClick?: (document: Document) => void;
}

const FolderList: React.FC<FolderListProps> = React.memo(function FolderList({
  folders,
  selectedFolder,
  setSelectedFolder,
  apiBaseUrl,
  authToken,
  refreshFolders,
  loading,
  refreshAction,
  uploadDialogComponent,
  onFolderCreate,
  unorganizedDocuments = [],
  onDocumentClick,
}) {
  const [showNewFolderDialog, setShowNewFolderDialog] = React.useState(false);
  const [newFolderName, setNewFolderName] = React.useState("");
  const [newFolderDescription, setNewFolderDescription] = React.useState("");
  const [isCreatingFolder, setIsCreatingFolder] = React.useState(false);

  // Use shared hooks
  const { updateSelectedFolder } = useFolderNavigation(setSelectedFolder);
  const {
    folderWorkflows,
    loadingWorkflows,
    showWorkflowDialog,
    setShowWorkflowDialog,
    availableWorkflows,
    showAddWorkflowDialog,
    setShowAddWorkflowDialog,
    selectedWorkflowToAdd,
    setSelectedWorkflowToAdd,
    fetchFolderWorkflows,
    fetchAvailableWorkflows,
    addWorkflow,
    removeWorkflow,
  } = useWorkflowManagement(apiBaseUrl, authToken, selectedFolder, folders);

  const {
    showDeleteModal,
    setShowDeleteModal,
    itemToDelete: folderToDelete,
    setItemToDelete: setFolderToDelete,
    isDeletingItem: isDeletingFolder,
    setIsDeletingItem: setIsDeletingFolder,
    openDeleteModal,
    closeDeleteModal,
  } = useDeleteConfirmation();

  // Handle folder deletion
  const handleDeleteFolder = React.useCallback(async () => {
    if (!folderToDelete) return;

    setIsDeletingFolder(true);
    try {
      const response = await fetch(`${apiBaseUrl}/folders/${folderToDelete}`, {
        method: "DELETE",
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });

      if (response.ok) {
        // Refresh folders list
        if (refreshAction) {
          refreshAction();
        }
        // If the deleted folder was selected, switch to "all"
        if (selectedFolder === folderToDelete) {
          updateSelectedFolder("all");
        }
      } else {
        const error = await response.text();
        alert(`Failed to delete folder: ${error}`);
      }
    } catch (error) {
      console.error("Failed to delete folder:", error);
      alert("Failed to delete folder. Please try again.");
    } finally {
      setIsDeletingFolder(false);
      setShowDeleteModal(false);
      setFolderToDelete(null);
    }
  }, [
    folderToDelete,
    apiBaseUrl,
    authToken,
    refreshAction,
    selectedFolder,
    updateSelectedFolder,
    setFolderToDelete,
    setIsDeletingFolder,
    setShowDeleteModal,
  ]);

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
      refreshFolders();

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

  // If we're viewing a specific folder, only show workflow dialogs
  if (selectedFolder !== null) {
    return (
      <WorkflowDialogs
        showWorkflowDialog={showWorkflowDialog}
        setShowWorkflowDialog={setShowWorkflowDialog}
        showAddWorkflowDialog={showAddWorkflowDialog}
        setShowAddWorkflowDialog={setShowAddWorkflowDialog}
        folderWorkflows={folderWorkflows}
        loadingWorkflows={loadingWorkflows}
        availableWorkflows={availableWorkflows}
        selectedWorkflowToAdd={selectedWorkflowToAdd}
        setSelectedWorkflowToAdd={setSelectedWorkflowToAdd}
        selectedFolder={selectedFolder}
        apiBaseUrl={apiBaseUrl}
        authToken={authToken}
        onFetchFolderWorkflows={fetchFolderWorkflows}
        onFetchAvailableWorkflows={fetchAvailableWorkflows}
        onAddWorkflow={addWorkflow}
        onRemoveWorkflow={removeWorkflow}
        folders={folders}
      />
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

        {/* Render unorganized documents */}
        {unorganizedDocuments.map(document => (
          <div
            key={document.external_id}
            className="group flex cursor-pointer flex-col items-center"
            onClick={() => onDocumentClick?.(document)}
          >
            <div className="mb-2 flex h-16 w-16 items-center justify-center transition-transform group-hover:scale-110">
              <span className="text-4xl" aria-hidden="true">
                📄
              </span>
            </div>
            <span className="w-full max-w-[100px] truncate text-center text-sm font-medium transition-colors group-hover:text-primary">
              {document.filename || document.external_id}
            </span>
          </div>
        ))}

        {folders.map(folder => (
          <div
            key={folder.name}
            className="group relative flex cursor-pointer flex-col items-center"
            onClick={() => updateSelectedFolder(folder.name)}
          >
            {/* Delete button */}
            <Button
              variant="ghost"
              size="icon"
              className="absolute -right-2 -top-2 z-10 h-6 w-6 rounded-full bg-background opacity-0 shadow-sm transition-opacity hover:bg-destructive hover:text-destructive-foreground group-hover:opacity-100"
              onClick={e => {
                e.stopPropagation();
                openDeleteModal(folder.name);
              }}
            >
              <X className="h-3 w-3" />
            </Button>
            <div className="mb-2 flex h-16 w-16 items-center justify-center transition-transform group-hover:scale-110">
              <Image src="/icons/folder-icon.png" alt="Folder" width={64} height={64} className="object-contain" />
            </div>
            <span className="w-full max-w-[100px] truncate text-center text-sm font-medium transition-colors group-hover:text-primary">
              {folder.name}
            </span>
          </div>
        ))}
      </div>

      {folders.length === 0 && <EmptyFolders loading={loading} />}

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={showDeleteModal}
        onClose={closeDeleteModal}
        onConfirm={handleDeleteFolder}
        itemName={folderToDelete || undefined}
        loading={isDeletingFolder}
      />
    </div>
  );
});

export default FolderList;
