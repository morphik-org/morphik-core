"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { PlusCircle, ArrowLeft, Trash2, Layers, Settings2, Plus, Eye, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
import { cn } from "@/lib/utils";
import DeleteConfirmationModal from "@/components/documents/DeleteConfirmationModal";

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

interface WorkflowStep {
  action_id: string;
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
  selectedDocuments = [],
  handleDeleteMultipleDocuments,
  uploadDialogComponent,
  onFolderCreate,
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [showNewFolderDialog, setShowNewFolderDialog] = React.useState(false);
  const [newFolderName, setNewFolderName] = React.useState("");
  const [newFolderDescription, setNewFolderDescription] = React.useState("");
  const [isCreatingFolder, setIsCreatingFolder] = React.useState(false);
  const [folderWorkflows, setFolderWorkflows] = React.useState<
    { id: string; name: string; description?: string; steps?: WorkflowStep[] }[]
  >([]);
  const [loadingWorkflows, setLoadingWorkflows] = React.useState(false);
  const [showWorkflowDialog, setShowWorkflowDialog] = React.useState(false);
  const [availableWorkflows, setAvailableWorkflows] = React.useState<
    { id: string; name: string; description?: string; steps?: WorkflowStep[] }[]
  >([]);
  const [showAddWorkflowDialog, setShowAddWorkflowDialog] = React.useState(false);
  const [selectedWorkflowToAdd, setSelectedWorkflowToAdd] = React.useState<string>("");

  // Delete confirmation state
  const [showDeleteModal, setShowDeleteModal] = React.useState(false);
  const [folderToDelete, setFolderToDelete] = React.useState<string | null>(null);
  const [isDeletingFolder, setIsDeletingFolder] = React.useState(false);

  // Function to update both state and URL
  const updateSelectedFolder = (folderName: string | null) => {
    setSelectedFolder(folderName);

    // If we're on the workflows page, navigate back to documents
    if (pathname === "/workflows") {
      if (folderName) {
        router.push(`/?folder=${encodeURIComponent(folderName)}`);
      } else {
        router.push("/");
      }
    } else {
      // Update URL to reflect the selected folder
      if (folderName) {
        router.push(`${pathname}?folder=${encodeURIComponent(folderName)}`);
      } else {
        router.push(pathname);
      }
    }
  };

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
  }, [folderToDelete, apiBaseUrl, authToken, refreshAction, selectedFolder, updateSelectedFolder]);

  // Fetch workflows for the selected folder
  const fetchFolderWorkflows = React.useCallback(
    async (folderId: string) => {
      setLoadingWorkflows(true);
      try {
        const response = await fetch(`${apiBaseUrl}/folders/${folderId}/workflows`, {
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });

        if (response.ok) {
          const workflows = await response.json();
          setFolderWorkflows(workflows);
        }
      } catch (error) {
        console.error("Failed to fetch folder workflows:", error);
        setFolderWorkflows([]);
      } finally {
        setLoadingWorkflows(false);
      }
    },
    [apiBaseUrl, authToken]
  );

  // Fetch workflows when a folder is selected
  React.useEffect(() => {
    if (selectedFolder && selectedFolder !== "all") {
      // Find the folder ID from the folder name
      const folder = folders.find(f => f.name === selectedFolder);
      if (folder) {
        fetchFolderWorkflows(folder.id);
      }
    } else {
      setFolderWorkflows([]);
    }
  }, [selectedFolder, folders, fetchFolderWorkflows]);

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
            <div className="flex items-center gap-3">
              {selectedFolder === "all" ? (
                <span className="text-3xl" aria-hidden="true">
                  📄
                </span>
              ) : (
                <Image src="/icons/folder-icon.png" alt="Folder" width={32} height={32} />
              )}
              <h2 className="text-xl font-medium">{selectedFolder === "all" ? "All Documents" : selectedFolder}</h2>

              {/* Workflows button for non-"all" folders */}
              {selectedFolder !== "all" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const folder = folders.find(f => f.name === selectedFolder);
                    if (folder) {
                      fetchFolderWorkflows(folder.id);
                      setShowWorkflowDialog(true);
                    }
                  }}
                  className="ml-2 flex items-center gap-2"
                >
                  <Layers className="h-4 w-4" />
                  <span>Workflows</span>
                  {folderWorkflows.length > 0 && (
                    <Badge variant="secondary" className="ml-1 px-1.5 py-0.5 text-xs">
                      {folderWorkflows.length}
                    </Badge>
                  )}
                </Button>
              )}
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

        {/* Workflow Management Dialog - Also show when viewing a folder */}
        <Dialog open={showWorkflowDialog} onOpenChange={setShowWorkflowDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Folder Workflows
              </DialogTitle>
              <DialogDescription>
                Manage workflows that automatically run when documents are added to this folder.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              {/* Current workflows */}
              {loadingWorkflows ? (
                <div className="flex items-center justify-center py-8">
                  <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-primary"></div>
                </div>
              ) : folderWorkflows.length > 0 ? (
                <div className="space-y-3">
                  <h4 className="text-sm font-medium">Active Workflows</h4>
                  {folderWorkflows.map(workflow => (
                    <div
                      key={workflow.id}
                      className="flex cursor-pointer items-center justify-between rounded-lg border p-3 transition-colors hover:bg-muted/50"
                      onClick={() => {
                        // Navigate to workflow detail page, stay on current page if already on workflows
                        if (pathname === "/workflows") {
                          router.push(`/workflows?id=${workflow.id}`);
                        } else {
                          router.push(`/?section=workflows&id=${workflow.id}`);
                        }
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <div className="rounded-md bg-primary/10 p-2">
                          <Layers className="h-4 w-4 text-primary" />
                        </div>
                        <div>
                          <p className="font-medium">{workflow.name}</p>
                          {workflow.description && (
                            <p className="text-sm text-muted-foreground">{workflow.description}</p>
                          )}
                          <div className="mt-1 flex items-center gap-2">
                            <Badge variant="secondary" className="text-xs">
                              {workflow.steps?.length || 0} steps
                            </Badge>
                            {workflow.steps?.map((step, idx: number) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {step.action_id?.split(".").pop()?.replace(/_/g, " ")}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={e => {
                            e.stopPropagation();
                            window.location.href = `/workflows?id=${workflow.id}`;
                          }}
                          className="h-8 w-8"
                          title="View workflow details"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={async e => {
                            e.stopPropagation();
                            const folder = folders.find(f => f.name === selectedFolder);
                            if (folder && window.confirm(`Remove "${workflow.name}" from this folder?`)) {
                              try {
                                const response = await fetch(
                                  `${apiBaseUrl}/folders/${folder.id}/workflows/${workflow.id}`,
                                  {
                                    method: "DELETE",
                                    headers: {
                                      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                                    },
                                  }
                                );
                                if (response.ok) {
                                  fetchFolderWorkflows(folder.id);
                                }
                              } catch (error) {
                                console.error("Failed to remove workflow:", error);
                              }
                            }
                          }}
                          className="h-8 w-8 text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-dashed p-8 text-center">
                  <Layers className="mx-auto h-12 w-12 text-muted-foreground/50" />
                  <p className="mt-2 text-sm text-muted-foreground">No workflows associated with this folder yet.</p>
                </div>
              )}

              {/* Add workflow button */}
              <div className="flex items-center justify-between border-t pt-4">
                <p className="text-sm text-muted-foreground">
                  Add workflows to automatically process documents in this folder.
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={async () => {
                      // Fetch available workflows
                      try {
                        const response = await fetch(`${apiBaseUrl}/workflows`, {
                          headers: {
                            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                          },
                        });
                        if (response.ok) {
                          const workflows = await response.json();
                          setAvailableWorkflows(workflows);
                          setShowAddWorkflowDialog(true);
                        }
                      } catch (error) {
                        console.error("Failed to fetch workflows:", error);
                      }
                    }}
                    className="flex items-center gap-2"
                  >
                    <Plus className="h-4 w-4" />
                    Add Workflow
                  </Button>
                  <Button
                    onClick={() => {
                      window.location.href = "/workflows";
                    }}
                    className="flex items-center gap-2"
                  >
                    <Settings2 className="h-4 w-4" />
                    Manage Workflows
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Add Workflow Dialog - Also show when viewing a folder */}
        <Dialog open={showAddWorkflowDialog} onOpenChange={setShowAddWorkflowDialog}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Add Workflow to Folder</DialogTitle>
              <DialogDescription>
                Select a workflow to automatically run when documents are added to &quot;{selectedFolder}&quot;.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              {/* Available workflows */}
              <div className="max-h-[300px] space-y-2 overflow-y-auto">
                {availableWorkflows.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    No workflows available. Create workflows first.
                  </div>
                ) : (
                  availableWorkflows
                    .filter(workflow => !folderWorkflows.some(fw => fw.id === workflow.id))
                    .map(workflow => (
                      <div
                        key={workflow.id}
                        className={cn(
                          "flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-all",
                          selectedWorkflowToAdd === workflow.id
                            ? "border-primary bg-primary/5"
                            : "hover:border-primary/50 hover:bg-muted/50"
                        )}
                        onClick={() => setSelectedWorkflowToAdd(workflow.id)}
                      >
                        <div className="rounded-md bg-primary/10 p-2">
                          <Layers className="h-4 w-4 text-primary" />
                        </div>
                        <div className="flex-1">
                          <p className="font-medium">{workflow.name}</p>
                          {workflow.description && (
                            <p className="text-sm text-muted-foreground">{workflow.description}</p>
                          )}
                          <Badge variant="secondary" className="mt-1 text-xs">
                            {workflow.steps?.length || 0} steps
                          </Badge>
                        </div>
                      </div>
                    ))
                )}
              </div>

              {/* Action buttons */}
              <div className="flex justify-end gap-2 border-t pt-4">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowAddWorkflowDialog(false);
                    setSelectedWorkflowToAdd("");
                  }}
                >
                  Cancel
                </Button>
                <Button
                  disabled={!selectedWorkflowToAdd}
                  onClick={async () => {
                    if (selectedWorkflowToAdd && selectedFolder) {
                      const folder = folders.find(f => f.name === selectedFolder);
                      if (folder) {
                        try {
                          const response = await fetch(
                            `${apiBaseUrl}/folders/${folder.id}/workflows/${selectedWorkflowToAdd}`,
                            {
                              method: "POST",
                              headers: {
                                ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                              },
                            }
                          );
                          if (response.ok) {
                            // Refresh folder workflows
                            await fetchFolderWorkflows(folder.id);
                            setShowAddWorkflowDialog(false);
                            setSelectedWorkflowToAdd("");
                          }
                        } catch (error) {
                          console.error("Failed to add workflow:", error);
                        }
                      }
                    }
                  }}
                >
                  Add Workflow
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
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
                setFolderToDelete(folder.name);
                setShowDeleteModal(true);
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

      {/* Workflow Management Dialog */}
      <Dialog open={showWorkflowDialog} onOpenChange={setShowWorkflowDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Folder Workflows
            </DialogTitle>
            <DialogDescription>
              Manage workflows that automatically run when documents are added to this folder.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* Current workflows */}
            {loadingWorkflows ? (
              <div className="flex items-center justify-center py-8">
                <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-primary"></div>
              </div>
            ) : folderWorkflows.length > 0 ? (
              <div className="space-y-3">
                <h4 className="text-sm font-medium">Active Workflows</h4>
                {folderWorkflows.map(workflow => (
                  <div
                    key={workflow.id}
                    className="flex cursor-pointer items-center justify-between rounded-lg border p-3 transition-colors hover:bg-muted/50"
                    onClick={() => {
                      // Navigate to workflow detail page
                      window.location.href = `/workflows?id=${workflow.id}`;
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div className="rounded-md bg-primary/10 p-2">
                        <Layers className="h-4 w-4 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{workflow.name}</p>
                        {workflow.description && (
                          <p className="text-sm text-muted-foreground">{workflow.description}</p>
                        )}
                        <div className="mt-1 flex items-center gap-2">
                          <Badge variant="secondary" className="text-xs">
                            {workflow.steps?.length || 0} steps
                          </Badge>
                          {workflow.steps?.map((step: WorkflowStep, idx: number) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {step.action_id?.split(".").pop()?.replace(/_/g, " ")}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={e => {
                          e.stopPropagation();
                          if (pathname === "/workflows") {
                            router.push(`/workflows?id=${workflow.id}`);
                          } else {
                            router.push(`/?section=workflows&id=${workflow.id}`);
                          }
                        }}
                        className="h-8 w-8"
                        title="View workflow details"
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={async e => {
                          e.stopPropagation();
                          const folder = folders.find(f => f.name === selectedFolder);
                          if (folder && window.confirm(`Remove "${workflow.name}" from this folder?`)) {
                            try {
                              const response = await fetch(
                                `${apiBaseUrl}/folders/${folder.id}/workflows/${workflow.id}`,
                                {
                                  method: "DELETE",
                                  headers: {
                                    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                                  },
                                }
                              );
                              if (response.ok) {
                                fetchFolderWorkflows(folder.id);
                              }
                            } catch (error) {
                              console.error("Failed to remove workflow:", error);
                            }
                          }
                        }}
                        className="h-8 w-8 text-destructive hover:text-destructive"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-dashed p-8 text-center">
                <Layers className="mx-auto h-12 w-12 text-muted-foreground/50" />
                <p className="mt-2 text-sm text-muted-foreground">No workflows associated with this folder yet.</p>
              </div>
            )}

            {/* Add workflow button */}
            <div className="flex items-center justify-between border-t pt-4">
              <p className="text-sm text-muted-foreground">
                Add workflows to automatically process documents in this folder.
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={async () => {
                    // Fetch available workflows
                    try {
                      const response = await fetch(`${apiBaseUrl}/workflows`, {
                        headers: {
                          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                        },
                      });
                      if (response.ok) {
                        const workflows = await response.json();
                        setAvailableWorkflows(workflows);
                        setShowAddWorkflowDialog(true);
                      }
                    } catch (error) {
                      console.error("Failed to fetch workflows:", error);
                    }
                  }}
                  className="flex items-center gap-2"
                >
                  <Plus className="h-4 w-4" />
                  Add Workflow
                </Button>
                <Button
                  onClick={() => {
                    if (pathname === "/workflows") {
                      setShowWorkflowDialog(false);
                    } else {
                      router.push("/workflows");
                    }
                  }}
                  className="flex items-center gap-2"
                >
                  <Settings2 className="h-4 w-4" />
                  Manage Workflows
                </Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Add Workflow Dialog */}
      <Dialog open={showAddWorkflowDialog} onOpenChange={setShowAddWorkflowDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Add Workflow to Folder</DialogTitle>
            <DialogDescription>
              Select a workflow to automatically run when documents are added to &quot;{selectedFolder}&quot;.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* Available workflows */}
            <div className="max-h-[300px] space-y-2 overflow-y-auto">
              {availableWorkflows.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">
                  No workflows available. Create workflows first.
                </div>
              ) : (
                availableWorkflows
                  .filter(workflow => !folderWorkflows.some(fw => fw.id === workflow.id))
                  .map(workflow => (
                    <div
                      key={workflow.id}
                      className={cn(
                        "flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-all",
                        selectedWorkflowToAdd === workflow.id
                          ? "border-primary bg-primary/5"
                          : "hover:border-primary/50 hover:bg-muted/50"
                      )}
                      onClick={() => setSelectedWorkflowToAdd(workflow.id)}
                    >
                      <div className="rounded-md bg-primary/10 p-2">
                        <Layers className="h-4 w-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium">{workflow.name}</p>
                        {workflow.description && (
                          <p className="text-sm text-muted-foreground">{workflow.description}</p>
                        )}
                        <Badge variant="secondary" className="mt-1 text-xs">
                          {workflow.steps?.length || 0} steps
                        </Badge>
                      </div>
                    </div>
                  ))
              )}
            </div>

            {/* Action buttons */}
            <div className="flex justify-end gap-2 border-t pt-4">
              <Button
                variant="outline"
                onClick={() => {
                  setShowAddWorkflowDialog(false);
                  setSelectedWorkflowToAdd("");
                }}
              >
                Cancel
              </Button>
              <Button
                disabled={!selectedWorkflowToAdd}
                onClick={async () => {
                  if (selectedWorkflowToAdd && selectedFolder) {
                    const folder = folders.find(f => f.name === selectedFolder);
                    if (folder) {
                      try {
                        const response = await fetch(
                          `${apiBaseUrl}/folders/${folder.id}/workflows/${selectedWorkflowToAdd}`,
                          {
                            method: "POST",
                            headers: {
                              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                            },
                          }
                        );
                        if (response.ok) {
                          // Refresh folder workflows
                          await fetchFolderWorkflows(folder.id);
                          setShowAddWorkflowDialog(false);
                          setSelectedWorkflowToAdd("");
                        }
                      } catch (error) {
                        console.error("Failed to add workflow:", error);
                      }
                    }
                  }
                }}
              >
                Add Workflow
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={showDeleteModal}
        onClose={() => {
          setShowDeleteModal(false);
          setFolderToDelete(null);
        }}
        onConfirm={handleDeleteFolder}
        itemName={folderToDelete || undefined}
        loading={isDeletingFolder}
      />
    </div>
  );
});

export default FolderList;
