"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Layers, Plus, Settings2, Eye, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useRouter, usePathname } from "next/navigation";

interface WorkflowStep {
  action_id: string;
}

interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps?: WorkflowStep[];
}

interface WorkflowDialogsProps {
  showWorkflowDialog: boolean;
  setShowWorkflowDialog: (show: boolean) => void;
  showAddWorkflowDialog: boolean;
  setShowAddWorkflowDialog: (show: boolean) => void;
  folderWorkflows: Workflow[];
  loadingWorkflows: boolean;
  availableWorkflows: Workflow[];
  selectedWorkflowToAdd: string;
  setSelectedWorkflowToAdd: (id: string) => void;
  selectedFolder: string | null;
  apiBaseUrl: string;
  authToken: string | null;
  onFetchFolderWorkflows: (folderId: string) => void;
  onFetchAvailableWorkflows: () => void;
  onAddWorkflow: (workflowId: string, folderId: string) => Promise<void>;
  onRemoveWorkflow: (workflowId: string, folderId: string) => Promise<void>;
  folders: Array<{ id: string; name: string }>;
}

const WorkflowDialogs: React.FC<WorkflowDialogsProps> = ({
  showWorkflowDialog,
  setShowWorkflowDialog,
  showAddWorkflowDialog,
  setShowAddWorkflowDialog,
  folderWorkflows,
  loadingWorkflows,
  availableWorkflows,
  selectedWorkflowToAdd,
  setSelectedWorkflowToAdd,
  selectedFolder,
  onFetchAvailableWorkflows,
  onAddWorkflow,
  onRemoveWorkflow,
  folders,
}) => {
  const router = useRouter();
  const pathname = usePathname();

  const currentFolder = folders.find(f => f.name === selectedFolder);

  return (
    <>
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
                      // Get the app ID from the current path if in cloud context
                      const pathSegments = (pathname || "").split("/").filter(Boolean);
                      const isCloudUI =
                        pathSegments.length > 1 && !["dashboard", "login", "signup"].includes(pathSegments[0]);

                      if (isCloudUI) {
                        // In cloud UI, route to /app_id/workflows
                        const appId = pathSegments[0];
                        router.push(`/${appId}/workflows?id=${workflow.id}`);
                      } else if (pathname === "/workflows") {
                        // In standalone, already on workflows page
                        router.push(`/workflows?id=${workflow.id}`);
                      } else {
                        // In standalone, not on workflows page
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
                          const pathSegments = (pathname || "").split("/").filter(Boolean);
                          const isCloudUI =
                            pathSegments.length > 1 && !["dashboard", "login", "signup"].includes(pathSegments[0]);

                          if (isCloudUI) {
                            const appId = pathSegments[0];
                            router.push(`/${appId}/workflows?id=${workflow.id}`);
                          } else if (pathname === "/workflows") {
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
                          if (currentFolder && window.confirm(`Remove "${workflow.name}" from this folder?`)) {
                            await onRemoveWorkflow(workflow.id, currentFolder.id);
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

            <div className="flex items-center justify-between border-t pt-4">
              <p className="text-sm text-muted-foreground">
                Add workflows to automatically process documents in this folder.
              </p>
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={onFetchAvailableWorkflows} className="flex items-center gap-2">
                  <Plus className="h-4 w-4" />
                  Add Workflow
                </Button>
                <Button
                  onClick={() => {
                    const pathSegments = (pathname || "").split("/").filter(Boolean);
                    const isCloudUI =
                      pathSegments.length > 1 && !["dashboard", "login", "signup"].includes(pathSegments[0]);

                    if (isCloudUI) {
                      const appId = pathSegments[0];
                      router.push(`/${appId}/workflows`);
                    } else if (pathname === "/workflows") {
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
                  if (selectedWorkflowToAdd && currentFolder) {
                    await onAddWorkflow(selectedWorkflowToAdd, currentFolder.id);
                    setShowAddWorkflowDialog(false);
                    setSelectedWorkflowToAdd("");
                  }
                }}
              >
                Add Workflow
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default WorkflowDialogs;
