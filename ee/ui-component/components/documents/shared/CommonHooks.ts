import { useState, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import { FolderSummary } from "@/components/types";

interface WorkflowStep {
  action_id: string;
}

interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps?: WorkflowStep[];
}

export const useWorkflowManagement = (
  apiBaseUrl: string,
  authToken: string | null,
  selectedFolder: string | null,
  folders: FolderSummary[]
) => {
  const [folderWorkflows, setFolderWorkflows] = useState<Workflow[]>([]);
  const [loadingWorkflows, setLoadingWorkflows] = useState(false);
  const [showWorkflowDialog, setShowWorkflowDialog] = useState(false);
  const [availableWorkflows, setAvailableWorkflows] = useState<Workflow[]>([]);
  const [showAddWorkflowDialog, setShowAddWorkflowDialog] = useState(false);
  const [selectedWorkflowToAdd, setSelectedWorkflowToAdd] = useState<string>("");

  const fetchFolderWorkflows = useCallback(
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

  const fetchAvailableWorkflows = useCallback(async () => {
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
  }, [apiBaseUrl, authToken]);

  const addWorkflow = useCallback(
    async (workflowId: string, folderId: string) => {
      try {
        const response = await fetch(`${apiBaseUrl}/folders/${folderId}/workflows/${workflowId}`, {
          method: "POST",
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        if (response.ok) {
          await fetchFolderWorkflows(folderId);
        }
      } catch (error) {
        console.error("Failed to add workflow:", error);
      }
    },
    [apiBaseUrl, authToken, fetchFolderWorkflows]
  );

  const removeWorkflow = useCallback(
    async (workflowId: string, folderId: string) => {
      try {
        const response = await fetch(`${apiBaseUrl}/folders/${folderId}/workflows/${workflowId}`, {
          method: "DELETE",
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });
        if (response.ok) {
          fetchFolderWorkflows(folderId);
        }
      } catch (error) {
        console.error("Failed to remove workflow:", error);
      }
    },
    [apiBaseUrl, authToken, fetchFolderWorkflows]
  );

  const openWorkflowDialog = useCallback(() => {
    const folder = folders.find(f => f.name === selectedFolder);
    if (folder) {
      fetchFolderWorkflows(folder.id);
      setShowWorkflowDialog(true);
    }
  }, [folders, selectedFolder, fetchFolderWorkflows]);

  return {
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
    openWorkflowDialog,
  };
};

export const useFolderNavigation = (setSelectedFolder: (folder: string | null) => void) => {
  const router = useRouter();
  const pathname = usePathname();

  const updateSelectedFolder = useCallback(
    (folderName: string | null) => {
      setSelectedFolder(folderName);

      if (pathname === "/workflows") {
        if (folderName) {
          router.push(`/?folder=${encodeURIComponent(folderName)}`);
        } else {
          router.push("/");
        }
      } else {
        if (folderName) {
          router.push(`${pathname}?folder=${encodeURIComponent(folderName)}`);
        } else {
          router.push(pathname);
        }
      }
    },
    [pathname, router, setSelectedFolder]
  );

  return { updateSelectedFolder };
};

export const useDeleteConfirmation = () => {
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<string | null>(null);
  const [isDeletingItem, setIsDeletingItem] = useState(false);

  const openDeleteModal = useCallback((itemName: string) => {
    setItemToDelete(itemName);
    setShowDeleteModal(true);
  }, []);

  const closeDeleteModal = useCallback(() => {
    setShowDeleteModal(false);
    setItemToDelete(null);
  }, []);

  return {
    showDeleteModal,
    setShowDeleteModal,
    itemToDelete,
    setItemToDelete,
    isDeletingItem,
    setIsDeletingItem,
    openDeleteModal,
    closeDeleteModal,
  };
};
