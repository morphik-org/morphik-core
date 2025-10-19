import { useState, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
export const useFolderNavigation = (setSelectedFolder: (folder: string | null) => void) => {
  const router = useRouter();
  const pathname = usePathname();

  const updateSelectedFolder = useCallback(
    (folderName: string | null) => {
      setSelectedFolder(folderName);

      if (folderName) {
        router.push(`${pathname || "/"}?folder=${encodeURIComponent(folderName)}`);
      } else {
        router.push(pathname || "/");
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
