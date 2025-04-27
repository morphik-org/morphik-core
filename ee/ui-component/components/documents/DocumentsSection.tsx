"use client";

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Upload } from 'lucide-react';
import { showAlert, removeAlert } from '@/components/ui/alert-system';
import DocumentList from './DocumentList';
import DocumentDetail from './DocumentDetail';
import FolderList from './FolderList';
import { UploadDialog, useUploadDialog } from './UploadDialog';
import { cn } from '@/lib/utils';

import { Document, Folder } from '@/components/types';

// Custom hook for drag and drop functionality
function useDragAndDrop({
  onDrop,
  disabled = false
}: {
  onDrop: (files: File[]) => void;
  disabled?: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, [disabled]);

  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, [disabled]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onDrop(files);
    }
  }, [disabled, onDrop]);

  return {
    isDragging,
    dragHandlers: {
      onDragOver: handleDragOver,
      onDragEnter: handleDragEnter,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop
    }
  };
}

interface DocumentsSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialFolder?: string | null;
  setSidebarCollapsed?: (collapsed: boolean) => void;
}

// Debug render counter
let renderCount = 0;

const DocumentsSection: React.FC<DocumentsSectionProps> = ({
  apiBaseUrl,
  authToken,
  initialFolder = null,
  setSidebarCollapsed
}) => {
  // Increment render counter for debugging
  renderCount++;
  console.log(`DocumentsSection rendered: #${renderCount}`);
  // Ensure apiBaseUrl is correctly formatted, especially for localhost
  const effectiveApiUrl = React.useMemo(() => {
    console.log('DocumentsSection: Input apiBaseUrl:', apiBaseUrl);
    // Check if it's a localhost URL and ensure it has the right format
    if (apiBaseUrl.includes('localhost') || apiBaseUrl.includes('127.0.0.1')) {
      if (!apiBaseUrl.includes('http')) {
        return `http://${apiBaseUrl}`;
      }
    }
    return apiBaseUrl;
  }, [apiBaseUrl]);

  // State for documents and folders
  const [documents, setDocuments] = useState<Document[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(initialFolder);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [foldersLoading, setFoldersLoading] = useState(false);
  // Use ref to track if this is the initial mount
  const isInitialMount = useRef(true);

  // Upload dialog state from custom hook
  const uploadDialogState = useUploadDialog();
  // Extract only the state variables we actually use in this component
  const {
    showUploadDialog,
    setShowUploadDialog,
    metadata,
    rules,
    useColpali,
    resetUploadDialog
  } = uploadDialogState;

  // Initialize drag and drop
  const { isDragging, dragHandlers } = useDragAndDrop({
    onDrop: (files) => {
      // Only allow drag and drop when inside a folder
      if (selectedFolder && selectedFolder !== null) {
        handleBatchFileUpload(files, true);
      }
    },
    disabled: !selectedFolder || selectedFolder === null
  });

  // No need for a separate header function, use authToken directly

  // Fetch all documents, optionally filtered by folder
  const fetchDocuments = useCallback(async (source: string = 'unknown') => {
    console.log(`fetchDocuments called from: ${source}, selectedFolder: ${selectedFolder}`);
    // Ensure API URL is valid before proceeding
    if (!effectiveApiUrl) {
        console.error('fetchDocuments: No valid API URL available.');
        setLoading(false);
        return;
    }

    // Immediately clear documents and set loading state if selectedFolder is null (folder grid view)
    if (selectedFolder === null) {
      console.log('fetchDocuments: No folder selected, clearing documents.');
      setDocuments([]);
      setLoading(false);
      return;
    }

    // Set loading state only for initial load or when explicitly changing folders
    if (documents.length === 0 || source === 'folders loaded or selectedFolder changed') {
      setLoading(true);
    }

    try {
      let documentsToFetch: Document[] = [];

      if (selectedFolder === "all") {
        // Fetch all documents for the "all" view
        console.log('fetchDocuments: Fetching all documents');
        const response = await fetch(`${effectiveApiUrl}/documents`, {
          method: 'POST', // Assuming POST is correct for fetching all
          headers: {
            'Content-Type': 'application/json',
            ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
          },
          body: JSON.stringify({}) // Empty body for all documents
        });
        if (!response.ok) {
          throw new Error(`Failed to fetch all documents: ${response.statusText}`);
        }
        documentsToFetch = await response.json();
        console.log(`fetchDocuments: Fetched ${documentsToFetch.length} total documents`);

      } else {
        // Fetch documents for a specific folder
        console.log(`fetchDocuments: Fetching documents for folder: ${selectedFolder}`);
        const targetFolder = folders.find(folder => folder.name === selectedFolder);

        if (targetFolder && Array.isArray(targetFolder.document_ids) && targetFolder.document_ids.length > 0) {
          // Folder found and has documents, fetch them by ID
          console.log(`fetchDocuments: Folder found with ${targetFolder.document_ids.length} IDs. Fetching details...`);
          const response = await fetch(`${effectiveApiUrl}/batch/documents`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
            },
            body: JSON.stringify({ document_ids: targetFolder.document_ids })
          });
          if (!response.ok) {
            throw new Error(`Failed to fetch batch documents: ${response.statusText}`);
          }
          documentsToFetch = await response.json();
          console.log(`fetchDocuments: Fetched details for ${documentsToFetch.length} documents`);
        } else {
          // Folder not found, or folder is empty
          if (targetFolder) {
            console.log(`fetchDocuments: Folder ${selectedFolder} found but is empty.`);
          } else {
            console.log(`fetchDocuments: Folder ${selectedFolder} not found in current state.`);
          }
          // In either case, the folder contains no documents to display
          documentsToFetch = [];
        }
      }

      // Process fetched documents (add status if needed)
      const processedData = documentsToFetch.map((doc: Document) => {
        if (!doc.system_metadata) {
          doc.system_metadata = {};
        }
        if (!doc.system_metadata.status && doc.system_metadata.folder_name) {
          doc.system_metadata.status = "processing";
        }
        return doc;
      });

      // Update state
      setDocuments(processedData);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      console.error(`Error in fetchDocuments (${source}): ${errorMsg}`);
      showAlert(errorMsg, {
        type: 'error',
        title: 'Error Fetching Documents',
        duration: 5000
      });
      // Clear documents on error to avoid showing stale/incorrect data
      setDocuments([]);
    } finally {
      // Always ensure loading state is turned off
      setLoading(false);
    }
  // Dependencies: URL, auth, selected folder, and the folder list itself
  }, [effectiveApiUrl, authToken, selectedFolder, folders, documents.length]);

  // Fetch all folders
  const fetchFolders = useCallback(async () => {
    console.log('fetchFolders called');
    setFoldersLoading(true);
    try {
      const response = await fetch(`${effectiveApiUrl}/folders`, {
        method: 'GET',
        headers: {
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        }
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch folders: ${response.statusText}`);
      }
      const data = await response.json();
      console.log(`Fetched ${data.length} folders`);
      setFolders(data);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      console.error(`Folder fetch error: ${errorMsg}`);
      showAlert(errorMsg, {
        type: 'error',
        title: 'Error',
        duration: 5000
      });
    } finally {
      setFoldersLoading(false);
    }
  }, [effectiveApiUrl, authToken]);

  // Function to refresh documents based on current folder state
  const refreshDocuments = useCallback(async () => {
    await fetchDocuments('refreshDocuments call');
  }, [fetchDocuments]);

  // Fetch folders initially
  useEffect(() => {
    console.log('DocumentsSection: Initial folder fetch');
    fetchFolders();
  }, [fetchFolders]);

  // Fetch documents when folders are loaded or selectedFolder changes
  useEffect(() => {
    if (!foldersLoading && folders.length > 0) {
      // Avoid fetching documents on initial mount if selectedFolder is null
      // unless initialFolder was specified
      if (isInitialMount.current && selectedFolder === null && !initialFolder) {
        console.log('Initial mount with no folder selected, skipping document fetch');
        isInitialMount.current = false;
        return;
      }
      console.log('Folders loaded or selectedFolder changed, fetching documents...', selectedFolder);
      fetchDocuments('folders loaded or selectedFolder changed');
      isInitialMount.current = false; // Mark initial mount as complete
    } else if (!foldersLoading && folders.length === 0 && selectedFolder === null) {
        // Handle case where there are no folders at all
        console.log('No folders found, clearing documents and stopping loading.');
        setDocuments([]);
        setLoading(false);
        isInitialMount.current = false;
    }
  }, [foldersLoading, folders, selectedFolder, fetchDocuments, initialFolder]);

  // Poll for document status if any document is processing
  useEffect(() => {
    const hasProcessing = documents.some(
      (doc) => doc.system_metadata?.status === 'processing'
    );

    if (hasProcessing) {
      console.log('Polling for document status...');
      const intervalId = setInterval(() => {
        console.log('Polling interval: calling refreshDocuments');
        refreshDocuments(); // Fetch documents again to check status
      }, 5000); // Poll every 5 seconds

      // Cleanup function to clear the interval when the component unmounts
      // or when there are no more processing documents
      return () => {
        console.log('Clearing polling interval');
        clearInterval(intervalId);
      };
    }
  }, [documents, refreshDocuments]);

  // Collapse sidebar when a folder is selected
  useEffect(() => {
    if (selectedFolder !== null && setSidebarCollapsed) {
      setSidebarCollapsed(true);
    } else if (setSidebarCollapsed) {
      setSidebarCollapsed(false);
    }
  }, [selectedFolder, setSidebarCollapsed]);

  // Fetch a specific document by ID
  const fetchDocument = async (documentId: string) => {
    try {
      const url = `${effectiveApiUrl}/documents/${documentId}`;
      console.log('DocumentsSection: Fetching document detail from:', url);

      // Use non-blocking fetch to avoid locking the UI
      fetch(url, {
        method: 'GET',
        headers: {
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch document: ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        console.log(`Fetched document details for ID: ${documentId}`);

        // Ensure document has a valid status in system_metadata
        if (!data.system_metadata) {
          data.system_metadata = {};
        }

        // If status is missing and we have a newly uploaded document, it should be "processing"
        if (!data.system_metadata.status && data.system_metadata.folder_name) {
          data.system_metadata.status = "processing";
        }

        setSelectedDocument(data);
      })
      .catch(err => {
        const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
        console.error(`Error fetching document details: ${errorMsg}`);
        showAlert(`Error fetching document: ${errorMsg}`, {
          type: 'error',
          duration: 5000
        });
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      console.error(`Error in fetchDocument: ${errorMsg}`);
      showAlert(`Error: ${errorMsg}`, {
        type: 'error',
        duration: 5000
      });
    }
  };

  // Handle document click
  const handleDocumentClick = (document: Document) => {
    fetchDocument(document.external_id);
  };

  // Helper function for document deletion API call
  const deleteDocumentApi = async (documentId: string) => {
    const response = await fetch(`${effectiveApiUrl}/documents/${documentId}`, {
      method: 'DELETE',
      headers: authToken ? { 'Authorization': `Bearer ${authToken}` } : {}
    });

    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`);
    }

    return response;
  };

  // Handle single document deletion
  const handleDeleteDocument = async (documentId: string) => {
    try {
      setLoading(true);

      console.log('DocumentsSection: Deleting document:', documentId);

      await deleteDocumentApi(documentId);

      // Clear selected document if it was the one deleted
      if (selectedDocument?.external_id === documentId) {
        setSelectedDocument(null);
      }

      // Refresh folders first, then documents
      await fetchFolders();
      await fetchDocuments();

      // Show success message
      showAlert("Document deleted successfully", {
        type: "success",
        duration: 3000
      });

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Delete Failed',
        duration: 5000
      });

      // Also remove the progress alert if there was an error
      removeAlert('delete-multiple-progress');
    } finally {
      setLoading(false);
    }
  };

  // Handle multiple document deletion
  const handleDeleteMultipleDocuments = async () => {
    if (selectedDocuments.length === 0) return;

    try {
      setLoading(true);

      // Show initial alert for deletion progress
      const alertId = 'delete-multiple-progress';
      showAlert(`Deleting ${selectedDocuments.length} documents...`, {
        type: 'info',
        dismissible: false,
        id: alertId
      });

      console.log('DocumentsSection: Deleting multiple documents:', selectedDocuments);

      // Perform deletions in parallel
      const results = await Promise.all(
        selectedDocuments.map(docId => deleteDocumentApi(docId))
      );

      // Check if any deletion failed
      const failedCount = results.filter(res => !res.ok).length;

      // Clear selected document if it was among deleted ones
      if (selectedDocument && selectedDocuments.includes(selectedDocument.external_id)) {
        setSelectedDocument(null);
      }

      // Clear selection
      setSelectedDocuments([]);

      // Refresh folders first, then documents
      await fetchFolders();
      await fetchDocuments();

      // Remove progress alert
      removeAlert(alertId);

      // Show final result alert
      if (failedCount > 0) {
        showAlert(`Deleted ${selectedDocuments.length - failedCount} documents. ${failedCount} deletions failed.`, {
          type: "warning",
          duration: 4000
        });
      } else {
        showAlert(`Successfully deleted ${selectedDocuments.length} documents`, {
          type: "success",
          duration: 3000
        });
      }

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Delete Failed',
        duration: 5000
      });

      // Also remove the progress alert if there was an error
      removeAlert('delete-multiple-progress');
    } finally {
      setLoading(false);
    }
  };

  // Handle checkbox change (wrapper function for use with shadcn checkbox)
  const handleCheckboxChange = (checked: boolean | "indeterminate", docId: string) => {
    setSelectedDocuments(prev => {
      if (checked === true && !prev.includes(docId)) {
        return [...prev, docId];
      } else if (checked === false && prev.includes(docId)) {
        return prev.filter(id => id !== docId);
      }
      return prev;
    });
  };

  // Helper function to get "indeterminate" state for select all checkbox
  const getSelectAllState = () => {
    if (selectedDocuments.length === 0) return false;
    if (selectedDocuments.length === documents.length) return true;
    return "indeterminate";
  };

  // Handle file upload
  const handleFileUpload = async (file: File | null) => {
    if (!file) {
      showAlert('Please select a file to upload', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    // Close dialog and update upload count using alert system
    setShowUploadDialog(false);
    const uploadId = 'upload-progress';
    showAlert(`Uploading 1 file...`, {
      type: 'upload',
      dismissible: false,
      id: uploadId
    });

    // Save file reference before we reset the form
    const fileToUploadRef = file;
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;

    // Reset form
    resetUploadDialog();

    try {
      const formData = new FormData();
      formData.append('file', fileToUploadRef);
      formData.append('metadata', metadataRef);
      formData.append('rules', rulesRef);

      // If we're in a specific folder (not "all" documents), add the folder_name to form data
      if (selectedFolder && selectedFolder !== "all") {
        try {
          // Parse metadata to validate it's proper JSON, but don't modify it
          JSON.parse(metadataRef || '{}');

          // The API expects folder_name as a direct Form parameter
          // This will be used by document_service._ensure_folder_exists()
          formData.set('metadata', metadataRef);
          formData.append('folder_name', selectedFolder);

          // Log for debugging
          console.log(`Adding file to folder: ${selectedFolder} as form field`);
        } catch (e) {
          console.error('Error parsing metadata:', e);
          formData.set('metadata', metadataRef);
          formData.append('folder_name', selectedFolder);
        }
      }

      const url = `${effectiveApiUrl}/ingest/file?use_colpali=${useColpaliRef}`;

      // Non-blocking fetch
      fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : ''
        },
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to upload: ${response.statusText}`);
        }
        return response.json();
      })
      .then((newDocument) => {
        // Log processing status of uploaded document
        if (newDocument && newDocument.system_metadata && newDocument.system_metadata.status === "processing") {
          console.log(`Document ${newDocument.external_id} is in processing status`);
          // No longer need to track processing documents for polling
        }

        // Force a fresh refresh after upload
        const refreshAfterUpload = async () => {
          try {
            console.log("Performing fresh refresh after upload (file)");
            // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
            await fetchFolders();
          } catch (err) {
            console.error('Error refreshing after file upload:', err);
          }
        };

        // Execute the refresh
        refreshAfterUpload();

        // Show success message and remove upload progress
        showAlert(`File uploaded successfully!`, {
          type: 'success',
          duration: 3000
        });

        // Remove the upload alert
        removeAlert('upload-progress');
      })
      .catch(err => {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        const errorMsg = `Error uploading ${fileToUploadRef.name}: ${errorMessage}`;

        // Show error alert and remove upload progress
        showAlert(errorMsg, {
          type: 'error',
          title: 'Upload Failed',
          duration: 5000
        });

        // Remove the upload alert
        removeAlert('upload-progress');
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      const errorMsg = `Error uploading ${fileToUploadRef.name}: ${errorMessage}`;

      // Show error alert
      showAlert(errorMsg, {
        type: 'error',
        title: 'Upload Failed',
        duration: 5000
      });

      // Remove the upload progress alert
      removeAlert('upload-progress');
    }
  };

  // Handle batch file upload
  const handleBatchFileUpload = async (files: File[], fromDragAndDrop: boolean = false) => {
    if (files.length === 0) {
      showAlert('Please select files to upload', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    // Close dialog if it's open (but not if drag and drop)
    if (!fromDragAndDrop) {
      setShowUploadDialog(false);
    }

    const fileCount = files.length;
    const uploadId = 'batch-upload-progress';
    showAlert(`Uploading ${fileCount} files...`, {
      type: 'upload',
      dismissible: false,
      id: uploadId
    });

    // Save form data locally
    const batchFilesRef = [...files];
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;

    // Only reset form if not from drag and drop
    if (!fromDragAndDrop) {
      resetUploadDialog();
    }

    try {
      const formData = new FormData();

      // Append each file to the formData with the same field name
      batchFilesRef.forEach(file => {
        formData.append('files', file);
      });

      // Add metadata to all cases
      formData.append('metadata', metadataRef);

      // If we're in a specific folder (not "all" documents), add the folder_name as a separate field
      if (selectedFolder && selectedFolder !== "all") {
        // The API expects folder_name directly, not ID
        formData.append('folder_name', selectedFolder);

        // Log for debugging
        console.log(`Adding batch files to folder: ${selectedFolder} as form field`);
      }

      formData.append('rules', rulesRef);
      formData.append('parallel', 'true');

      // Always set explicit use_colpali value
      const url = `${effectiveApiUrl}/ingest/files?use_colpali=${useColpaliRef}`;

      // Non-blocking fetch
      fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : ''
        },
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to upload: ${response.statusText}`);
        }
        return response.json();
      })
      .then(result => {
        // Log processing status of uploaded documents
        if (result && result.document_ids && result.document_ids.length > 0) {
          console.log(`${result.document_ids.length} documents are in processing status`);
          // No need for polling, just wait for manual refresh
        }

        // Force a fresh refresh after upload
        const refreshAfterUpload = async () => {
          try {
            console.log("Performing fresh refresh after upload (batch)");
            // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
            await fetchFolders();
          } catch (err) {
            console.error('Error refreshing after batch upload:', err);
          }
        };

        // Execute the refresh
        refreshAfterUpload();

        // If there are errors, show them in the error alert
        if (result.errors && result.errors.length > 0) {
          const errorMsg = `${result.errors.length} of ${fileCount} files failed to upload`;

          showAlert(errorMsg, {
            type: 'error',
            title: 'Upload Partially Failed',
            duration: 5000
          });
        } else {
          // Show success message
          showAlert(`${fileCount} files uploaded successfully!`, {
            type: 'success',
            duration: 3000
          });
        }

        // Remove the upload alert
        removeAlert('batch-upload-progress');
      })
      .catch(err => {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        const errorMsg = `Error uploading files: ${errorMessage}`;

        // Show error alert
        showAlert(errorMsg, {
          type: 'error',
          title: 'Upload Failed',
          duration: 5000
        });

        // Remove the upload alert
        removeAlert('batch-upload-progress');
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      const errorMsg = `Error uploading files: ${errorMessage}`;

      // Show error alert
      showAlert(errorMsg, {
        type: 'error',
        title: 'Upload Failed',
        duration: 5000
      });

      // Remove the upload progress alert
      removeAlert('batch-upload-progress');
    }
  };

  // Handle text upload
  const handleTextUpload = async (text: string, meta: string, rulesText: string, useColpaliFlag: boolean) => {
    if (!text.trim()) {
      showAlert('Please enter text content', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    // Close dialog and update upload count using alert system
    setShowUploadDialog(false);
    const uploadId = 'text-upload-progress';
    showAlert(`Uploading text document...`, {
      type: 'upload',
      dismissible: false,
      id: uploadId
    });

    // Save content before resetting
    const textContentRef = text;
    let metadataObj = {};
    let folderToUse = null;

    try {
      metadataObj = JSON.parse(meta || '{}');

      // If we're in a specific folder (not "all" documents), set folder variable
      if (selectedFolder && selectedFolder !== "all") {
        // The API expects the folder name directly
        folderToUse = selectedFolder;
        // Log for debugging
        console.log(`Will add text document to folder: ${selectedFolder}`);
      }
    } catch (e) {
      console.error('Error parsing metadata JSON:', e);
    }

    const rulesRef = rulesText;
    const useColpaliRef = useColpaliFlag;

    // Reset form immediately
    resetUploadDialog();

    try {
      // Non-blocking fetch with explicit use_colpali parameter
      const url = `${effectiveApiUrl}/ingest/text?use_colpali=${useColpaliRef}`;

      fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : '',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: textContentRef,
          metadata: metadataObj,
          rules: JSON.parse(rulesRef || '[]'),
          folder_name: folderToUse
        })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to upload: ${response.statusText}`);
        }
        return response.json();
      })
      .then((newDocument) => {
        // Log processing status of uploaded document
        if (newDocument && newDocument.system_metadata && newDocument.system_metadata.status === "processing") {
          console.log(`Document ${newDocument.external_id} is in processing status`);
          // No longer need to track processing documents for polling
        }

        // Force a fresh refresh after upload
        const refreshAfterUpload = async () => {
          try {
            console.log("Performing fresh refresh after upload (text)");
            // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
            await fetchFolders();
          } catch (err) {
            console.error('Error refreshing after text upload:', err);
          }
        };

        // Execute the refresh
        refreshAfterUpload();

        // Show success message
        showAlert(`Text document uploaded successfully!`, {
          type: 'success',
          duration: 3000
        });

        // Remove the upload alert
        removeAlert('text-upload-progress');
      })
      .catch(err => {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        const errorMsg = `Error uploading text: ${errorMessage}`;

        // Show error alert
        showAlert(errorMsg, {
          type: 'error',
          title: 'Upload Failed',
          duration: 5000
        });

        // Remove the upload alert
        removeAlert('text-upload-progress');
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      const errorMsg = `Error uploading text: ${errorMessage}`;

      // Show error alert
      showAlert(errorMsg, {
        type: 'error',
        title: 'Upload Failed',
        duration: 5000
      });

      // Remove the upload progress alert
      removeAlert('text-upload-progress');
    }
  };

  // Function to trigger refresh
  const handleRefresh = () => {
    console.log("Manual refresh triggered");
    showAlert("Refreshing documents and folders...", {
      type: 'info',
      duration: 1500
    });

    setLoading(true);

    // Create a new function to perform a truly fresh fetch
    const performFreshFetch = async () => {
      try {
        // ONLY fetch folders. The useEffect watching folders will trigger fetchDocuments.
        await fetchFolders();

        // Show success message (consider moving this if fetchFolders doesn't guarantee documents are loaded)
        showAlert("Refresh initiated. Data will update shortly.", {
          type: 'success',
          duration: 1500
        });
      } catch (error) {
        console.error("Error during refresh fetchFolders:", error);
        showAlert(`Error refreshing: ${error instanceof Error ? error.message : 'Unknown error'}`, {
          type: 'error',
          duration: 3000
        });
      } finally {
        // setLoading(false); // Loading will be handled by fetchDocuments triggered by useEffect
      }
    };

    // Execute the fresh fetch
    performFreshFetch();
  };

  return (
    <div
      className={cn(
        "flex-1 flex flex-col h-full relative p-4",
        selectedFolder && isDragging ? "drag-active" : ""
      )}
      {...(selectedFolder ? dragHandlers : {})}
    >
      {/* Drag overlay - only visible when dragging files over the folder */}
      {isDragging && selectedFolder && (
        <div className="absolute inset-0 bg-primary/10 backdrop-blur-sm z-50 flex items-center justify-center rounded-lg border-2 border-dashed border-primary animate-pulse">
          <div className="bg-background p-8 rounded-lg shadow-lg text-center">
            <Upload className="h-12 w-12 mx-auto mb-4 text-primary" />
            <h3 className="text-xl font-medium mb-2">Drop to Upload</h3>
            <p className="text-muted-foreground">
              Files will be added to {selectedFolder === "all" ? "your documents" : `folder "${selectedFolder}"`}
            </p>
          </div>
        </div>
      )}
      {/* Folder view controls - only show when not in a specific folder */}
      {/* No longer needed - controls will be provided in FolderList */}

      {/* Render the FolderList with header at all times when selectedFolder is not null */}
      {selectedFolder !== null && (
        <FolderList
          folders={folders}
          selectedFolder={selectedFolder}
          setSelectedFolder={setSelectedFolder}
          apiBaseUrl={effectiveApiUrl}
          authToken={authToken}
          refreshFolders={fetchFolders}
          loading={foldersLoading}
          refreshAction={handleRefresh}
          selectedDocuments={selectedDocuments}
          handleDeleteMultipleDocuments={handleDeleteMultipleDocuments}
          uploadDialogComponent={
            <UploadDialog
              showUploadDialog={showUploadDialog}
              setShowUploadDialog={setShowUploadDialog}
              loading={loading}
              onFileUpload={handleFileUpload}
              onBatchFileUpload={handleBatchFileUpload}
              onTextUpload={handleTextUpload}
            />
          }
        />
      )}

      {selectedFolder && documents.length === 0 && !loading ? (
        <div className="text-center py-8 border border-dashed rounded-lg flex-1 flex items-center justify-center">
          <div>
            <Upload className="mx-auto h-12 w-12 mb-2 text-muted-foreground" />
            <p className="text-muted-foreground">Drag and drop files here to upload to this folder.</p>
            <p className="text-xs text-muted-foreground mt-2">Or use the upload button in the top right.</p>
          </div>
        </div>
      ) : selectedFolder && loading ? (
        <div className="text-center py-8 flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p className="text-muted-foreground">Loading documents...</p>
          </div>
        </div>
      ) : selectedFolder === null ? (
        <div className="flex flex-col gap-4 flex-1">
          <FolderList
            folders={folders}
            selectedFolder={selectedFolder}
            setSelectedFolder={setSelectedFolder}
            apiBaseUrl={effectiveApiUrl}
            authToken={authToken}
            refreshFolders={fetchFolders}
            loading={foldersLoading}
            refreshAction={handleRefresh}
            selectedDocuments={selectedDocuments}
            handleDeleteMultipleDocuments={handleDeleteMultipleDocuments}
            uploadDialogComponent={
              <UploadDialog
                showUploadDialog={showUploadDialog}
                setShowUploadDialog={setShowUploadDialog}
                loading={loading}
                onFileUpload={handleFileUpload}
                onBatchFileUpload={handleBatchFileUpload}
                onTextUpload={handleTextUpload}
              />
            }
          />
        </div>
      ) : (
        <div className="flex flex-col md:flex-row gap-4 flex-1">
          <div className={cn(
            "w-full transition-all duration-300",
            selectedDocument ? "md:w-2/3" : "md:w-full"
          )}>
            <DocumentList
              documents={documents}
              selectedDocument={selectedDocument}
              selectedDocuments={selectedDocuments}
              handleDocumentClick={handleDocumentClick}
              handleCheckboxChange={handleCheckboxChange}
              getSelectAllState={getSelectAllState}
              setSelectedDocuments={setSelectedDocuments}
              setDocuments={setDocuments}
              loading={loading}
              apiBaseUrl={effectiveApiUrl}
              authToken={authToken}
              selectedFolder={selectedFolder}
            />
          </div>

          {selectedDocument && (
            <div className="w-full md:w-1/3 animate-in slide-in-from-right duration-300">
              <DocumentDetail
                selectedDocument={selectedDocument}
                handleDeleteDocument={handleDeleteDocument}
                folders={folders}
                apiBaseUrl={effectiveApiUrl}
                authToken={authToken}
                refreshDocuments={fetchDocuments}
                refreshFolders={fetchFolders}
                loading={loading}
                onClose={() => setSelectedDocument(null)}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentsSection;
