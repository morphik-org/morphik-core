"use client";

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
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
    console.log(`fetchDocuments called from: ${source}`)
    try {
      // Only set loading state for initial load, not for refreshes
      if (documents.length === 0) {
        setLoading(true);
      }
      
      // Don't fetch if no folder is selected (showing folder grid view)
      if (selectedFolder === null) {
        setDocuments([]);
        setLoading(false);
        return;
      }
      
      // Prepare for document fetching
      let apiUrl = `${effectiveApiUrl}/documents`;
      // CRITICAL FIX: The /documents endpoint uses POST method
      let method = 'POST';
      let requestBody = {};
      
      // If we're looking at a specific folder (not "all" documents)
      if (selectedFolder && selectedFolder !== "all") {
        console.log(`Fetching documents for folder: ${selectedFolder}`);
        
        // Find the target folder to get its document IDs
        const targetFolder = folders.find(folder => folder.name === selectedFolder);
        
        if (targetFolder) {
          // Ensure document_ids is always an array
          const documentIds = Array.isArray(targetFolder.document_ids) ? targetFolder.document_ids : [];
          
          if (documentIds.length > 0) {
            // If we found the folder and it contains documents,
            // Get document details for each document ID in the folder
            console.log(`Found folder ${targetFolder.name} with ${documentIds.length} documents`);
            
            // Use batch/documents endpoint which accepts document_ids for efficient fetching
            apiUrl = `${effectiveApiUrl}/batch/documents`;
            method = 'POST';
            requestBody = {
              document_ids: [...documentIds]
            };
          } else {
            console.log(`Folder ${targetFolder.name} has no documents`);
            // For empty folder, we'll send an empty request body to the documents endpoint
            requestBody = {};
          }
        } else {
          console.log(`Folder ${selectedFolder} has no documents or couldn't be found`);
          // For unknown folder, we'll send an empty request body to the documents endpoint
          requestBody = {};
        }
      } else {
        // For "all" documents request
        requestBody = {};
      }
      
      console.log(`DocumentsSection: Sending ${method} request to: ${apiUrl}`);
      
      // Use non-blocking fetch with appropriate method
      fetch(apiUrl, {
        method: method,
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        },
        body: JSON.stringify(requestBody)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch documents: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data: Document[]) => {
        // Ensure all documents have a valid status in system_metadata
        const processedData = data.map((doc: Document) => {
          // If system_metadata doesn't exist, create it
          if (!doc.system_metadata) {
            doc.system_metadata = {};
          }
          
          // If status is missing and we have a newly uploaded document, it should be "processing"
          if (!doc.system_metadata.status && doc.system_metadata.folder_name) {
            doc.system_metadata.status = "processing";
          }
          
          return doc;
        });
        
        console.log(`Fetched ${processedData.length} documents from ${apiUrl}`);
        
        // Only update state if component is still mounted
        setDocuments(processedData);
        setLoading(false);
        
        // Log for debugging
        const processingCount = processedData.filter(doc => doc.system_metadata?.status === "processing").length;
        if (processingCount > 0) {
          console.log(`Found ${processingCount} documents still processing`);
        }
      })
      .catch(err => {
        const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
        console.error(`Document fetch error: ${errorMsg}`);
        showAlert(errorMsg, {
          type: 'error',
          title: 'Error',
          duration: 5000
        });
        setLoading(false);
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Error',
        duration: 5000
      });
      setLoading(false);
    }
  }, [effectiveApiUrl, authToken, documents.length, selectedFolder, folders]);

  // Fetch all folders
  const fetchFolders = useCallback(async (source: string = 'unknown') => {
    console.log(`fetchFolders called from: ${source}`)
    try {
      setFoldersLoading(true);
      
      // Use non-blocking fetch with GET method
      const url = `${effectiveApiUrl}/folders`;
      console.log(`Fetching folders from: ${url}`);
      
      fetch(url, {
        method: 'GET',
        headers: {
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch folders: ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        console.log(`Fetched ${data.length} folders`);
        setFolders(data);
        setFoldersLoading(false);
      })
      .catch(err => {
        const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
        console.error(`Error fetching folders: ${errorMsg}`);
        showAlert(`Error fetching folders: ${errorMsg}`, {
          type: 'error',
          duration: 3000
        });
        setFoldersLoading(false);
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      console.error(`Error in fetchFolders: ${errorMsg}`);
      setFoldersLoading(false);
    }
  }, [effectiveApiUrl, authToken]);

  // No automatic polling - we'll only refresh on upload and manual refresh button clicks
  
  // Create memoized fetch functions that don't cause infinite loops
  // We need to make sure they don't trigger re-renders that cause further fetches
  const stableFetchFolders = useCallback((source: string = 'stable-call') => {
    console.log(`stableFetchFolders called from: ${source}`);
    return fetchFolders(source);
  // Keep dependencies minimal to prevent recreation on every render
  }, [effectiveApiUrl, authToken]);
  
  const stableFetchDocuments = useCallback((source: string = 'stable-call') => {
    console.log(`stableFetchDocuments called from: ${source}`);
    return fetchDocuments(source);
  // Keep dependencies minimal to prevent recreation on every render  
  }, [effectiveApiUrl, authToken, selectedFolder]);
  
  // Fetch data when auth token or API URL changes
  useEffect(() => {
    // Only run this effect if we have auth or are on localhost 
    if (authToken || effectiveApiUrl.includes('localhost')) {
      console.log('DocumentsSection: Fetching initial data');
      
      // Clear current data and reset state
      setDocuments([]);
      setFolders([]);
      setSelectedDocument(null);
      setSelectedDocuments([]);
      
      // Use a flag to track component mounting state
      let isMounted = true;
      
      // Create an abort controller for request cancellation
      const controller = new AbortController();
      
      // Add a slight delay to prevent multiple rapid calls
      const timeoutId = setTimeout(() => {
        if (isMounted) {
          // Fetch folders first
          stableFetchFolders('initial-load')
            .then(() => {
              // Only fetch documents if we're still mounted
              if (isMounted && selectedFolder !== null) {
                return stableFetchDocuments('initial-load');
              }
            })
            .catch(err => {
              console.error("Error during initial data fetch:", err);
            });
        }
      }, 100);
      
      // Cleanup when component unmounts or the effect runs again
      return () => {
        clearTimeout(timeoutId);
        isMounted = false;
        controller.abort();
      };
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authToken, effectiveApiUrl, stableFetchFolders, stableFetchDocuments, selectedFolder]);
  
  // Helper function to refresh documents based on current view
  const refreshDocuments = async (folders: Folder[]) => {
    try {
      if (selectedFolder && selectedFolder !== "all") {
        // Find the folder by name
        const targetFolder = folders.find(folder => folder.name === selectedFolder);
        
        if (targetFolder) {
          console.log(`Refresh: Found folder ${targetFolder.name} in fresh data`);
          
          // Get the document IDs from the folder
          const documentIds = Array.isArray(targetFolder.document_ids) ? targetFolder.document_ids : [];
          console.log(`Refresh: Folder has ${documentIds.length} documents`);
          
          if (documentIds.length > 0) {
            // Fetch document details for the IDs
            const docResponse = await fetch(`${effectiveApiUrl}/batch/documents`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
              },
              body: JSON.stringify({
                document_ids: [...documentIds]
              })
            });
            
            if (!docResponse.ok) {
              throw new Error(`Failed to fetch documents: ${docResponse.statusText}`);
            }
            
            const freshDocs = await docResponse.json();
            console.log(`Refresh: Fetched ${freshDocs.length} document details`);
            
            // Update documents state
            setDocuments(freshDocs);
          } else {
            // Empty folder
            setDocuments([]);
          }
        } else {
          console.log(`Refresh: Selected folder ${selectedFolder} not found in fresh data`);
          setDocuments([]);
        }
      } else if (selectedFolder === "all") {
        // For "all" documents view, fetch all documents
        const allDocsResponse = await fetch(`${effectiveApiUrl}/documents`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
          },
          body: JSON.stringify({})
        });
        
        if (!allDocsResponse.ok) {
          throw new Error(`Failed to fetch all documents: ${allDocsResponse.statusText}`);
        }
        
        const allDocs = await allDocsResponse.json();
        console.log(`Refresh: Fetched ${allDocs.length} documents for "all" view`);
        setDocuments(allDocs);
      }
    } catch (error) {
      console.error("Error refreshing documents:", error);
    }
  };

  // Collapse sidebar when a folder is selected
  useEffect(() => {
    if (selectedFolder !== null && setSidebarCollapsed) {
      setSidebarCollapsed(true);
    } else if (setSidebarCollapsed) {
      setSidebarCollapsed(false);
    }
  }, [selectedFolder, setSidebarCollapsed]);

  // Fetch documents when selected folder changes
  useEffect(() => {
    // Skip initial render to prevent double fetching with the auth useEffect
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    
    console.log(`Folder selection changed to: ${selectedFolder}`);
    
    // Clear selected document when changing folders
    setSelectedDocument(null);
    setSelectedDocuments([]);
    
    // CRITICAL: Clear document list immediately to prevent showing old documents
    // This prevents showing documents from previous folders while loading
    setDocuments([]);
    
    // Create a flag to handle component unmounting
    let isMounted = true;
    
    // Create an abort controller for fetch operations
    const controller = new AbortController();
    
    // Only fetch if we have a valid auth token or running locally
    if ((authToken || effectiveApiUrl.includes('localhost')) && isMounted) {
      // Add a small delay to prevent rapid consecutive calls
      const timeoutId = setTimeout(async () => {
        try {
          // Set loading state to show we're fetching new data
          setLoading(true);
          
          // Start with a fresh folder fetch
          const folderResponse = await fetch(`${effectiveApiUrl}/folders`, {
            method: 'GET',
            headers: {
              ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
            },
            signal: controller.signal
          });
          
          if (!folderResponse.ok) {
            throw new Error(`Failed to fetch folders: ${folderResponse.statusText}`);
          }
          
          // Get fresh folder data
          const freshFolders = await folderResponse.json();
          console.log(`Folder change: Fetched ${freshFolders.length} folders with fresh data`);
          
          // Update folder state
          if (isMounted) {
            setFolders(freshFolders);
            
            // Then fetch documents with fresh folder data
            await refreshDocuments(freshFolders);
          }
        } catch (err) {
          if (err instanceof Error && err.name !== 'AbortError') {
            console.error("Error during folder change fetch:", err);
          }
        } finally {
          if (isMounted) {
            setLoading(false);
          }
        }
      }, 100);
      
      // Clean up timeout if unmounted
      return () => {
        clearTimeout(timeoutId);
        isMounted = false;
        controller.abort();
      };
    }
    
    // Cleanup function to prevent updates after unmount
    return () => {
      isMounted = false;
      controller.abort();
    };
  // Only depend on these specific props to prevent infinite loops
  }, [selectedFolder, authToken, effectiveApiUrl]);

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
      
      const url = `${effectiveApiUrl}/ingest/file${useColpaliRef ? '?use_colpali=true' : ''}`;
      
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
        // This is a special function to ensure we get truly fresh data
        const refreshAfterUpload = async () => {
          try {
            console.log("Performing fresh refresh after upload");
            // Clear folder data to force a clean refresh
            setFolders([]);
            
            // Get fresh folder data from the server
            const folderResponse = await fetch(`${effectiveApiUrl}/folders`, {
              method: 'GET',
              headers: {
                ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
              }
            });
            
            if (!folderResponse.ok) {
              throw new Error(`Failed to fetch folders: ${folderResponse.statusText}`);
            }
            
            const freshFolders = await folderResponse.json();
            console.log(`Upload: Fetched ${freshFolders.length} folders with fresh data`);
            
            // Update folders state with fresh data
            setFolders(freshFolders);
            
            // Now fetch documents based on the current view
            await refreshDocuments(freshFolders);
          } catch (err) {
            console.error('Error refreshing after upload:', err);
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
      if (useColpaliRef !== undefined) {
        formData.append('use_colpali', useColpaliRef.toString());
      }
      
      // Non-blocking fetch
      fetch(`${effectiveApiUrl}/ingest/files`, {
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
        // This is a special function to ensure we get truly fresh data
        const refreshAfterUpload = async () => {
          try {
            console.log("Performing fresh refresh after upload");
            // Clear folder data to force a clean refresh
            setFolders([]);
            
            // Get fresh folder data from the server
            const folderResponse = await fetch(`${effectiveApiUrl}/folders`, {
              method: 'GET',
              headers: {
                ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
              }
            });
            
            if (!folderResponse.ok) {
              throw new Error(`Failed to fetch folders: ${folderResponse.statusText}`);
            }
            
            const freshFolders = await folderResponse.json();
            console.log(`Upload: Fetched ${freshFolders.length} folders with fresh data`);
            
            // Update folders state with fresh data
            setFolders(freshFolders);
            
            // Now fetch documents based on the current view
            await refreshDocuments(freshFolders);
          } catch (err) {
            console.error('Error refreshing after upload:', err);
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
      // Non-blocking fetch
      fetch(`${effectiveApiUrl}/ingest/text`, {
        method: 'POST',
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : '',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: textContentRef,
          metadata: metadataObj,
          rules: JSON.parse(rulesRef || '[]'),
          use_colpali: useColpaliRef,
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
        // This is a special function to ensure we get truly fresh data
        const refreshAfterUpload = async () => {
          try {
            console.log("Performing fresh refresh after upload");
            // Clear folder data to force a clean refresh
            setFolders([]);
            
            // Get fresh folder data from the server
            const folderResponse = await fetch(`${effectiveApiUrl}/folders`, {
              method: 'GET',
              headers: {
                ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
              }
            });
            
            if (!folderResponse.ok) {
              throw new Error(`Failed to fetch folders: ${folderResponse.statusText}`);
            }
            
            const freshFolders = await folderResponse.json();
            console.log(`Upload: Fetched ${freshFolders.length} folders with fresh data`);
            
            // Update folders state with fresh data
            setFolders(freshFolders);
            
            // Now fetch documents based on the current view
            await refreshDocuments(freshFolders);
          } catch (err) {
            console.error('Error refreshing after upload:', err);
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

  // Title based on selected folder
  const sectionTitle = selectedFolder === null 
    ? "Folders" 
    : selectedFolder === "all" 
      ? "All Documents" 
      : `Folder: ${selectedFolder}`;

  // Function to trigger refresh
  const handleRefresh = () => {
    console.log("Manual refresh triggered");
    // Show a loading indicator
    showAlert("Refreshing documents and folders...", {
      type: 'info',
      duration: 1500
    });
    
    // First clear folder data to force a clean refresh
    setLoading(true);
    setFolders([]);
    
    // Create a new function to perform a truly fresh fetch
    const performFreshFetch = async () => {
      try {
        // First get fresh folder data from the server
        const folderResponse = await fetch(`${effectiveApiUrl}/folders`, {
          method: 'GET',
          headers: {
            ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
          }
        });
        
        if (!folderResponse.ok) {
          throw new Error(`Failed to fetch folders: ${folderResponse.statusText}`);
        }
        
        // Get the fresh folder data
        const freshFolders = await folderResponse.json();
        console.log(`Refresh: Fetched ${freshFolders.length} folders with fresh data`);
        
        // Update folders state with fresh data
        setFolders(freshFolders);
        
        // Use our helper function to refresh documents with fresh folder data
        await refreshDocuments(freshFolders);
        
        // Show success message
        showAlert("Refresh completed successfully", {
          type: 'success',
          duration: 1500
        });
      } catch (error) {
        console.error("Error during refresh:", error);
        showAlert(`Error refreshing: ${error instanceof Error ? error.message : 'Unknown error'}`, {
          type: 'error',
          duration: 3000
        });
      } finally {
        setLoading(false);
      }
    };
    
    // Execute the fresh fetch
    performFreshFetch();
  };

  return (
    <div 
      className={cn(
        "flex-1 flex flex-col h-full relative",
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
      {/* Hide the main header when viewing a specific folder - it will be merged with the FolderList header */}
      {selectedFolder === null && (
        <div className="flex justify-between items-center py-3 mb-4">
          <div>
            <h2 className="text-2xl font-bold leading-tight">Folders</h2>
            <p className="text-muted-foreground">Manage your uploaded documents and view their metadata.</p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={handleRefresh}
              disabled={loading}
              className="flex items-center"
              title="Refresh folders"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                <path d="M21 3v5h-5"></path>
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                <path d="M8 16H3v5"></path>
              </svg>
              Refresh
            </Button>
            <UploadDialog
              showUploadDialog={showUploadDialog}
              setShowUploadDialog={setShowUploadDialog}
              loading={loading}
              onFileUpload={handleFileUpload}
              onBatchFileUpload={handleBatchFileUpload}
              onTextUpload={handleTextUpload}
            />
          </div>
        </div>
      )}

      {documents.length === 0 && !loading && folders.length === 0 && !foldersLoading ? (
        <div className="text-center py-8 border border-dashed rounded-lg flex-1 flex items-center justify-center">
          <div>
            <Upload className="mx-auto h-12 w-12 mb-2 text-muted-foreground" />
            <p className="text-muted-foreground">No documents found. Upload your first document.</p>
          </div>
        </div>
      ) : selectedFolder && documents.length === 0 && !loading ? (
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
      ) : (
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
          
          {selectedFolder !== null && (
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
      )}
    </div>
  );
};

export default DocumentsSection;