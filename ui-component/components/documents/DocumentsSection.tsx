"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';
import { showAlert, removeAlert } from '@/components/ui/alert-system';
import DocumentList from './DocumentList';
import DocumentDetail from './DocumentDetail';
import { UploadDialog, useUploadDialog } from './UploadDialog';

import { Document } from '@/components/types';

interface DocumentsSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
}

const DocumentsSection: React.FC<DocumentsSectionProps> = ({ apiBaseUrl, authToken }) => {
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
  // State for documents
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

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

  // Headers for API requests - ensure this updates when props change
  const headers = React.useMemo(() => {
    return {
      'Authorization': authToken ? `Bearer ${authToken}` : ''
    };
  }, [authToken]);

  // Fetch all documents
  const fetchDocuments = useCallback(async () => {
    try {
      // Only set loading state for initial load, not for refreshes
      if (documents.length === 0) {
        setLoading(true);
      }
      
      console.log('DocumentsSection: Sending request to:', `${effectiveApiUrl}/documents`);
      console.log('DocumentsSection: Headers:', JSON.stringify(headers));
      
      // Use non-blocking fetch
      fetch(`${effectiveApiUrl}/documents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        },
        body: JSON.stringify({})
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch documents: ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        setDocuments(data);
        setLoading(false);
      })
      .catch(err => {
        const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
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
  }, [effectiveApiUrl, authToken, headers, documents.length]);

  // Fetch documents when auth token or API URL changes (but not when fetchDocuments changes)
  useEffect(() => {
    if (authToken || effectiveApiUrl.includes('localhost')) {
      console.log('DocumentsSection: Fetching documents on auth/API change');
      
      // Clear current documents and reset state
      setDocuments([]);
      setSelectedDocument(null);
      fetchDocuments();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authToken, effectiveApiUrl]);

  // Fetch a specific document by ID
  const fetchDocument = async (documentId: string) => {
    try {
      console.log('DocumentsSection: Fetching document detail from:', `${effectiveApiUrl}/documents/${documentId}`);
      
      // Use non-blocking fetch to avoid locking the UI
      fetch(`${effectiveApiUrl}/documents/${documentId}`, {
        headers: authToken ? { 'Authorization': `Bearer ${authToken}` } : {}
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch document: ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        setSelectedDocument(data);
      })
      .catch(err => {
        const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
        showAlert(errorMsg, {
          type: 'error',
          title: 'Error',
          duration: 5000
        });
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Error',
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
      
      // Refresh documents list
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
      
      // Refresh documents list
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
      .then(() => {
        fetchDocuments(); // Refresh document list (non-blocking)
        
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
  const handleBatchFileUpload = async (files: File[]) => {
    if (files.length === 0) {
      showAlert('Please select files to upload', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    // Close dialog and update upload count using alert system
    setShowUploadDialog(false);
    const fileCount = files.length;
    const uploadId = 'batch-upload-progress';
    showAlert(`Uploading ${fileCount} files...`, {
      type: 'upload',
      dismissible: false,
      id: uploadId
    });
    
    // Save form data locally before resetting
    const batchFilesRef = [...files];
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;
    
    // Reset form immediately
    resetUploadDialog();
    
    try {      
      const formData = new FormData();
      
      // Append each file to the formData with the same field name
      batchFilesRef.forEach(file => {
        formData.append('files', file);
      });
      
      formData.append('metadata', metadataRef);
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
        fetchDocuments(); // Refresh document list (non-blocking)
        
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
    const metadataRef = meta;
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
          metadata: JSON.parse(metadataRef || '{}'),
          rules: JSON.parse(rulesRef || '[]'),
          use_colpali: useColpaliRef
        })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to upload: ${response.statusText}`);
        }
        return response.json();
      })
      .then(() => {
        fetchDocuments(); // Refresh document list (non-blocking)
        
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

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="flex justify-between items-center py-3 mb-4">
        <div className="flex items-center gap-4">
          <div>
            <h2 className="text-2xl font-bold leading-tight">Your Documents</h2>
            <p className="text-muted-foreground">Manage your uploaded documents and view their metadata.</p>
          </div>
          {selectedDocuments.length > 0 && (
            <Button 
              variant="outline" 
              onClick={handleDeleteMultipleDocuments} 
              disabled={loading}
              className="border-red-500 text-red-500 hover:bg-red-50 ml-4"
            >
              Delete {selectedDocuments.length} selected
            </Button>
          )}
        </div>
        <UploadDialog
          showUploadDialog={showUploadDialog}
          setShowUploadDialog={setShowUploadDialog}
          loading={loading}
          onFileUpload={handleFileUpload}
          onBatchFileUpload={handleBatchFileUpload}
          onTextUpload={handleTextUpload}
        />
      </div>

      {documents.length === 0 && !loading ? (
        <div className="text-center py-8 border border-dashed rounded-lg flex-1 flex items-center justify-center">
          <div>
            <Upload className="mx-auto h-12 w-12 mb-2 text-muted-foreground" />
            <p className="text-muted-foreground">No documents found. Upload your first document.</p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col md:flex-row gap-4 flex-1">
          <div className="w-full md:w-2/3">
            <DocumentList
              documents={documents}
              selectedDocument={selectedDocument}
              selectedDocuments={selectedDocuments}
              handleDocumentClick={handleDocumentClick}
              handleCheckboxChange={handleCheckboxChange}
              getSelectAllState={getSelectAllState}
              setSelectedDocuments={setSelectedDocuments}
              loading={loading}
            />
          </div>
          
          <div className="w-full md:w-1/3">
            <DocumentDetail
              selectedDocument={selectedDocument}
              handleDeleteDocument={handleDeleteDocument}
              loading={loading}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentsSection;