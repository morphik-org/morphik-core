"use client";

import React, { useState, useEffect, ChangeEvent } from 'react';
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Switch } from '@/components/ui/switch';
import { Upload, Search, MessageSquare, Info, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { Sidebar } from '@/components/ui/sidebar';
import GraphSection from '@/components/GraphSection';
import NotebookSection from '@/components/NotebookSection';
import { showAlert, removeAlert } from '@/components/ui/alert-system';
import Image from 'next/image';

// API base URL - change this to match your Morphik server
const API_BASE_URL = 'http://localhost:8000';

interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, unknown>;
  system_metadata: Record<string, unknown>;
  additional_metadata: Record<string, unknown>;
}

interface SearchResult {
  document_id: string;
  chunk_number: number;
  content: string;
  content_type: string;
  score: number;
  filename?: string;
  metadata: Record<string, unknown>;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface SearchOptions {
  filters: string;
  k: number;
  min_score: number;
  use_reranking: boolean;
  embed_as_image: boolean;
}

interface QueryOptions extends SearchOptions {
  max_tokens: number;
  temperature: number;
  graph_name?: string;
}

// Commented out as currently unused
// interface BatchUploadError {
//   filename: string;
//   error: string;
// }

const MorphikUI = () => {
  const [activeSection, setActiveSection] = useState('documents');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [chatQuery, setChatQuery] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  // No longer need error state as we're using alert system
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  // Alert system now handles upload status messages
  const [uploadType, setUploadType] = useState<'file' | 'text' | 'batch'>('file');
  const [textContent, setTextContent] = useState('');
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  const [batchFilesToUpload, setBatchFilesToUpload] = useState<File[]>([]);
  const [metadata, setMetadata] = useState('{}');
  const [rules, setRules] = useState('[]');
  const [useColpali, setUseColpali] = useState(true);
  
  // Advanced options for search
  const [showSearchAdvanced, setShowSearchAdvanced] = useState(false);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    embed_as_image: true
  });

  // Advanced options for chat/query
  const [showChatAdvanced, setShowChatAdvanced] = useState(false);
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    embed_as_image: true,
    max_tokens: 500,
    temperature: 0.7
  });

  // Auth token - in a real application, you would get this from your auth system
  const authToken = 'YOUR_AUTH_TOKEN';

  // Headers for API requests
  const headers = {
    'Authorization': authToken
  };

  // Fetch all documents - non-blocking implementation
  const fetchDocuments = async () => {
    try {
      // Only set loading state for initial load, not for refreshes
      if (documents.length === 0) {
        setLoading(true);
      }
      // Using alerts instead of error state
      
      // Use non-blocking fetch
      fetch(`${API_BASE_URL}/documents`, {
        method: 'POST',
        headers: {
          ...headers,
          'Content-Type': 'application/json'
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
  };

  // Fetch documents on component mount
  useEffect(() => {
    fetchDocuments();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch a specific document by ID - fully non-blocking
  const fetchDocument = async (documentId: string) => {
    try {
      // Using alerts instead of error state
      
      // Use non-blocking fetch to avoid locking the UI
      fetch(`${API_BASE_URL}/documents/${documentId}`, {
        headers
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
  
  // Handle document deletion
  const handleDeleteDocument = async (documentId: string) => {
    try {
      setLoading(true);
      // Using alerts instead of error state
      
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
        method: 'DELETE',
        headers
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete document: ${response.statusText}`);
      }
      
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
      
      // Perform deletions sequentially
      const results = await Promise.all(
        selectedDocuments.map(docId =>
          fetch(`${API_BASE_URL}/documents/${docId}`, {
            method: 'DELETE',
            headers
          })
        )
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
    } finally {
      setLoading(false);
    }
  };
  
  // Toggle document selection - currently handled by handleCheckboxChange
  // Keeping implementation in comments for reference
  /*
  const toggleDocumentSelection = (e: React.MouseEvent, docId: string) => {
    e.stopPropagation(); // Prevent document selection/details view
    
    setSelectedDocuments(prev => {
      if (prev.includes(docId)) {
        return prev.filter(id => id !== docId);
      } else {
        return [...prev, docId];
      }
    });
  };
  */
  
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
  const handleFileUpload = async () => {
    if (!fileToUpload) {
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
    
    // Reset form data immediately so users can initiate another upload if desired
    const fileToUploadRef = fileToUpload;
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;
    
    // Reset form
    setFileToUpload(null);
    setMetadata('{}');
    setRules('[]');
    setUseColpali(true);
    
    try {
      // Using alerts instead of error state
      
      const formData = new FormData();
      formData.append('file', fileToUploadRef);
      formData.append('metadata', metadataRef);
      formData.append('rules', rulesRef);
      
      const url = `${API_BASE_URL}/ingest/file${useColpaliRef ? '?embed_as_image=true' : ''}`;
      
      // Non-blocking fetch
      fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': authToken
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
  const handleBatchFileUpload = async () => {
    if (batchFilesToUpload.length === 0) {
      showAlert('Please select files to upload', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    // Close dialog and update upload count using alert system
    setShowUploadDialog(false);
    const fileCount = batchFilesToUpload.length;
    const uploadId = 'batch-upload-progress';
    showAlert(`Uploading ${fileCount} files...`, {
      type: 'upload',
      dismissible: false,
      id: uploadId
    });
    
    // Save form data locally before resetting
    const batchFilesRef = [...batchFilesToUpload];
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;
    
    // Reset form immediately
    setBatchFilesToUpload([]);
    setMetadata('{}');
    setRules('[]');
    setUseColpali(true);
    
    try {
      // Using alerts instead of error state
      
      const formData = new FormData();
      
      // Append each file to the formData with the same field name
      batchFilesRef.forEach(file => {
        formData.append('files', file);
      });
      
      formData.append('metadata', metadataRef);
      formData.append('rules', rulesRef);
      formData.append('parallel', 'true');
      if (useColpaliRef !== undefined) {
        formData.append('embed_as_image', useColpaliRef.toString());
      }
      
      // Non-blocking fetch
      fetch(`${API_BASE_URL}/ingest/files`, {
        method: 'POST',
        headers: {
          'Authorization': authToken
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
  const handleTextUpload = async () => {
    if (!textContent.trim()) {
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
    const textContentRef = textContent;
    const metadataRef = metadata;
    const rulesRef = rules;
    const useColpaliRef = useColpali;
    
    // Reset form immediately
    setTextContent('');
    setMetadata('{}');
    setRules('[]');
    setUseColpali(true);
    
    try {
      // Using alerts instead of error state
      
      // Non-blocking fetch
      fetch(`${API_BASE_URL}/ingest/text`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: textContentRef,
          metadata: JSON.parse(metadataRef || '{}'),
          rules: JSON.parse(rulesRef || '[]'),
          embed_as_image: useColpaliRef
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

  // Handle search
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      showAlert('Please enter a search query', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    try {
      setLoading(true);
      // Using alerts instead of error state
      
      const response = await fetch(`${API_BASE_URL}/retrieve/chunks`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchQuery,
          filters: JSON.parse(searchOptions.filters || '{}'),
          k: searchOptions.k,
          min_score: searchOptions.min_score,
          use_reranking: searchOptions.use_reranking,
          embed_as_image: searchOptions.embed_as_image
        })
      });
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSearchResults(data);
      
      if (data.length === 0) {
        showAlert("No search results found for the query", {
          type: "info",
          duration: 3000
        });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Search Failed',
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle chat
  const handleChat = async () => {
    if (!chatQuery.trim()) {
      showAlert('Please enter a message', {
        type: 'error',
        duration: 3000
      });
      return;
    }

    try {
      setLoading(true);
      // Using alerts instead of error state
      
      // Add user message to chat
      const userMessage: ChatMessage = { role: 'user', content: chatQuery };
      setChatMessages(prev => [...prev, userMessage]);
      
      // Prepare options with graph_name if it exists
      const options = {
        filters: JSON.parse(queryOptions.filters || '{}'),
        k: queryOptions.k,
        min_score: queryOptions.min_score,
        use_reranking: queryOptions.use_reranking,
        embed_as_image: queryOptions.embed_as_image,
        max_tokens: queryOptions.max_tokens,
        temperature: queryOptions.temperature,
        graph_name: queryOptions.graph_name
      };
      
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: chatQuery,
          ...options
        })
      });
      
      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Add assistant response to chat
      const assistantMessage: ChatMessage = { role: 'assistant', content: data.completion };
      setChatMessages(prev => [...prev, assistantMessage]);
      setChatQuery(''); // Clear input
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred';
      showAlert(errorMsg, {
        type: 'error',
        title: 'Chat Query Failed',
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  };

  // Render content based on content type
  const renderContent = (content: string, contentType: string) => {
    if (contentType.startsWith('image/')) {
      return (
        <div className="flex justify-center p-4 bg-gray-100 rounded-md">
          <Image 
            src={content} 
            alt="Document content" 
            className="max-w-full max-h-96 object-contain"
            width={500}
            height={300}
          />
        </div>
      );
    } else if (content.startsWith('data:image/png;base64,') || content.startsWith('data:image/jpeg;base64,')) {
      return (
        <div className="flex justify-center p-4 bg-gray-100 rounded-md">
          <Image 
            src={content} 
            alt="Base64 image content" 
            className="max-w-full max-h-96 object-contain"
            width={500}
            height={300}
          />
        </div>
      );
    } else {
      return (
        <div className="bg-gray-50 p-4 rounded-md whitespace-pre-wrap font-mono text-sm">
          {content}
        </div>
      );
    }
  };

  // Reset upload dialog
  const resetUploadDialog = () => {
    setUploadType('file');
    setFileToUpload(null);
    setBatchFilesToUpload([]);
    setTextContent('');
    setMetadata('{}');
    setRules('[]');
    setUseColpali(true);
  };

  // Update search options
  const updateSearchOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    setSearchOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Fetch available graphs for dropdown
  const [availableGraphs, setAvailableGraphs] = useState<string[]>([]);
  
  // Fetch graphs
  const fetchGraphs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/graphs`);
      if (!response.ok) {
        throw new Error(`Failed to fetch graphs: ${response.statusText}`);
      }
      const graphsData = await response.json();
      setAvailableGraphs(graphsData.map((graph: { name: string }) => graph.name));
    } catch (err) {
      console.error('Error fetching available graphs:', err);
    }
  };

  // Fetch graphs on component mount
  useEffect(() => {
    fetchGraphs();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update query options
  const updateQueryOption = <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => {
    setQueryOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <div className="flex h-screen">
      <Sidebar 
        activeSection={activeSection} 
        onSectionChange={setActiveSection}
        className="h-screen"
      />
      
      <div className="flex-1 p-6 flex flex-col h-screen overflow-hidden">
        {/* Upload status is now handled by the AlertSystem */}
        
        {/* Error alerts now appear in the bottom-right via the AlertSystem */}
        
        {/* Documents Section */}
        {activeSection === 'documents' && (
          <div className="flex-1 flex flex-col h-full">
            <div className="flex justify-between items-center bg-white py-3 mb-4">
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
              <Dialog 
                open={showUploadDialog} 
                onOpenChange={(open) => {
                  setShowUploadDialog(open);
                  if (!open) resetUploadDialog();
                }}
              >
                <DialogTrigger asChild>
                  <Button onClick={() => setShowUploadDialog(true)}>
                    <Upload className="mr-2 h-4 w-4" /> Upload Document
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle>Upload Document</DialogTitle>
                    <DialogDescription>
                      Upload a file or text to your Morphik repository.
                    </DialogDescription>
                  </DialogHeader>
                  
                  <div className="grid gap-4 py-4">
                    <div className="flex gap-2">
                      <Button 
                        variant={uploadType === 'file' ? "default" : "outline"} 
                        onClick={() => setUploadType('file')}
                      >
                        File
                      </Button>
                      <Button 
                        variant={uploadType === 'batch' ? "default" : "outline"} 
                        onClick={() => setUploadType('batch')}
                      >
                        Batch Files
                      </Button>
                      <Button 
                        variant={uploadType === 'text' ? "default" : "outline"} 
                        onClick={() => setUploadType('text')}
                      >
                        Text
                      </Button>
                    </div>
                    
                    {uploadType === 'file' ? (
                      <div>
                        <Label htmlFor="file" className="block mb-2">File</Label>
                        <Input 
                          id="file" 
                          type="file" 
                          onChange={(e) => {
                            const files = e.target.files;
                            if (files && files.length > 0) {
                              setFileToUpload(files[0]);
                            }
                          }}
                        />
                      </div>
                    ) : uploadType === 'batch' ? (
                      <div>
                        <Label htmlFor="batchFiles" className="block mb-2">Select Multiple Files</Label>
                        <Input 
                          id="batchFiles" 
                          type="file" 
                          multiple
                          onChange={(e: ChangeEvent<HTMLInputElement>) => {
                            const files = e.target.files;
                            if (files && files.length > 0) {
                              setBatchFilesToUpload(Array.from(files));
                            }
                          }}
                        />
                        {batchFilesToUpload.length > 0 && (
                          <div className="mt-2">
                            <p className="text-sm font-medium mb-1">{batchFilesToUpload.length} files selected:</p>
                            <ScrollArea className="h-24 w-full rounded-md border p-2">
                              <ul className="text-xs">
                                {Array.from(batchFilesToUpload).map((file, index) => (
                                  <li key={index} className="py-1 border-b border-gray-100 last:border-0">
                                    {file.name} ({(file.size / 1024).toFixed(1)} KB)
                                  </li>
                                ))}
                              </ul>
                            </ScrollArea>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div>
                        <Label htmlFor="text" className="block mb-2">Text Content</Label>
                        <Textarea 
                          id="text" 
                          value={textContent} 
                          onChange={(e) => setTextContent(e.target.value)}
                          placeholder="Enter text content"
                          rows={6}
                        />
                      </div>
                    )}
                    
                    <div>
                      <Label htmlFor="metadata" className="block mb-2">Metadata (JSON)</Label>
                      <Textarea 
                        id="metadata" 
                        value={metadata} 
                        onChange={(e) => setMetadata(e.target.value)}
                        placeholder='{"key": "value"}'
                        rows={3}
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="rules" className="block mb-2">Rules (JSON)</Label>
                      <Textarea 
                        id="rules" 
                        value={rules} 
                        onChange={(e) => setRules(e.target.value)}
                        placeholder='[{"type": "metadata_extraction", "schema": {...}}]'
                        rows={3}
                      />
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="useColpali"
                        checked={useColpali}
                        onChange={(e) => setUseColpali(e.target.checked)}
                      />
                      <Label htmlFor="useColpali">Use Colpali</Label>
                    </div>
                  </div>
                  
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setShowUploadDialog(false)}>
                      Cancel
                    </Button>
                    <Button 
                      onClick={uploadType === 'file' ? handleFileUpload : uploadType === 'batch' ? handleBatchFileUpload : handleTextUpload}
                      disabled={loading}
                    >
                      {loading ? 'Uploading...' : 'Upload'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>

            {loading && !documents.length ? (
              <div className="text-center py-8 flex-1">Loading documents...</div>
            ) : documents.length > 0 ? (
              <div className="flex flex-col md:flex-row gap-4 flex-1">
                <div className="w-full md:w-2/3">
                  {/* Document List */}
                  <div className="border rounded-md">
                    <div className="bg-gray-100 border-b p-3 font-medium sticky top-0">
                      <div className="grid grid-cols-12">
                        <div className="col-span-1 flex items-center justify-center">
                          <Checkbox
                            id="select-all-documents"
                            checked={getSelectAllState()}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedDocuments(documents.map(doc => doc.external_id));
                              } else {
                                setSelectedDocuments([]);
                              }
                            }}
                            aria-label="Select all documents"
                          />
                        </div>
                        <div className="col-span-4">Filename</div>
                        <div className="col-span-3">Type</div>
                        <div className="col-span-4">ID</div>
                      </div>
                    </div>

                    <ScrollArea className="h-[calc(100vh-200px)]">
                      {documents.map((doc) => (
                        <div 
                          key={doc.external_id}
                          onClick={() => handleDocumentClick(doc)}
                          className="grid grid-cols-12 p-3 cursor-pointer hover:bg-gray-50 border-b"
                        >
                          <div className="col-span-1 flex items-center justify-center">
                            <Checkbox 
                              id={`doc-${doc.external_id}`}
                              checked={selectedDocuments.includes(doc.external_id)}
                              onCheckedChange={(checked) => handleCheckboxChange(checked, doc.external_id)}
                              onClick={(e) => e.stopPropagation()}
                              aria-label={`Select ${doc.filename || 'document'}`}
                            />
                          </div>
                          <div className="col-span-4 flex items-center">
                            {doc.filename || 'N/A'}
                            {doc.external_id === selectedDocument?.external_id && (
                              <Badge variant="outline" className="ml-2">Selected</Badge>
                            )}
                          </div>
                          <div className="col-span-3">
                            <Badge variant="secondary">
                              {doc.content_type.split('/')[0]}
                            </Badge>
                          </div>
                          <div className="col-span-4 font-mono text-xs">
                            {doc.external_id.substring(0, 8)}...
                          </div>
                        </div>
                      ))}
                    </ScrollArea>
                  </div>
                </div>
                
                <div className="w-full md:w-1/3">
                  {selectedDocument ? (
                    <div className="border rounded-lg">
                      <div className="bg-gray-50 px-4 py-3 border-b sticky top-0">
                        <h3 className="text-lg font-semibold">Document Details</h3>
                      </div>
                      
                      <ScrollArea className="h-[calc(100vh-200px)]">
                        <div className="p-4 space-y-4">
                          <div>
                            <h3 className="font-medium mb-1">Filename</h3>
                            <p>{selectedDocument.filename || 'N/A'}</p>
                          </div>
                          
                          <div>
                            <h3 className="font-medium mb-1">Content Type</h3>
                            <Badge>{selectedDocument.content_type}</Badge>
                          </div>
                          
                          <div>
                            <h3 className="font-medium mb-1">Document ID</h3>
                            <p className="font-mono text-xs">{selectedDocument.external_id}</p>
                          </div>
                          
                          <Accordion type="single" collapsible>
                            <AccordionItem value="metadata">
                              <AccordionTrigger>Metadata</AccordionTrigger>
                              <AccordionContent>
                                <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                                  {JSON.stringify(selectedDocument.metadata, null, 2)}
                                </pre>
                              </AccordionContent>
                            </AccordionItem>
                            
                            <AccordionItem value="system-metadata">
                              <AccordionTrigger>System Metadata</AccordionTrigger>
                              <AccordionContent>
                                <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                                  {JSON.stringify(selectedDocument.system_metadata, null, 2)}
                                </pre>
                              </AccordionContent>
                            </AccordionItem>
                            
                            <AccordionItem value="additional-metadata">
                              <AccordionTrigger>Additional Metadata</AccordionTrigger>
                              <AccordionContent>
                                <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                                  {JSON.stringify(selectedDocument.additional_metadata, null, 2)}
                                </pre>
                              </AccordionContent>
                            </AccordionItem>
                          </Accordion>
                          
                          <div className="pt-4 border-t mt-4">
                            <Dialog>
                              <DialogTrigger asChild>
                                <Button variant="outline" size="sm" className="w-full border-red-500 text-red-500 hover:bg-red-50">
                                  Delete Document
                                </Button>
                              </DialogTrigger>
                              <DialogContent>
                                <DialogHeader>
                                  <DialogTitle>Delete Document</DialogTitle>
                                  <DialogDescription>
                                    Are you sure you want to delete this document? This action cannot be undone.
                                  </DialogDescription>
                                </DialogHeader>
                                <div className="py-3">
                                  <p className="font-medium">Document: {selectedDocument.filename || selectedDocument.external_id}</p>
                                  <p className="text-sm text-gray-500 mt-1">ID: {selectedDocument.external_id}</p>
                                </div>
                                <DialogFooter>
                                  <Button variant="outline" onClick={() => (document.querySelector('[data-state="open"] button[data-state="closed"]') as HTMLElement)?.click()}>Cancel</Button>
                                  <Button 
                                    variant="outline" 
                                    className="border-red-500 text-red-500 hover:bg-red-50"
                                    onClick={() => handleDeleteDocument(selectedDocument.external_id)}
                                    disabled={loading}
                                  >
                                    {loading ? 'Deleting...' : 'Delete'}
                                  </Button>
                                </DialogFooter>
                              </DialogContent>
                            </Dialog>
                          </div>
                        </div>
                      </ScrollArea>
                    </div>
                  ) : (
                    <div className="h-[calc(100vh-200px)] flex items-center justify-center p-8 border border-dashed rounded-lg">
                      <div className="text-center text-gray-500">
                        <Info className="mx-auto h-12 w-12 mb-2" />
                        <p>Select a document to view details</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 border border-dashed rounded-lg flex-1 flex items-center justify-center">
                <div>
                  <Upload className="mx-auto h-12 w-12 mb-2 text-gray-400" />
                  <p className="text-gray-500">No documents found. Upload your first document.</p>
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Search Section */}
        {activeSection === 'search' && (
          <Card className="flex-1 flex flex-col h-full">
            <CardHeader>
              <CardTitle>Search Documents</CardTitle>
              <CardDescription>
                Search across your documents to find relevant information.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <div className="space-y-4">
                <div className="flex gap-2">
                  <Input 
                    placeholder="Enter search query" 
                    value={searchQuery} 
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSearch();
                    }}
                  />
                  <Button onClick={handleSearch} disabled={loading}>
                    <Search className="mr-2 h-4 w-4" />
                    {loading ? 'Searching...' : 'Search'}
                  </Button>
                </div>
                
                <div>
                  <Dialog open={showSearchAdvanced} onOpenChange={setShowSearchAdvanced}>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="sm" className="flex items-center">
                        <Settings className="mr-2 h-4 w-4" />
                        Advanced Options
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle>Search Options</DialogTitle>
                        <DialogDescription>
                          Configure advanced search parameters
                        </DialogDescription>
                      </DialogHeader>
                      
                      <div className="grid gap-4 py-4">
                        <div>
                          <Label htmlFor="search-filters" className="block mb-2">Filters (JSON)</Label>
                          <Textarea 
                            id="search-filters" 
                            value={searchOptions.filters} 
                            onChange={(e) => updateSearchOption('filters', e.target.value)}
                            placeholder='{"key": "value"}'
                            rows={3}
                          />
                        </div>
                        
                        <div>
                          <Label htmlFor="search-k" className="block mb-2">
                            Number of Results (k): {searchOptions.k}
                          </Label>
                          <Input 
                            id="search-k" 
                            type="number" 
                            min={1} 
                            value={searchOptions.k}
                            onChange={(e) => updateSearchOption('k', parseInt(e.target.value) || 1)}
                          />
                        </div>
                        
                        <div>
                          <Label htmlFor="search-min-score" className="block mb-2">
                            Minimum Score: {searchOptions.min_score.toFixed(2)}
                          </Label>
                          <Input 
                            id="search-min-score" 
                            type="number" 
                            min={0} 
                            max={1} 
                            step={0.01}
                            value={searchOptions.min_score}
                            onChange={(e) => updateSearchOption('min_score', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <Label htmlFor="search-reranking">Use Reranking</Label>
                          <Switch 
                            id="search-reranking"
                            checked={searchOptions.use_reranking}
                            onCheckedChange={(checked) => updateSearchOption('use_reranking', checked)}
                          />
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <Label htmlFor="search-colpali">Use Colpali</Label>
                          <Switch 
                            id="search-colpali"
                            checked={searchOptions.embed_as_image}
                            onCheckedChange={(checked) => updateSearchOption('embed_as_image', checked)}
                          />
                        </div>
                      </div>
                      
                      <DialogFooter>
                        <Button onClick={() => setShowSearchAdvanced(false)}>Apply</Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
              
              <div className="mt-6 flex-1 overflow-hidden">
                {searchResults.length > 0 ? (
                  <div>
                    <h3 className="text-lg font-medium mb-4">Results ({searchResults.length})</h3>
                    
                    <ScrollArea className="h-[calc(100vh-320px)]">
                      <div className="space-y-6 pr-4">
                        {searchResults.map((result) => (
                          <Card key={`${result.document_id}-${result.chunk_number}`}>
                            <CardHeader className="pb-2">
                              <div className="flex justify-between items-start">
                                <div>
                                  <CardTitle className="text-base">
                                    {result.filename || `Document ${result.document_id.substring(0, 8)}...`}
                                  </CardTitle>
                                  <CardDescription>
                                    Chunk {result.chunk_number} • Score: {result.score.toFixed(2)}
                                  </CardDescription>
                                </div>
                                <Badge variant="outline">
                                  {result.content_type}
                                </Badge>
                              </div>
                            </CardHeader>
                            <CardContent>
                              {renderContent(result.content, result.content_type)}
                              
                              <Accordion type="single" collapsible className="mt-4">
                                <AccordionItem value="metadata">
                                  <AccordionTrigger className="text-sm">Metadata</AccordionTrigger>
                                  <AccordionContent>
                                    <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
                                      {JSON.stringify(result.metadata, null, 2)}
                                    </pre>
                                  </AccordionContent>
                                </AccordionItem>
                              </Accordion>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                ) : (
                  <div className="text-center py-16 border border-dashed rounded-lg">
                    <Search className="mx-auto h-12 w-12 mb-2 text-gray-400" />
                    <p className="text-gray-500">
                      {searchQuery.trim() ? 'No results found. Try a different query.' : 'Enter a query to search your documents.'}
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Chat Section */}
        {activeSection === 'chat' && (
          <Card className="h-[calc(100vh-12rem)] flex flex-col">
            <CardHeader>
              <CardTitle>Chat with Your Documents</CardTitle>
              <CardDescription>
                Ask questions about your documents and get AI-powered answers.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-grow overflow-hidden flex flex-col">
              <ScrollArea className="flex-grow pr-4 mb-4">
                {chatMessages.length > 0 ? (
                  <div className="space-y-4">
                    {chatMessages.map((message, index) => (
                      <div 
                        key={index} 
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div 
                          className={`max-w-3/4 p-3 rounded-lg ${
                            message.role === 'user' 
                              ? 'bg-primary text-primary-foreground' 
                              : 'bg-muted'
                          }`}
                        >
                          <div className="whitespace-pre-wrap">{message.content}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center text-gray-500">
                      <MessageSquare className="mx-auto h-12 w-12 mb-2" />
                      <p>Start a conversation about your documents</p>
                    </div>
                  </div>
                )}
              </ScrollArea>
              
              <div className="pt-4 border-t">
                <div className="space-y-4">
                  <div className="flex gap-2">
                    <Textarea 
                      placeholder="Ask a question..." 
                      value={chatQuery}
                      onChange={(e) => setChatQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleChat();
                        }
                      }}
                      className="min-h-10"
                    />
                    <Button onClick={handleChat} disabled={loading}>
                      {loading ? 'Sending...' : 'Send'}
                    </Button>
                  </div>
                  
                  <div>
                    <button
                      type="button" 
                      className="flex items-center text-sm text-gray-600 hover:text-gray-900"
                      onClick={() => setShowChatAdvanced(!showChatAdvanced)}
                    >
                      <Settings className="mr-1 h-4 w-4" />
                      Advanced Options
                      {showChatAdvanced ? <ChevronUp className="ml-1 h-4 w-4" /> : <ChevronDown className="ml-1 h-4 w-4" />}
                    </button>
                    
                    {showChatAdvanced && (
                      <div className="mt-3 p-4 border rounded-md bg-gray-50">
                        <div className="space-y-4">
                          <div>
                            <Label htmlFor="query-filters" className="block mb-2">Filters (JSON)</Label>
                            <Textarea 
                              id="query-filters" 
                              value={queryOptions.filters} 
                              onChange={(e) => updateQueryOption('filters', e.target.value)}
                              placeholder='{"key": "value"}'
                              rows={3}
                            />
                          </div>
                          
                          <div>
                            <Label htmlFor="query-k" className="block mb-2">
                              Number of Results (k): {queryOptions.k}
                            </Label>
                            <Input 
                              id="query-k" 
                              type="number" 
                              min={1} 
                              value={queryOptions.k}
                              onChange={(e) => updateQueryOption('k', parseInt(e.target.value) || 1)}
                            />
                          </div>
                          
                          <div>
                            <Label htmlFor="query-min-score" className="block mb-2">
                              Minimum Score: {queryOptions.min_score.toFixed(2)}
                            </Label>
                            <Input 
                              id="query-min-score" 
                              type="number" 
                              min={0} 
                              max={1} 
                              step={0.01}
                              value={queryOptions.min_score}
                              onChange={(e) => updateQueryOption('min_score', parseFloat(e.target.value) || 0)}
                            />
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <Label htmlFor="query-reranking">Use Reranking</Label>
                            <Switch 
                              id="query-reranking"
                              checked={queryOptions.use_reranking}
                              onCheckedChange={(checked) => updateQueryOption('use_reranking', checked)}
                            />
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <Label htmlFor="query-colpali">Use Colpali</Label>
                            <Switch 
                              id="query-colpali"
                              checked={queryOptions.embed_as_image}
                              onCheckedChange={(checked) => updateQueryOption('embed_as_image', checked)}
                            />
                          </div>
                          
                          <div>
                            <Label htmlFor="query-max-tokens" className="block mb-2">
                              Max Tokens: {queryOptions.max_tokens}
                            </Label>
                            <Input 
                              id="query-max-tokens" 
                              type="number" 
                              min={1} 
                              max={2048}
                              value={queryOptions.max_tokens}
                              onChange={(e) => updateQueryOption('max_tokens', parseInt(e.target.value) || 1)}
                            />
                          </div>
                          
                          <div>
                            <Label htmlFor="query-temperature" className="block mb-2">
                              Temperature: {queryOptions.temperature.toFixed(2)}
                            </Label>
                            <Input 
                              id="query-temperature" 
                              type="number" 
                              min={0} 
                              max={2} 
                              step={0.01}
                              value={queryOptions.temperature}
                              onChange={(e) => updateQueryOption('temperature', parseFloat(e.target.value) || 0)}
                            />
                          </div>

                          <div>
                            <Label htmlFor="graphName" className="block mb-2">Knowledge Graph</Label>
                            <select
                              id="graphName"
                              className="w-full p-2 border rounded-md dark:bg-gray-800"
                              value={queryOptions.graph_name || ''}
                              onChange={(e) => setQueryOptions({
                                ...queryOptions,
                                graph_name: e.target.value || undefined
                              })}
                            >
                              <option value="">None (Standard RAG)</option>
                              {availableGraphs.map(graphName => (
                                <option key={graphName} value={graphName}>
                                  {graphName}
                                </option>
                              ))}
                            </select>
                            <p className="text-sm text-gray-500">
                              Select a knowledge graph to enhance your query with structured relationships
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <p className="text-xs text-gray-500 mt-2">
                  Press Enter to send, Shift+Enter for a new line
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Notebooks Section */}
        {activeSection === 'notebooks' && (
          <NotebookSection apiBaseUrl={API_BASE_URL} />
        )}

        {/* Graphs Section */}
        {activeSection === 'graphs' && (
          <div className="space-y-4">
            <div className="flex justify-end items-center">
              {queryOptions.graph_name && (
                <Badge variant="outline" className="bg-blue-50 px-3 py-1">
                  Current Query Graph: {queryOptions.graph_name}
                </Badge>
              )}
            </div>
            
            <GraphSection apiBaseUrl={API_BASE_URL} />
          </div>
        )}
      </div>
    </div>
  );
};

export default MorphikUI;