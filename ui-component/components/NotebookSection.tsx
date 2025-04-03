"use client";

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  AlertCircle, 
  Plus,
  Book,
  FileText,
  Trash,
  MessageSquare,
  Settings,
  Info,
  File,
  ChevronDown,
  ChevronUp,
  ChevronLeft
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
// Removed unused accordion imports
import OpenAI from 'openai';
// Import OpenAI types for chat completions
import type { ChatCompletionContentPart } from 'openai/resources';
import { 
  Notebook, 
  loadNotebooksFromAPI,
  saveNotebooksToAPI,
  loadNotebooksFromLocalStorage,
  saveNotebooksToLocalStorage
} from '@/lib/notebooks-data';

interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, unknown>;
  system_metadata: Record<string, unknown>;
  additional_metadata: Record<string, unknown>;
}

interface ChunkResult {
  content: string;
  score: number;
  document_id: string;
  chunk_number: number;
  metadata: Record<string, unknown>;
  content_type: string;
  filename?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatOptions {
  filters: string;
  k: number;
  min_score: number;
  use_reranking: boolean;
  use_colpali: boolean;
  max_tokens: number;
  temperature: number;
  model_provider: 'openai' | 'claude' | 'ollama';
  model: string;
  custom_model?: string;
  graph_name?: string;
}

interface NotebookSectionProps {
  apiBaseUrl: string;
}

const NotebookSection: React.FC<NotebookSectionProps> = ({ apiBaseUrl }) => {
  // State variables
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [selectedNotebook, setSelectedNotebook] = useState<Notebook | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [notebookDocuments, setNotebookDocuments] = useState<Document[]>([]);
  const [activeTab, setActiveTab] = useState('list');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Create Notebook states
  const [notebookName, setNotebookName] = useState('');
  const [notebookDescription, setNotebookDescription] = useState('');

  // Chat states
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatQuery, setChatQuery] = useState('');
  const [showChatAdvanced, setShowChatAdvanced] = useState(false);
  const [chatOptions, setChatOptions] = useState<ChatOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    use_colpali: true,
    max_tokens: 500,
    temperature: 0.7,
    model_provider: 'openai',
    model: 'gpt-4o'
  });
  const [chatResults, setChatResults] = useState<ChunkResult[]>([]);
  
  // Document selection states
  const [documentsToAdd, setDocumentsToAdd] = useState<string[]>([]);
  const [documentFilter, setDocumentFilter] = useState('');
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [uploadType, setUploadType] = useState<'file' | 'text'>('file');
  const [textContent, setTextContent] = useState('');
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  const [metadata, setMetadata] = useState('{}');
  const [rules, setRules] = useState('[]');
  const [useColpali, setUseColpali] = useState(true);
  
  // Background upload states
  interface UploadTask {
    id: string;
    name: string;
    status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
    progress: number;
    notebookId: string;
    error?: string;
    type: 'file' | 'text';
    startTime: Date;
  }
  const [uploadTasks, setUploadTasks] = useState<UploadTask[]>([]);
  const [showUploadStatus, setShowUploadStatus] = useState(false);
  
  // Toast notification
  interface Toast {
    id: string;
    type: 'success' | 'error' | 'info';
    message: string;
    duration?: number;
  }
  const [toasts, setToasts] = useState<Toast[]>([]);
  
  // Show a toast notification
  const showToast = (type: 'success' | 'error' | 'info', message: string, duration = 5000) => {
    const id = `toast-${Date.now()}`;
    const newToast: Toast = { id, type, message, duration };
    
    setToasts(prev => [...prev, newToast]);
    
    // Auto-remove toast after duration
    if (duration > 0) {
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id));
      }, duration);
    }
    
    return id;
  };
  
  // Dialog states
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showSourceDetail, setShowSourceDetail] = useState(false);
  const [selectedSource, setSelectedSource] = useState<ChunkResult | null>(null);
  
  // API keys and model configuration
  const [openaiApiKey, setOpenaiApiKey] = useState<string>('');
  const [claudeApiKey, setClaudeApiKey] = useState<string>('');
  const [ollamaUrl, setOllamaUrl] = useState<string>('http://localhost:11434');
  const [showApiKeyDialog, setShowApiKeyDialog] = useState(false);
  
  // Available models
  const openaiModels = [
    { id: 'gpt-4o', name: 'GPT-4o' },
    { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
    { id: 'o3-mini', name: 'o3 Mini' },
    { id: 'other', name: 'Other (Custom)' },
  ];
  
  const claudeModels = [
    { id: 'claude-3-7-sonnet-latest', name: 'Claude 3.7 Sonnet' },
    { id: 'claude-3-5-haiku-latest', name: 'Claude 3.5 Haiku' },
    { id: 'other', name: 'Other (Custom)' },
  ];
  
  // For Ollama, we'll use a text input instead of a dropdown
  
  // Auth token - in a real application, you would get this from your auth system
  const authToken = 'YOUR_AUTH_TOKEN';
  
  // Check if API keys are available
  const getOpenAIClient = useCallback(() => {
    if (!openaiApiKey) {
      throw new Error('OpenAI API key is not set');
    }
    
    return new OpenAI({
      apiKey: openaiApiKey,
      dangerouslyAllowBrowser: true // Only for demo purposes
    });
  }, [openaiApiKey]);
  
  const getClaudeClient = useCallback(() => {
    if (!claudeApiKey) {
      throw new Error('Claude API key is not set');
    }
    
    return new OpenAI({
      apiKey: claudeApiKey,
      baseURL: 'https://api.anthropic.com/v1/',
      dangerouslyAllowBrowser: true // Only for demo purposes
    });
  }, [claudeApiKey]);
  
  const getOllamaApiUrl = useCallback(() => {
    if (!ollamaUrl) {
      throw new Error('Ollama URL is not set');
    }
    
    console.log('[DEBUG] Using Ollama API URL:', ollamaUrl);
    return ollamaUrl;
  }, [ollamaUrl]);

  // Load notebooks and API keys on component mount - Using a dedicated state to track loading
  const [isLoadingNotebooks, setIsLoadingNotebooks] = useState(false);
  
  // Default notebooks in case loading fails completely
  const DEFAULT_NOTEBOOKS: Notebook[] = [
    {
      id: "nb_default_1",
      name: "Research Papers",
      description: "Collection of scientific papers and research documents",
      created_at: "2023-01-15T12:00:00Z"
    },
    {
      id: "nb_default_2",
      name: "Project Documentation", 
      description: "Technical specifications and project documents",
      created_at: "2023-01-20T14:30:00Z"
    }
  ];
  
  useEffect(() => {
    // Only run once on mount
    const loadData = async () => {
      // Don't attempt to load if we're already loading
      if (isLoadingNotebooks) return;
      
      try {
        setIsLoadingNotebooks(true);
        // Prevent auto-save during initial load
        shouldSaveNotebooks.current = false;
        
        // First try to load from the file-based API
        console.log('Loading notebooks from API');
        const apiNotebooks = await loadNotebooksFromAPI();
        if (apiNotebooks && apiNotebooks.length > 0) {
          console.log('Loaded notebooks from file-based API:', apiNotebooks.length);
          setNotebooks(apiNotebooks);
        } else {
          // Fall back to localStorage if API fails
          const localNotebooks = loadNotebooksFromLocalStorage();
          if (localNotebooks && localNotebooks.length > 0) {
            console.log('Loaded notebooks from localStorage:', localNotebooks.length);
            setNotebooks(localNotebooks);
            
            // Save to the file API for future use, but don't trigger the useEffect
            await saveNotebooksToAPI(localNotebooks);
          } else {
            console.log('Using default notebooks');
            // Both API and localStorage failed, use the default notebooks
            setNotebooks(DEFAULT_NOTEBOOKS);
          }
        }
      } catch (error) {
        console.error('Error loading notebooks:', error);
        // Use default notebooks in case of error
        setNotebooks(DEFAULT_NOTEBOOKS);
      } finally {
        setIsLoadingNotebooks(false);
      }
      
      // Load API keys
      const storedOpenaiApiKey = localStorage.getItem('openai_api_key');
      if (storedOpenaiApiKey) {
        setOpenaiApiKey(storedOpenaiApiKey);
      }
      
      const storedClaudeApiKey = localStorage.getItem('claude_api_key');
      if (storedClaudeApiKey) {
        setClaudeApiKey(storedClaudeApiKey);
      }
      
      const storedOllamaUrl = localStorage.getItem('ollama_url');
      if (storedOllamaUrl) {
        setOllamaUrl(storedOllamaUrl);
      }
      
      // If no API keys are set, show the dialog
      if (!storedOpenaiApiKey && !storedClaudeApiKey && !storedOllamaUrl) {
        setShowApiKeyDialog(true);
      }
    };
    
    loadData();
    // We need these dependencies for ESLint, but we only want this to run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save notebooks to both file API and localStorage when they change
  // Using a ref to track whether notebooks were changed programmatically 
  // to avoid infinite loop with the load/save cycle
  const notebooksRef = useRef<Notebook[]>([]);
  const shouldSaveNotebooks = useRef(false);
  
  useEffect(() => {
    // Skip if we're currently loading notebooks or if notebooks are empty
    if (isLoadingNotebooks || notebooks.length === 0) {
      return;
    }
    
    // Only save if notebooks changed from user actions AND we're not in the initial load
    if (shouldSaveNotebooks.current) {
      console.log('Saving notebooks after user action');
      
      // Check if the notebooks array actually changed
      const prevNotebooksStr = JSON.stringify(notebooksRef.current);
      const currNotebooksStr = JSON.stringify(notebooks);
      
      if (prevNotebooksStr !== currNotebooksStr) {
        // Save to both storage mechanisms for redundancy
        saveNotebooksToAPI(notebooks)
          .then(success => {
            if (!success) {
              console.warn('Failed to save notebooks to API, falling back to localStorage');
            }
            // Always save to localStorage as a backup
            saveNotebooksToLocalStorage(notebooks);
          })
          .catch(error => {
            console.error('Error saving notebooks to API:', error);
            // Save to localStorage as fallback
            saveNotebooksToLocalStorage(notebooks);
          });
      }
    } else {
      // First time - enable saving for subsequent changes
      shouldSaveNotebooks.current = true;
    }
    
    // Update ref with current notebooks
    notebooksRef.current = notebooks;
  }, [notebooks, isLoadingNotebooks]);
  
  // Notebooks will be automatically persisted and loaded from the data store
  
  // Save API keys
  const saveApiKeys = () => {
    if (openaiApiKey) {
      localStorage.setItem('openai_api_key', openaiApiKey);
    }
    
    if (claudeApiKey) {
      localStorage.setItem('claude_api_key', claudeApiKey);
    }
    
    if (ollamaUrl) {
      localStorage.setItem('ollama_url', ollamaUrl);
    }
    
    setShowApiKeyDialog(false);
  };

  // Fetch all documents
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      
      // Define headers inside the callback to avoid dependency issues
      const headers = {
        'Authorization': authToken
      };
      
      const response = await fetch(`${apiBaseUrl}/documents`, {
        method: 'POST',
        headers: {
          ...headers,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.statusText}`);
      }
      
      const data = await response.json();
      setDocuments(data);
      
      // If a notebook is selected, update its documents as well
      if (selectedNotebook) {
        updateNotebookDocuments(selectedNotebook, data);
      }
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error fetching documents: ${error.message}`);
      console.error('Error fetching documents:', err);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, selectedNotebook]);

  // Update notebook documents based on metadata
  const updateNotebookDocuments = (notebook: Notebook, allDocs: Document[]) => {
    const notebookDocs = allDocs.filter(doc => {
      // Check if the document has the notebook metadata tag
      return doc.metadata && 
             typeof doc.metadata === 'object' && 
             'notebook' in doc.metadata && 
             doc.metadata.notebook === notebook.name;
    });
    
    setNotebookDocuments(notebookDocs);
  };

  // Fetch documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Create new notebook
  const handleCreateNotebook = () => {
    if (!notebookName.trim()) {
      setError('Please enter a notebook name');
      return;
    }

    // Check if notebook name already exists
    if (notebooks.some(nb => nb.name === notebookName)) {
      setError('A notebook with this name already exists');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Generate a unique ID for the notebook
      const id = `nb_${Date.now()}`;
      
      // Create a new notebook object
      const newNotebook: Notebook = {
        id,
        name: notebookName,
        description: notebookDescription,
        created_at: new Date().toISOString()
      };
      
      // Add the notebook to the list and ensure it's saved
      shouldSaveNotebooks.current = true;
      setNotebooks(prev => [...prev, newNotebook]);
      
      // Select the new notebook
      setSelectedNotebook(newNotebook);
      setNotebookDocuments([]);
      
      // Update chat options to include the notebook filter
      setChatOptions(prev => ({
        ...prev,
        filters: JSON.stringify({ notebook: newNotebook.name })
      }));
      
      // Reset form
      setNotebookName('');
      setNotebookDescription('');
      
      // Switch to documents tab
      setActiveTab('documents');
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error creating notebook: ${error.message}`);
      console.error('Error creating notebook:', err);
    } finally {
      setLoading(false);
    }
  };

  // Remove document from notebook by updating its metadata
  const handleRemoveDocument = async (documentId: string) => {
    if (!selectedNotebook) {
      setError('No notebook selected');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Find the document
      const document = notebookDocuments.find(doc => doc.external_id === documentId);
      if (!document) {
        throw new Error('Document not found');
      }
      
      // Remove the notebook from metadata
      const updatedMetadata = { ...document.metadata };
      delete updatedMetadata.notebook;
      
      // Update the document metadata
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}/update_metadata`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedMetadata)
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update document metadata: ${response.statusText}`);
      }
      
      // Refresh documents list
      await fetchDocuments();
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error removing document: ${error.message}`);
      console.error('Error removing document:', err);
    } finally {
      setLoading(false);
    }
  };

  // Delete notebook
  const handleDeleteNotebook = () => {
    if (!selectedNotebook) {
      setError('No notebook selected');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Remove the notebook from the list and ensure it's saved
      shouldSaveNotebooks.current = true;
      setNotebooks(prev => prev.filter(nb => nb.id !== selectedNotebook.id));
      
      // Reset selected notebook
      setSelectedNotebook(null);
      setNotebookDocuments([]);
      setChatMessages([]);
      setChatResults([]);
      
      // Switch to list tab
      setActiveTab('list');
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error deleting notebook: ${error.message}`);
      console.error('Error deleting notebook:', err);
    } finally {
      setLoading(false);
    }
  };

  // Add documents to notebook by updating their metadata
  const handleAddDocuments = async () => {
    if (!selectedNotebook) {
      setError('No notebook selected');
      return;
    }

    if (documentsToAdd.length === 0) {
      setError('Please select at least one document to add');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Update each document's metadata to include the notebook
      for (const documentId of documentsToAdd) {
        const document = documents.find(doc => doc.external_id === documentId);
        if (!document) continue;
        
        const updatedMetadata = { 
          ...document.metadata,
          notebook: selectedNotebook.name 
        };
        
        // Update the document metadata
        const response = await fetch(`${apiBaseUrl}/documents/${documentId}/update_metadata`, {
          method: 'POST',
          headers: {
            'Authorization': authToken,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(updatedMetadata)
        });
        
        if (!response.ok) {
          throw new Error(`Failed to update document metadata: ${response.statusText}`);
        }
      }
      
      // Refresh documents list
      await fetchDocuments();
      
      // Reset selection
      setDocumentsToAdd([]);
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error adding documents: ${error.message}`);
      console.error('Error adding documents:', err);
    } finally {
      setLoading(false);
    }
  };

  // Identify if chunk is an image based on metadata or content
  const isImageChunk = (chunk: ChunkResult): boolean => {
    // 1. First check for explicit is_image flag in metadata
    const hasIsImageFlag = chunk.metadata && 
           typeof chunk.metadata === 'object' && 
           'is_image' in chunk.metadata && 
           chunk.metadata.is_image === true;
    
    if (hasIsImageFlag) {
      console.log('[DEBUG] Found image chunk with is_image flag:', chunk.document_id);
      return true;
    }
    
    // 2. Check content_type for image
    const hasImageContentType = chunk.content_type && 
                               chunk.content_type.startsWith('image/');
    
    if (hasImageContentType) {
      console.log('[DEBUG] Found image chunk with image content type:', chunk.content_type);
      return true;
    }
    
    // 3. Check if content starts with data URI for image
    const hasDataImageUri = chunk.content && 
                           (chunk.content.startsWith('data:image/'));
    
    if (hasDataImageUri) {
      console.log('[DEBUG] Found image chunk with data:image URI:', chunk.document_id);
      return true;
    }
    
    // 4. As a final check, see if this looks like base64 content
    if (chunk.content && chunk.content.length > 1000) {
      // Check for base64 pattern (long string of base64 characters)
      const isLongBase64String = /^[A-Za-z0-9+/]{1000,}=*$/.test(chunk.content.substring(0, 1100));
      
      if (isLongBase64String) {
        console.log('[DEBUG] Treating long base64 content as image:', {
          documentId: chunk.document_id,
          contentLength: chunk.content.length,
          contentStart: chunk.content.substring(0, 30)
        });
        
        // IMPORTANT: Here we modify the chunk to properly mark it as an image
        // This ensures it will be handled correctly
        if (!chunk.metadata) chunk.metadata = {};
        (chunk.metadata as Record<string, boolean>).is_image = true;
        
        // Set content type if missing
        if (!chunk.content_type) {
          chunk.content_type = 'image/jpeg';
        }
        
        return true;
      }
    }
    
    return false;
  };

  // Format chunks as context for the LLM
  const formatChunksAsContext = (chunks: ChunkResult[]) => {
    // Filter out image chunks as they'll be handled separately
    const textChunks = chunks.filter(chunk => !isImageChunk(chunk));
    
    // Log any large content to help identify potential base64 data being included
    textChunks.forEach(chunk => {
      if (chunk.content && chunk.content.length > 10000) {
        console.log(`[DEBUG] Large text chunk detected (${chunk.content.length} chars)`, {
          chunkId: chunk.document_id,
          contentStart: chunk.content.substring(0, 50),
          contentEnd: chunk.content.substring(chunk.content.length - 50)
        });
      }
    });
    
    return textChunks.map((chunk, index) => {
      let source = '';
      if (chunk.filename) {
        source = `Document: ${chunk.filename}`;
      } else {
        source = `Document ID: ${chunk.document_id.substring(0, 8)}...`;
      }
      
      // Extra check to avoid including very large content (likely base64 data)
      const content = chunk.content.length > 50000 ? 
        `[Content too large (${chunk.content.length} chars) - likely an image that wasn't properly identified]` : 
        chunk.content;
      
      return `
[SOURCE ${index + 1}]: ${source}
${content}
`.trim();
    }).join('\n\n');
  };

  // Chat with notebook
  const handleChatWithNotebook = async () => {
    if (!selectedNotebook) {
      setError('No notebook selected');
      return;
    }

    if (!chatQuery.trim()) {
      setError('Please enter a query');
      return;
    }
    
    // Check if we have the required API key or URL
    if ((chatOptions.model_provider === 'openai' && !openaiApiKey) || 
        (chatOptions.model_provider === 'claude' && !claudeApiKey) ||
        (chatOptions.model_provider === 'ollama' && !ollamaUrl)) {
      setShowApiKeyDialog(true);
      setError(`Please provide a valid ${
        chatOptions.model_provider === 'openai' ? 'OpenAI API key' : 
        chatOptions.model_provider === 'claude' ? 'Claude API key' : 
        'Ollama URL'
      }`);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Add user message to chat
      const userMessage: ChatMessage = { role: 'user', content: chatQuery };
      setChatMessages(prev => [...prev, userMessage]);
      
      // Parse the filters to ensure notebook filter is included
      let filters = { notebook: selectedNotebook.name };
      try {
        const parsedFilters = JSON.parse(chatOptions.filters);
        filters = { ...parsedFilters, notebook: selectedNotebook.name };
      } catch {
        // Keep default filters
      }
      
      // First, get relevant chunks
      const chunksResponse = await fetch(`${apiBaseUrl}/retrieve/chunks`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: chatQuery,
          filters,
          k: chatOptions.k,
          min_score: chatOptions.min_score,
          use_reranking: chatOptions.use_reranking,
          use_colpali: chatOptions.use_colpali,
        })
      });
      
      if (!chunksResponse.ok) {
        throw new Error(`Failed to retrieve chunks: ${chunksResponse.statusText}`);
      }
      
      const chunksData = await chunksResponse.json();
      setChatResults(chunksData);
      
      // Separate image chunks from text chunks
      const allImageChunks = chunksData.filter((chunk: ChunkResult) => isImageChunk(chunk));
      console.log('[DEBUG] Number of image chunks identified:', allImageChunks.length);
      
      // IMPORTANT: Limit to only the top 4 images to avoid token limits
      const imageChunks = allImageChunks.slice(0, 4);
      console.log('[DEBUG] Using up to 4 top images');
      
      if (imageChunks.length > 0) {
        console.log('[DEBUG] First image chunk metadata:', JSON.stringify(imageChunks[0].metadata));
        console.log('[DEBUG] First image chunk content type:', imageChunks[0].content_type);
        console.log('[DEBUG] First image chunk content length:', imageChunks[0].content?.length || 0);
        // Log first 30 chars to verify if it's base64 or already has a data URI
        console.log('[DEBUG] First image chunk content start:', imageChunks[0].content?.substring(0, 30));
      }
      
      // Format the text chunks as context
      const context = formatChunksAsContext(chunksData);
      console.log('[DEBUG] Context text length:', context.length);
      
      // Create the system message with instructions
      const systemMessage = `You are a helpful assistant answering questions based on the provided context. 
Use the information in the context to answer the user's question. If the answer is not in the context, say you don't know.
Be concise and only use information from the provided context.
${imageChunks.length > 0 ? `You will also be provided with ${imageChunks.length} relevant image${imageChunks.length > 1 ? 's' : ''} from the documents.` : ''}`;
      
      // Create the prompt with context for text
      const textPrompt = `
Context information is below.
---------------------
${context}
---------------------

Given the context information and not prior knowledge, answer the question: ${chatQuery}`;
    
      // Call the selected model API
      let completionText = '';
      
      if (chatOptions.model_provider === 'openai') {
        // Create an instance of the OpenAI client
        const openai = getOpenAIClient();
        
        if (imageChunks.length > 0) {
          // Create a new content array following OpenAI's multimodal format exactly
          console.log('[DEBUG] Creating multimodal message array for OpenAI');
          const contentArray: ChatCompletionContentPart[] = [];
          
          // First element is the text context
          contentArray.push({
            type: "text",
            text: textPrompt
          });
          
          // Process image chunks - limit to top 4 only
          imageChunks.forEach((chunk: ChunkResult, index: number) => {
            console.log(`[DEBUG] Processing image ${index} for OpenAI`);
            
            // Determine image format
            let imageFormat = 'jpeg'; // Default fallback
            if (chunk.content_type && chunk.content_type.includes('/')) {
              imageFormat = chunk.content_type.split('/')[1];
            }
            console.log(`[DEBUG] Using image format: ${imageFormat}`);
            
            // Create valid data URI
            let imageUrl;
            if (chunk.content.startsWith('data:image/')) {
              // Already has data URI prefix
              imageUrl = chunk.content;
              console.log(`[DEBUG] Image ${index} already has data URI`);
            } else {
              // Create data URI from base64
              imageUrl = `data:image/${imageFormat};base64,${chunk.content}`;
              console.log(`[DEBUG] Created data URI for image ${index}`);
            }
            
            // Add image in exact OpenAI format - FOLLOW THE EXAMPLE EXACTLY
            console.log(`[DEBUG] Adding image ${index} to content array`);
            contentArray.push({
              type: "image_url",
              image_url: {
                url: imageUrl
              }
            });
          });
          
          // Log the content array structure (without the actual base64 data)
          // console.log('[DEBUG] Content array structure:', contentArray.map(item => {
          //   if (item.type === 'image_url') {
          //     return { 
          //       type: 'image_url', 
          //       image_url: { 
          //         url: item.image_url.url.substring(0, 30) + '... [truncated]' 
          //       } 
          //     };
          //   }
          //   return item;
          // }));
          
          let response;
          try {
            // Add a special function to check for base64 content in text
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const checkForLargeTextContent = (content: any) => {
              if (Array.isArray(content)) {
                return content.some(item => {
                  if (item.type === 'text' && item.text && item.text.length > 10000) {
                    console.log(`[DEBUG] Found very large text content: ${item.text.length} chars`);
                    
                    // Check if it looks like base64
                    const sample = item.text.substring(0, 50);
                    const isLikelyBase64 = /^[A-Za-z0-9+/=]{30,}/.test(sample);
                    
                    if (isLikelyBase64) {
                      console.log('[DEBUG] Text content appears to be base64:', sample);
                      return true;
                    }
                  }
                  return false;
                });
              }
              return false;
            };
            
            // Check if we might have base64 data in text
            const hasLargeTextContent = checkForLargeTextContent(contentArray);
            console.log('[DEBUG] Content has suspicious large text:', hasLargeTextContent);
                
            // For debugging - output total size
            let totalTextSize = 0;
            contentArray.forEach(item => {
              if (item.type === 'text') totalTextSize += item.text?.length || 0;
            });
            console.log(`[DEBUG] Total text content size: ${totalTextSize} chars`);
            
            // Important: Log image information
            const imageItems = contentArray.filter(item => item.type === 'image_url');
            console.log(`[DEBUG] Sending ${imageItems.length} images to OpenAI API`);
            
            // Call OpenAI API with multimodal content
            console.log('[DEBUG] Calling OpenAI API with multimodal content');
            response = await openai.chat.completions.create({
              model: chatOptions.model,
              messages: [
                { role: "system", content: systemMessage },
                { role: "user", content: contentArray as unknown as ChatCompletionContentPart[] } // Type assertion with proper typing
              ],
              temperature: chatOptions.temperature,
              max_tokens: chatOptions.max_tokens,
            });
            console.log('[DEBUG] OpenAI API call successful');
          } catch (error) {
            console.error('[DEBUG] OpenAI API error:', error);
            throw error;
          }
          
          completionText = response.choices[0].message.content || 'No response generated';
        } else {
          // Text-only query
          const response = await openai.chat.completions.create({
            model: chatOptions.model,
            messages: [
              { role: "system", content: systemMessage },
              { role: "user", content: textPrompt }
            ],
            temperature: chatOptions.temperature,
            max_tokens: chatOptions.max_tokens,
          });
          
          completionText = response.choices[0].message.content || 'No response generated';
        }
      } else if (chatOptions.model_provider === 'claude') {
        // Create an instance of the Claude client (using OpenAI compatible interface)
        const claude = getClaudeClient();
        
        if (imageChunks.length > 0) {
          // Create a new content array following the exact same format for Claude
          console.log('[DEBUG] Creating multimodal message array for Claude');
          const contentArray = [];
          
          // First element is the text context
          contentArray.push({
            type: "text",
            text: textPrompt
          });
          
          // Process images (up to 4) - use the exact same format as OpenAI
          imageChunks.forEach((chunk: ChunkResult, index: number) => {
            console.log(`[DEBUG] Processing image ${index} for Claude`);
            
            // Determine image format
            let imageFormat = 'jpeg'; // Default fallback
            if (chunk.content_type && chunk.content_type.includes('/')) {
              imageFormat = chunk.content_type.split('/')[1];
            }
            console.log(`[DEBUG] Using image format for Claude: ${imageFormat}`);
            
            // Create valid data URI
            let imageUrl;
            if (chunk.content.startsWith('data:image/')) {
              // Already has data URI prefix
              imageUrl = chunk.content;
              console.log(`[DEBUG] Claude image ${index} already has data URI`);
            } else {
              // Create data URI from base64
              imageUrl = `data:image/${imageFormat};base64,${chunk.content}`;
              console.log(`[DEBUG] Created data URI for Claude image ${index}`);
            }
            
            // Add image in exact format - match OpenAI format
            contentArray.push({
              type: "image_url",
              image_url: {
                url: imageUrl
              }
            });
          });
          
          // Log the content array structure (without the actual base64 data)
          console.log('[DEBUG] Claude content array structure:', contentArray.map(item => {
            if (item.type === 'image_url') {
              // Use two-step type assertion to safely handle the image_url property
              const imageItem = item as unknown as { type: 'image_url', image_url: { url: string } };
              return { 
                type: 'image_url', 
                image_url: { 
                  url: imageItem.image_url.url.substring(0, 30) + '... [truncated]' 
                } 
              };
            }
            return item;
          }));
          
          let response;
          try {
            // Call Claude API with multimodal content
            console.log('[DEBUG] Calling Claude API with multimodal content');
            response = await claude.chat.completions.create({
              model: chatOptions.model,
              messages: [
                { role: "system", content: systemMessage },
                { role: "user", content: contentArray as unknown as ChatCompletionContentPart[] } // Type assertion with proper typing
              ],
              temperature: chatOptions.temperature,
              max_tokens: chatOptions.max_tokens,
            });
            console.log('[DEBUG] Claude API call successful');
          } catch (error) {
            console.error('[DEBUG] Claude API error:', error);
            throw error;
          }
          
          completionText = response.choices[0].message.content || 'No response generated';
        } else {
          // Text-only query
          const response = await claude.chat.completions.create({
            model: chatOptions.model,
            messages: [
              { role: "system", content: systemMessage },
              { role: "user", content: textPrompt }
            ],
            temperature: chatOptions.temperature,
            max_tokens: chatOptions.max_tokens,
          });
          
          completionText = response.choices[0].message.content || 'No response generated';
        }
      } else if (chatOptions.model_provider === 'ollama') {
        // Get the Ollama API URL for direct API calls
        getOllamaApiUrl(); // Just validate the URL is set
        
        console.log('[DEBUG] Using Ollama with model:', chatOptions.model);
        
        // Ollama supports images too but with a slightly different format
        if (imageChunks.length > 0) {
          // Create a new content array following OpenAI's format but adapting for Ollama
          console.log('[DEBUG] Creating multimodal message array for Ollama');
          const contentArray = [];
          
          // First element is the text context
          contentArray.push({
            type: "text",
            text: textPrompt
          });
          
          // Process only the first image as Ollama might have limitations with multiple images
          if (imageChunks.length > 0) {
            const chunk = imageChunks[0];
            console.log(`[DEBUG] Processing image for Ollama`);
            
            // Determine image format
            let imageFormat = 'jpeg'; // Default fallback
            if (chunk.content_type && chunk.content_type.includes('/')) {
              imageFormat = chunk.content_type.split('/')[1];
            }
            console.log(`[DEBUG] Using image format: ${imageFormat}`);
            
            // Create valid data URI
            let imageUrl;
            if (chunk.content.startsWith('data:image/')) {
              // Already has data URI prefix
              imageUrl = chunk.content;
              console.log(`[DEBUG] Image already has data URI`);
            } else {
              // Create data URI from base64
              imageUrl = `data:image/${imageFormat};base64,${chunk.content}`;
              console.log(`[DEBUG] Created data URI for image`);
            }
            
            // Add image in OpenAI format which Ollama supports
            contentArray.push({
              type: "image_url",
              image_url: {
                url: imageUrl
              }
            });
          }
          
          try {
            // Call Ollama API with content using native API
            console.log('[DEBUG] Calling Ollama API with content using native API');
            
            // Extract base64 image content
            let extractedBase64 = '';
            const chunk = imageChunks[0];
            
            if (chunk.content.startsWith('data:image/')) {
              // Extract base64 from data URI
              extractedBase64 = chunk.content.split(',')[1];
            } else {
              // Already have base64
              extractedBase64 = chunk.content;
            }
            
            // Format body for Ollama's native API
            const apiUrl = getOllamaApiUrl();
            const response = await fetch(`${apiUrl}/api/generate`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model: chatOptions.model,
                prompt: textPrompt,
                system: systemMessage,
                images: [extractedBase64],
                stream: false,
                options: {
                  temperature: chatOptions.temperature,
                  num_predict: chatOptions.max_tokens,
                }
              })
            });
            
            if (!response.ok) {
              throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
            }
            
            const responseData = await response.json();
            console.log('[DEBUG] Ollama API call successful');
            completionText = responseData.response || 'No response generated';
          } catch (error) {
            console.error('[DEBUG] Ollama API error:', error);
            // Fallback to text-only if multimodal fails with Ollama
            console.log('[DEBUG] Falling back to text-only query for Ollama');
            try {
              // Fallback to text-only using native API
              const apiUrl = getOllamaApiUrl();
              const response = await fetch(`${apiUrl}/api/generate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  model: chatOptions.model,
                  prompt: textPrompt,
                  system: systemMessage,
                  stream: false,
                  options: {
                    temperature: chatOptions.temperature,
                    num_predict: chatOptions.max_tokens,
                  }
                })
              });
              
              if (!response.ok) {
                throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
              }
              
              const responseData = await response.json();
              completionText = responseData.response || 'No response generated';
            } catch (fallbackError) {
              console.error('[DEBUG] Ollama fallback error:', fallbackError);
              
              // Check if it's a CORS error - use type assertion to safely access message property
              const errorWithMessage = fallbackError as { message?: string };
              if (errorWithMessage.message && errorWithMessage.message.includes('Failed to fetch')) {
                throw new Error(
                  'Unable to connect to Ollama. This might be a CORS issue. ' +
                  'Try running Ollama with CORS allowed: OLLAMA_ORIGINS=http://localhost:3000 ollama serve'
                );
              }
              
              throw fallbackError;
            }
          }
        } else {
          // Text-only query for Ollama using native API
          try {
            const apiUrl = getOllamaApiUrl();
            const response = await fetch(`${apiUrl}/api/chat`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model: chatOptions.model,
                messages: [
                  { role: "system", content: systemMessage },
                  { role: "user", content: textPrompt }
                ],
                stream: false,
                options: {
                  temperature: chatOptions.temperature,
                  num_predict: chatOptions.max_tokens,
                }
              })
            });
            
            if (!response.ok) {
              throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
            }
            
            const responseData = await response.json();
            completionText = responseData.message.content || 'No response generated';
          } catch (error) {
            console.error('[DEBUG] Ollama API error:', error);
            
            // Check if it's a CORS error - use type assertion to safely access message property
            const errorWithMessage = error as { message?: string };
            if (errorWithMessage.message && errorWithMessage.message.includes('Failed to fetch')) {
              throw new Error(
                'Unable to connect to Ollama. This might be a CORS issue. ' +
                'Try running Ollama with CORS allowed: OLLAMA_ORIGINS=http://localhost:3000 ollama serve'
              );
            }
            
            throw error;
          }
        }
      }
      
      // Add assistant response to chat
      const assistantMessage: ChatMessage = { role: 'assistant', content: completionText };
      setChatMessages(prev => [...prev, assistantMessage]);
      setChatQuery('');
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error in chat: ${error.message}`);
      console.error('Error in chat:', err);
    } finally {
      setLoading(false);
    }
  };

  // Handle notebook selection
  const handleNotebookSelect = (notebook: Notebook) => {
    setSelectedNotebook(notebook);
    
    // Update notebook documents based on current documents
    updateNotebookDocuments(notebook, documents);
    
    // Reset chat
    setChatMessages([]);
    setChatResults([]);
    
    // Update chat options to include the notebook filter
    setChatOptions(prev => ({
      ...prev,
      filters: JSON.stringify({ notebook: notebook.name })
    }));
    
    // Update metadata for uploads with the notebook name
    try {
      const metadataObj = JSON.parse(metadata);
      metadataObj.notebook = notebook.name;
      setMetadata(JSON.stringify(metadataObj, null, 2));
    } catch {
      setMetadata(JSON.stringify({ notebook: notebook.name }, null, 2));
    }
    
    // Switch to documents tab if we're on list
    if (activeTab === 'list') {
      setActiveTab('documents');
    }
  };

  // Handle file upload
  const handleFileUpload = () => {
    if (!fileToUpload) {
      setError('Please select a file to upload');
      return;
    }

    if (!selectedNotebook) {
      setError('No notebook selected');
      return;
    }

    try {
      setError(null);
      
      // Ensure metadata includes notebook name
      let metadataObj;
      try {
        metadataObj = JSON.parse(metadata);
        if (!metadataObj.notebook) {
          metadataObj.notebook = selectedNotebook.name;
        }
      } catch {
        metadataObj = { notebook: selectedNotebook.name };
      }
      
      // Create a new upload task
      const taskId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newTask: UploadTask = {
        id: taskId,
        name: fileToUpload.name,
        status: 'pending',
        progress: 0,
        notebookId: selectedNotebook.id,
        type: 'file',
        startTime: new Date()
      };
      
      setUploadTasks(prevTasks => [...prevTasks, newTask]);
      setShowUploadStatus(true);
      
      // Close dialog and reset form
      setShowUploadDialog(false);
      
      // Start the upload process in the background
      (async () => {
        try {
          // Update task status to uploading
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { ...task, status: 'uploading', progress: 10 } : task
            )
          );
          
          const formData = new FormData();
          formData.append('file', fileToUpload);
          formData.append('metadata', JSON.stringify(metadataObj));
          formData.append('rules', rules);
          
          const url = `${apiBaseUrl}/ingest/file${useColpali ? '?use_colpali=true' : ''}`;
          
          // Use XMLHttpRequest to track upload progress
          const xhr = new XMLHttpRequest();
          
          // Set up progress tracking
          xhr.upload.onprogress = (event) => {
            if (event.lengthComputable) {
              const progress = Math.round((event.loaded / event.total) * 50); // Upload is 50% of total progress
              setUploadTasks(prevTasks => 
                prevTasks.map(task => 
                  task.id === taskId ? { ...task, progress } : task
                )
              );
            }
          };
          
          // Create a promise to handle the response
          const uploadPromise = new Promise<void>((resolve, reject) => {
            xhr.onload = () => {
              if (xhr.status >= 200 && xhr.status < 300) {
                // Update task to processing (ingestion continues on server)
                setUploadTasks(prevTasks => 
                  prevTasks.map(task => 
                    task.id === taskId ? { ...task, status: 'processing', progress: 70 } : task
                  )
                );
                resolve();
              } else {
                reject(new Error(`Upload failed with status ${xhr.status}: ${xhr.statusText}`));
              }
            };
            
            xhr.onerror = () => {
              reject(new Error('Network error during upload'));
            };
          });
          
          // Set up request
          xhr.open('POST', url, true);
          xhr.setRequestHeader('Authorization', authToken);
          xhr.send(formData);
          
          // Wait for upload to complete
          await uploadPromise;
          
          // Simulate processing time with gradual progress updates
          const processingInterval = setInterval(() => {
            setUploadTasks(prevTasks => {
              const task = prevTasks.find(t => t.id === taskId);
              if (task && task.progress < 95 && task.status === 'processing') {
                return prevTasks.map(t => 
                  t.id === taskId ? { ...t, progress: t.progress + 5 } : t
                );
              } else {
                clearInterval(processingInterval);
                return prevTasks;
              }
            });
          }, 2000);
          
          // Wait a bit for processing (in reality, you'd poll the server for status)
          await new Promise(resolve => setTimeout(resolve, 5000));
          
          // Mark as completed and update document list
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { ...task, status: 'completed', progress: 100 } : task
            )
          );
          
          // Show a success toast
          showToast('success', `File "${fileToUpload.name}" successfully uploaded`);
          
          // Refresh documents list
          await fetchDocuments();
          
          // Clean up old tasks after some time
          setTimeout(() => {
            setUploadTasks(prevTasks => 
              prevTasks.filter(task => 
                task.id !== taskId || 
                (new Date().getTime() - new Date(task.startTime).getTime()) < 1000 * 60 * 30 // Keep for 30 minutes
              )
            );
          }, 1000 * 60 * 30); // 30 minutes
          
        } catch (err: unknown) {
          const error = err as Error;
          console.error('Error in background upload:', error);
          
          // Update task status to error
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { 
                ...task, 
                status: 'error', 
                error: error.message || 'Upload failed' 
              } : task
            )
          );
          
          // Show error toast
          showToast('error', `File upload failed: ${error.message || 'Unknown error'}`);
        }
      })();
      
      // Reset form for next upload
      setFileToUpload(null);
      setMetadata(JSON.stringify({ notebook: selectedNotebook.name }, null, 2));
      setRules('[]');
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error preparing upload: ${error.message}`);
      console.error('Error preparing upload:', err);
    }
  };

  // Handle text upload
  const handleTextUpload = () => {
    if (!textContent.trim()) {
      setError('Please enter text content');
      return;
    }

    if (!selectedNotebook) {
      setError('No notebook selected');
      return;
    }

    try {
      setError(null);
      
      // Ensure metadata includes notebook name
      let metadataObj;
      try {
        metadataObj = JSON.parse(metadata);
        if (!metadataObj.notebook) {
          metadataObj.notebook = selectedNotebook.name;
        }
      } catch {
        metadataObj = { notebook: selectedNotebook.name };
      }
      
      // Create a title based on first line or character limit
      const title = textContent.split('\n')[0].substring(0, 40) + (textContent.split('\n')[0].length > 40 ? '...' : '');
      
      // Create a new upload task
      const taskId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newTask: UploadTask = {
        id: taskId,
        name: title,
        status: 'pending',
        progress: 0,
        notebookId: selectedNotebook.id,
        type: 'text',
        startTime: new Date()
      };
      
      setUploadTasks(prevTasks => [...prevTasks, newTask]);
      setShowUploadStatus(true);
      
      // Close dialog and reset form
      setShowUploadDialog(false);
      
      // Start the upload process in the background
      (async () => {
        try {
          // Update task status to uploading
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { ...task, status: 'uploading', progress: 30 } : task
            )
          );
          
          const response = await fetch(`${apiBaseUrl}/ingest/text`, {
            method: 'POST',
            headers: {
              'Authorization': authToken,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              content: textContent,
              metadata: metadataObj,
              rules: JSON.parse(rules || '[]'),
              use_colpali: useColpali
            })
          });
          
          if (!response.ok) {
            throw new Error(`Failed to upload text: ${response.statusText}`);
          }
          
          // Update to processing
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { ...task, status: 'processing', progress: 70 } : task
            )
          );
          
          // Simulate processing time with gradual progress updates
          const processingInterval = setInterval(() => {
            setUploadTasks(prevTasks => {
              const task = prevTasks.find(t => t.id === taskId);
              if (task && task.progress < 95 && task.status === 'processing') {
                return prevTasks.map(t => 
                  t.id === taskId ? { ...t, progress: t.progress + 5 } : t
                );
              } else {
                clearInterval(processingInterval);
                return prevTasks;
              }
            });
          }, 1500);
          
          // Wait a bit for processing (in reality, you'd poll the server for status)
          await new Promise(resolve => setTimeout(resolve, 3000));
          
          // Mark as completed and update document list
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { ...task, status: 'completed', progress: 100 } : task
            )
          );
          
          // Show a success toast
          showToast('success', `Text "${title}" successfully uploaded`);
          
          // Refresh documents list
          await fetchDocuments();
          
          // Clean up old tasks after some time
          setTimeout(() => {
            setUploadTasks(prevTasks => 
              prevTasks.filter(task => 
                task.id !== taskId || 
                (new Date().getTime() - new Date(task.startTime).getTime()) < 1000 * 60 * 30 // Keep for 30 minutes
              )
            );
          }, 1000 * 60 * 30); // 30 minutes
          
        } catch (err: unknown) {
          const error = err as Error;
          console.error('Error in background upload:', error);
          
          // Update task status to error
          setUploadTasks(prevTasks => 
            prevTasks.map(task => 
              task.id === taskId ? { 
                ...task, 
                status: 'error', 
                error: error.message || 'Upload failed' 
              } : task
            )
          );
          
          // Show error toast
          showToast('error', `Text upload failed: ${error.message || 'Unknown error'}`);
        }
      })();
      
      // Reset form for next upload
      setTextContent('');
      setMetadata(JSON.stringify({ notebook: selectedNotebook.name }, null, 2));
      setRules('[]');
      
    } catch (err: unknown) {
      const error = err as Error;
      setError(`Error preparing upload: ${error.message}`);
      console.error('Error preparing upload:', err);
    }
  };

  // Update chat options
  const updateChatOption = <K extends keyof ChatOptions>(key: K, value: ChatOptions[K]) => {
    setChatOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Reset upload dialog
  const resetUploadDialog = () => {
    setUploadType('file');
    setFileToUpload(null);
    setTextContent('');
    
    // Set default metadata with notebook
    if (selectedNotebook) {
      setMetadata(JSON.stringify({ notebook: selectedNotebook.name }, null, 2));
    } else {
      setMetadata('{"notebook": ""}');
    }
    setRules('[]');
    setUseColpali(true);
  };

  // Render content based on content type
  const renderContent = (content: string, contentType: string) => {
    if (contentType.startsWith('image/') || content.startsWith('data:image/')) {
      return (
        <div className="flex justify-center p-4 bg-gray-100 rounded-md">
          { /* eslint-disable-next-line @next/next/no-img-element */ }
          <img 
            src={content} 
            alt="Document content" 
            className="max-w-full max-h-96 object-contain"
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

  // Handle going back to notebooks list
  const handleBackToNotebooks = () => {
    setSelectedNotebook(null);
    setActiveTab('list');
  };

  return (
    <div className="space-y-6">
      {/* Toast notifications */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`p-4 rounded-md shadow-md max-w-md transform transition-all duration-300 ease-in-out animate-[slide-in_0.3s_ease-out] ${
              toast.type === 'success' ? 'bg-green-100 border-l-4 border-green-500 text-green-800' :
              toast.type === 'error' ? 'bg-red-100 border-l-4 border-red-500 text-red-800' :
              'bg-blue-100 border-l-4 border-blue-500 text-blue-800'
            }`}
          >
            <div className="flex items-start">
              <div className="flex-shrink-0 mr-2">
                {toast.type === 'success' && <div className="text-green-500">✓</div>}
                {toast.type === 'error' && <div className="text-red-500">✗</div>}
                {toast.type === 'info' && <div className="text-blue-500">ℹ</div>}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">{toast.message}</p>
              </div>
              <button
                type="button"
                className="ml-4 text-gray-400 hover:text-gray-700"
                onClick={() => setToasts(toasts.filter(t => t.id !== toast.id))}
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>
      
      {/* API Key Dialog */}
      <Dialog 
        open={showApiKeyDialog} 
        onOpenChange={(open) => setShowApiKeyDialog(open)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>API Keys</DialogTitle>
            <DialogDescription>
              Enter your API keys to use OpenAI, Claude, or Ollama models.
            </DialogDescription>
          </DialogHeader>
          
          <div className="grid gap-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="openai-api-key">OpenAI API Key</Label>
              <Input
                id="openai-api-key"
                type="password"
                placeholder="sk-..."
                value={openaiApiKey}
                onChange={(e) => setOpenaiApiKey(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="claude-api-key">Claude API Key</Label>
              <Input
                id="claude-api-key"
                type="password"
                placeholder="sk-ant-..."
                value={claudeApiKey}
                onChange={(e) => setClaudeApiKey(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="ollama-url">Ollama URL</Label>
              <Input
                id="ollama-url"
                type="text"
                placeholder="http://localhost:11434"
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
              />
              <p className="text-xs text-gray-500">
                No API key needed for Ollama. Just ensure Ollama is running locally.
                <br />
                <strong>Note:</strong> You may need to start Ollama with CORS allowed: <br />
                <code className="bg-gray-100 px-1 py-0.5 text-xs rounded">OLLAMA_ORIGINS=http://localhost:3000 ollama serve</code>
              </p>
            </div>
          </div>
          
          <DialogFooter>
            <Button onClick={saveApiKeys}>
              Save Settings
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Upload Status Dialog */}
      <Dialog
        open={showUploadStatus && uploadTasks.length > 0}
        onOpenChange={setShowUploadStatus}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Upload Progress</DialogTitle>
            <DialogDescription>
              Track the status of your document uploads
            </DialogDescription>
          </DialogHeader>
          
          <div className="py-4">
            <ScrollArea className="h-[300px]">
              <div className="space-y-4">
                {uploadTasks.map((task) => (
                  <div key={task.id} className="border rounded-md p-3">
                    <div className="flex justify-between items-center mb-2">
                      <div className="font-medium truncate max-w-[200px]">{task.name}</div>
                      <Badge
                        variant={
                          task.status === 'completed' ? 'default' :
                          task.status === 'error' ? 'destructive' :
                          task.status === 'processing' ? 'secondary' : 'outline'
                        }
                      >
                        {task.status === 'uploading' ? 'Uploading' :
                         task.status === 'processing' ? 'Processing' :
                         task.status === 'completed' ? 'Completed' :
                         task.status === 'error' ? 'Failed' : 'Pending'}
                      </Badge>
                    </div>
                    
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
                      <div
                        className={`h-2.5 rounded-full ${
                          task.status === 'completed' ? 'bg-green-600' :
                          task.status === 'error' ? 'bg-red-600' :
                          'bg-blue-600'
                        }`}
                        style={{ width: `${task.progress}%` }}
                      ></div>
                    </div>
                    
                    <div className="flex justify-between text-xs text-gray-500">
                      <div>
                        {task.progress}%
                      </div>
                      <div>
                        {new Date(task.startTime).toLocaleTimeString()}
                      </div>
                    </div>
                    
                    {task.status === 'error' && task.error && (
                      <div className="mt-2 text-red-500 text-sm">
                        Error: {task.error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowUploadStatus(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Page header with upload indicator */}
      {!selectedNotebook && (
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold flex items-center">
              <Book className="mr-2 h-6 w-6" />
              Notebooks
            </h2>
            {uploadTasks.length > 0 && (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setShowUploadStatus(true)}
                className="flex items-center"
              >
                <div className="relative mr-2">
                  <div className="h-3 w-3 bg-blue-500 rounded-full"></div>
                  {uploadTasks.some(task => task.status === 'uploading' || task.status === 'processing') && (
                    <div className="absolute inset-0 h-3 w-3 animate-ping bg-blue-400 rounded-full"></div>
                  )}
                </div>
                {uploadTasks.filter(t => t.status === 'uploading' || t.status === 'processing').length} Uploading
              </Button>
            )}
          </div>
          <p className="text-gray-600">
            Create notebooks to organize your documents and chat with AI about specific topics.
          </p>
        </div>
      )}

      {/* Error message */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Notebook Gallery */}
      {!selectedNotebook ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">My Notebooks</h3>
          </div>
          
          {loading && !notebooks.length ? (
            <div className="flex justify-center items-center p-8">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {/* Existing notebooks */}
              {notebooks.map((notebook) => (
                <Card
                  key={notebook.id}
                  className="cursor-pointer hover:shadow-lg transition-all duration-300 hover:-translate-y-1 hover:border-blue-300"
                  onClick={() => {
                    handleNotebookSelect(notebook);
                    // Directly navigate to chat
                    setActiveTab('chat');
                  }}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2">
                      <Book className="h-4 w-4 text-blue-500" />
                      <span className="truncate">{notebook.name}</span>
                    </CardTitle>
                    <CardDescription className="line-clamp-2 h-10">
                      {notebook.description || 'No description'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center">
                      <Badge variant="outline" className="text-xs">
                        {new Date(notebook.created_at).toLocaleDateString()}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {notebooks.find(nb => nb.id === notebook.id) && 
                         documents.filter(doc => 
                           doc.metadata && 
                           typeof doc.metadata === 'object' && 
                           'notebook' in doc.metadata && 
                           doc.metadata.notebook === notebook.name
                         ).length > 0 ? 
                          `${documents.filter(doc => 
                             doc.metadata && 
                             typeof doc.metadata === 'object' && 
                             'notebook' in doc.metadata && 
                             doc.metadata.notebook === notebook.name
                           ).length} docs` : 
                          ''}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
              
              {/* Create new notebook card */}
              <Card 
                className="cursor-pointer border-dashed border-2 border-gray-300 bg-gray-50 hover:bg-gray-100 hover:border-blue-300 hover:-translate-y-1 transition-all duration-300 flex flex-col items-center justify-center"
                onClick={() => setShowCreateDialog(true)}
              >
                <div className="py-8 flex flex-col items-center">
                  <div className="h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center mb-3 group-hover:bg-blue-200">
                    <Plus className="h-6 w-6 text-blue-500" />
                  </div>
                  <p className="font-medium">Create New Notebook</p>
                  <p className="text-sm text-gray-500 mt-1">Add a new collection</p>
                </div>
              </Card>
            </div>
          )}
        </div>
      ) : null}
      
      {/* Removed Import Dialog - Using automatic persistence now */}
      
      {/* Create Notebook Dialog */}
      <Dialog 
        open={showCreateDialog} 
        onOpenChange={setShowCreateDialog}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              Create New Notebook
            </DialogTitle>
            <DialogDescription>
              Create a notebook to organize your documents and chat with them.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="dialog-notebook-name">Notebook Name</Label>
              <Input
                id="dialog-notebook-name"
                placeholder="Enter a name for your notebook"
                value={notebookName}
                onChange={(e) => setNotebookName(e.target.value)}
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="dialog-notebook-description">Description (Optional)</Label>
              <Textarea
                id="dialog-notebook-description"
                placeholder="Enter a description for your notebook"
                value={notebookDescription}
                onChange={(e) => setNotebookDescription(e.target.value)}
                rows={3}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={() => {
                handleCreateNotebook();
                setShowCreateDialog(false);
              }} 
              disabled={!notebookName || loading}
            >
              {loading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              ) : null}
              Create Notebook
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

        {/* Document Management View */}
        {selectedNotebook && activeTab === 'documents' && (
          <div className="space-y-4">
            {/* Notebook header with actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleBackToNotebooks}
                >
                  <ChevronLeft className="h-4 w-4 mr-1" />
                  Back
                </Button>
                <h3 className="text-lg font-medium flex items-center">
                  <FileText className="mr-2 h-5 w-5 text-blue-500" /> 
                  Documents in {selectedNotebook.name}
                </h3>
              </div>
              
              <div className="flex items-center space-x-2">
                <Dialog 
                  open={showUploadDialog} 
                  onOpenChange={(open) => {
                    setShowUploadDialog(open);
                    if (!open) resetUploadDialog();
                  }}
                >
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="h-4 w-4 mr-1" />
                      Add Document
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                      <DialogTitle>Add to {selectedNotebook.name}</DialogTitle>
                      <DialogDescription>
                        Upload a file or text to your notebook.
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
                          rows={3}
                        />
                        <p className="text-xs text-gray-500 mt-1">
                          The notebook name is automatically included in metadata.
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="useColpali"
                          checked={useColpali}
                          onCheckedChange={setUseColpali}
                        />
                        <Label htmlFor="useColpali">Use Colpali</Label>
                      </div>
                    </div>
                    
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setShowUploadDialog(false)}>
                        Cancel
                      </Button>
                      <Button 
                        onClick={uploadType === 'file' ? handleFileUpload : handleTextUpload}
                        disabled={loading}
                      >
                        {loading ? 'Uploading...' : 'Upload'}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                
                <Button 
                  variant="outline"
                  size="sm"
                  onClick={() => setActiveTab('chat')}
                >
                  <MessageSquare className="h-4 w-4 mr-1" />
                  Back to Chat
                </Button>
              </div>
            </div>
            
            {/* Document tabs */}
            <Tabs defaultValue="notebook-docs">
              <TabsList>
                <TabsTrigger value="notebook-docs">Notebook Documents</TabsTrigger>
                <TabsTrigger value="add-existing">Add Existing Documents</TabsTrigger>
              </TabsList>
              
              <TabsContent value="notebook-docs" className="mt-4">
                {loading ? (
                  <div className="flex justify-center items-center p-8">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
                  </div>
                ) : notebookDocuments.length === 0 ? (
                  <div className="text-center p-8 border-2 border-dashed rounded-lg">
                    <FileText className="mx-auto h-12 w-12 mb-3 text-gray-400" />
                    <p className="text-gray-500 mb-3">No documents in this notebook yet.</p>
                    <div className="flex justify-center gap-3">
                      <Button onClick={() => setShowUploadDialog(true)}>
                        <Plus className="h-4 w-4 mr-1" />
                        Upload Document
                      </Button>
                      <Button variant="outline" onClick={() => (document.querySelector('[data-value="add-existing"]') as HTMLElement)?.click()}>
                        Add Existing Documents
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Input
                      placeholder="Filter documents..."
                      className="max-w-md"
                      onChange={(e) => setDocumentFilter(e.target.value)}
                    />
                    
                    <div className="border rounded-md">
                      <div className="grid grid-cols-12 bg-gray-100 p-3 font-medium border-b">
                        <div className="col-span-5">Filename</div>
                        <div className="col-span-3">Type</div>
                        <div className="col-span-3">ID</div>
                        <div className="col-span-1">Actions</div>
                      </div>
                      <div className="divide-y">
                        {notebookDocuments
                          .filter(doc => 
                            !documentFilter || 
                            (doc.filename && doc.filename.toLowerCase().includes(documentFilter.toLowerCase()))
                          )
                          .map((doc) => (
                            <div 
                              key={doc.external_id}
                              className="grid grid-cols-12 p-3 items-center"
                            >
                              <div className="col-span-5 truncate">
                                {doc.filename || 'N/A'}
                              </div>
                              <div className="col-span-3">
                                <Badge variant="secondary">
                                  {doc.content_type.split('/')[0]}
                                </Badge>
                              </div>
                              <div className="col-span-3 font-mono text-xs">
                                {doc.external_id.substring(0, 8)}...
                              </div>
                              <div className="col-span-1 flex justify-center">
                                <Button 
                                  variant="ghost" 
                                  size="sm"
                                  onClick={() => handleRemoveDocument(doc.external_id)}
                                  title="Remove from notebook"
                                >
                                  <Trash className="h-4 w-4 text-red-500" />
                                </Button>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="add-existing" className="mt-4">
                <div className="space-y-4">
                  <div className="flex gap-2 items-center">
                    <Input
                      placeholder="Filter documents by name..."
                      value={documentFilter}
                      onChange={(e) => setDocumentFilter(e.target.value)}
                      className="max-w-md"
                    />
                    <Button 
                      onClick={handleAddDocuments}
                      disabled={documentsToAdd.length === 0 || loading}
                      size="sm"
                    >
                      {loading ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      ) : (
                        <Plus className="h-4 w-4 mr-1" />
                      )}
                      Add Selected ({documentsToAdd.length})
                    </Button>
                  </div>
                  
                  {documents
                    .filter(doc => !notebookDocuments.some(nbDoc => nbDoc.external_id === doc.external_id))
                    .filter(doc => 
                      !documentFilter || 
                      (doc.filename && doc.filename.toLowerCase().includes(documentFilter.toLowerCase()))
                    ).length === 0 ? (
                    <div className="text-center p-8 border-2 border-dashed rounded-lg">
                      <p className="text-gray-500">No additional documents available to add.</p>
                    </div>
                  ) : (
                    <div className="border rounded-md">
                      <div className="grid grid-cols-12 bg-gray-100 p-3 font-medium border-b">
                        <div className="col-span-1"></div>
                        <div className="col-span-7">Filename</div>
                        <div className="col-span-2">Type</div>
                        <div className="col-span-2">ID</div>
                      </div>
                      <div className="divide-y">
                        {documents
                          .filter(doc => !notebookDocuments.some(nbDoc => nbDoc.external_id === doc.external_id))
                          .filter(doc => 
                            !documentFilter || 
                            (doc.filename && doc.filename.toLowerCase().includes(documentFilter.toLowerCase()))
                          )
                          .map(doc => (
                            <div 
                              key={doc.external_id} 
                              className="grid grid-cols-12 p-3 items-center hover:bg-gray-50"
                            >
                              <div className="col-span-1 flex justify-center">
                                <input
                                  type="checkbox"
                                  id={`doc-${doc.external_id}`}
                                  checked={documentsToAdd.includes(doc.external_id)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setDocumentsToAdd([...documentsToAdd, doc.external_id]);
                                    } else {
                                      setDocumentsToAdd(documentsToAdd.filter(id => id !== doc.external_id));
                                    }
                                  }}
                                  className="h-4 w-4"
                                />
                              </div>
                              <label htmlFor={`doc-${doc.external_id}`} className="col-span-7 cursor-pointer truncate">
                                {doc.filename || doc.external_id.substring(0, 8) + '...'}
                              </label>
                              <div className="col-span-2">
                                <Badge variant="outline">{doc.content_type.split('/')[0]}</Badge>
                              </div>
                              <div className="col-span-2 font-mono text-xs">
                                {doc.external_id.substring(0, 8)}...
                              </div>
                            </div>
                          ))
                        }
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>
            
            <div className="flex justify-end mt-6">
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" className="border-red-500 text-red-500 hover:bg-red-50">
                    <Trash className="mr-2 h-4 w-4" />
                    Delete Notebook
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Delete Notebook</DialogTitle>
                    <DialogDescription>
                      Are you sure you want to delete this notebook? This will not delete the documents themselves.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="py-3">
                    <p className="font-medium">Notebook: {selectedNotebook.name}</p>
                    <p className="text-sm text-gray-500 mt-1">
                      This notebook contains {notebookDocuments.length} document{notebookDocuments.length !== 1 ? 's' : ''}.
                    </p>
                  </div>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => (document.querySelector('[data-state="open"] button[data-state="closed"]') as HTMLElement)?.click()}>Cancel</Button>
                    <Button 
                      variant="outline" 
                      className="border-red-500 text-red-500 hover:bg-red-50"
                      onClick={handleDeleteNotebook}
                      disabled={loading}
                    >
                      {loading ? 'Deleting...' : 'Delete Notebook'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        )}

        {/* Chat with Notebook */}
        {selectedNotebook && activeTab === 'chat' && (
          <div className="space-y-4">
            {/* Notebook header with actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => { 
                    setSelectedNotebook(null);
                    setActiveTab('list');
                  }}
                >
                  <ChevronLeft className="h-4 w-4 mr-1" />
                  Back
                </Button>
                <h3 className="text-lg font-medium flex items-center">
                  <Book className="mr-2 h-5 w-5 text-blue-500" /> 
                  {selectedNotebook.name}
                </h3>
              </div>
              
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setActiveTab('documents')}
                >
                  <FileText className="h-4 w-4 mr-1" />
                  Manage Documents
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowApiKeyDialog(true)}
                >
                  <Settings className="h-4 w-4 mr-1" />
                  API Settings
                </Button>
              </div>
            </div>
            
            {/* Chat interface */}
            <div className="h-[calc(100vh-12rem)] flex flex-col bg-white rounded-lg border shadow-sm">
              <div className="flex-grow flex overflow-hidden">
                {/* Chat window - larger now */}
                <div className="flex-grow flex flex-col overflow-hidden p-4">
                  {/* Chat messages */}
                  <ScrollArea className="flex-grow mb-4">
                    {chatMessages.length > 0 ? (
                      <div className="space-y-6">
                        {chatMessages.map((message, index) => (
                          <div key={index} className="space-y-2">
                            <div 
                              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                              <div 
                                className={`max-w-[80%] p-4 rounded-lg ${
                                  message.role === 'user' 
                                    ? 'bg-primary text-primary-foreground' 
                                    : 'bg-gray-100'
                                }`}
                              >
                                <div className="whitespace-pre-wrap">{message.content}</div>
                              </div>
                            </div>
                            
                            {/* Source citations - show after AI responses */}
                            {message.role === 'assistant' && index === chatMessages.length - 1 && chatResults.length > 0 && (
                              <div className="ml-2 mt-3">
                                <details className="group">
                                  <summary className="text-sm text-gray-500 mb-2 cursor-pointer hover:text-gray-700 flex items-center justify-between list-none">
                                    <div className="flex items-center gap-1">
                                      <Info className="h-3.5 w-3.5" />
                                      <span>Sources ({chatResults.length})</span>
                                    </div>
                                    <ChevronDown className="h-3.5 w-3.5 ml-1 transform group-open:rotate-180 transition-transform" />
                                  </summary>
                                  <div className="flex flex-wrap gap-2 mt-2 pl-1">
                                    {chatResults.map((result, idx) => (
                                      <Button
                                        key={`source-${idx}`}
                                        variant="outline"
                                        size="sm"
                                        className="flex items-center gap-1 text-xs hover:bg-blue-50 hover:border-blue-300 transition-colors"
                                        onClick={() => {
                                          setSelectedSource(result);
                                          setShowSourceDetail(true);
                                        }}
                                      >
                                        <FileText className="h-3 w-3" />
                                        {result.filename || `Doc ${result.document_id.substring(0, 6)}...`}
                                        <Badge variant="secondary" className="ml-1 px-1 py-0 text-[10px]">
                                          {result.score.toFixed(2)}
                                        </Badge>
                                      </Button>
                                    ))}
                                  </div>
                                </details>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="h-full flex items-center justify-center">
                        <div className="text-center p-8 max-w-md">
                          <MessageSquare className="mx-auto h-12 w-12 mb-4 text-blue-500 opacity-80" />
                          <h3 className="text-xl font-medium mb-2">Chat with {selectedNotebook.name}</h3>
                          <p className="text-gray-500 mb-6">
                            Ask questions about the documents in this notebook to get AI-powered answers based on their content.
                          </p>
                          <div className="text-sm text-gray-600 bg-gray-50 p-4 rounded-lg">
                            <p className="font-medium mb-2">Try asking:</p>
                            <ul className="list-disc list-inside space-y-1">
                              <li>Summarize the main points in these documents</li>
                              <li>What are the key findings?</li>
                              <li>Compare the information across documents</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    )}
                  </ScrollArea>
                  
                  {/* Input area */}
                  <div className="pt-4 border-t">
                    <div className="flex gap-2">
                      <Textarea 
                        placeholder="Ask a question about the notebook documents..." 
                        value={chatQuery}
                        onChange={(e) => setChatQuery(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleChatWithNotebook();
                          }
                        }}
                        className="min-h-10 resize-none"
                      />
                      <Button onClick={handleChatWithNotebook} disabled={loading} className="px-6">
                        {loading ? (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        ) : (
                          <MessageSquare className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                    
                    <div className="flex justify-between items-center mt-2">
                      <p className="text-xs text-gray-500">
                        Press Enter to send, Shift+Enter for a new line
                      </p>
                      
                      <button
                        type="button" 
                        className="flex items-center text-xs text-gray-600 hover:text-gray-900"
                        onClick={() => setShowChatAdvanced(!showChatAdvanced)}
                      >
                        <Settings className="mr-1 h-3 w-3" />
                        {showChatAdvanced ? 'Hide Options' : 'Show Options'}
                        {showChatAdvanced ? <ChevronUp className="ml-1 h-3 w-3" /> : <ChevronDown className="ml-1 h-3 w-3" />}
                      </button>
                    </div>
                    
                    {/* Collapsible advanced options */}
                    {showChatAdvanced && (
                      <div className="mt-3 p-3 border rounded-md bg-gray-50 text-sm">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <Label htmlFor="model-provider" className="text-xs block mb-1">
                              Provider
                            </Label>
                            <div className="flex gap-1 flex-wrap">
                              <Button
                                type="button"
                                size="sm"
                                variant={chatOptions.model_provider === 'openai' ? 'default' : 'outline'}
                                onClick={() => {
                                  updateChatOption('model_provider', 'openai');
                                  updateChatOption('model', 'gpt-4o');
                                }}
                                className="flex-1 h-8 text-xs"
                              >
                                OpenAI
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                variant={chatOptions.model_provider === 'claude' ? 'default' : 'outline'}
                                onClick={() => {
                                  updateChatOption('model_provider', 'claude');
                                  updateChatOption('model', 'claude-3-7-sonnet-latest');
                                }}
                                className="flex-1 h-8 text-xs"
                              >
                                Claude
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                variant={chatOptions.model_provider === 'ollama' ? 'default' : 'outline'}
                                onClick={() => {
                                  updateChatOption('model_provider', 'ollama');
                                  updateChatOption('model', 'llama3');
                                }}
                                className="flex-1 h-8 text-xs"
                              >
                                Ollama
                              </Button>
                            </div>
                          </div>
                          
                          <div>
                            <Label htmlFor="model" className="text-xs block mb-1">
                              Model
                            </Label>
                            {chatOptions.model_provider === 'ollama' ? (
                              // Text input for Ollama models
                              <Input
                                id="model"
                                className="w-full text-xs h-8"
                                value={chatOptions.model}
                                onChange={(e) => updateChatOption('model', e.target.value)}
                                placeholder="llama3, llama2, mistral, gemma, etc."
                              />
                            ) : (
                              <div className="space-y-2">
                                <select
                                  id="model"
                                  className="w-full p-1 text-xs border rounded-md h-8"
                                  value={chatOptions.model}
                                  onChange={(e) => updateChatOption('model', e.target.value)}
                                >
                                  {chatOptions.model_provider === 'openai' ? (
                                    openaiModels.map(model => (
                                      <option key={model.id} value={model.id}>
                                        {model.name}
                                      </option>
                                    ))
                                  ) : (
                                    claudeModels.map(model => (
                                      <option key={model.id} value={model.id}>
                                        {model.name}
                                      </option>
                                    ))
                                  )}
                                </select>
                                
                                {/* Custom model input field shown when "other" is selected */}
                                {chatOptions.model === 'other' && (
                                  <Input
                                    className="w-full text-xs h-8 mt-1"
                                    placeholder="Enter custom model name"
                                    onChange={(e) => {
                                      if (e.target.value) {
                                        // Update the model directly to the custom value
                                        updateChatOption('model', e.target.value);
                                      } else {
                                        // Keep it as "other" if empty (to keep the input box visible)
                                        updateChatOption('model', 'other');
                                      }
                                    }}
                                  />
                                )}
                              </div>
                            )}
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <Label htmlFor="chat-reranking" className="text-xs">Use Reranking</Label>
                            <Switch 
                              id="chat-reranking"
                              checked={chatOptions.use_reranking}
                              onCheckedChange={(checked) => updateChatOption('use_reranking', checked)}
                              className="scale-75"
                            />
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <Label htmlFor="chat-colpali" className="text-xs">Use Colpali</Label>
                            <Switch 
                              id="chat-colpali"
                              checked={chatOptions.use_colpali}
                              onCheckedChange={(checked) => updateChatOption('use_colpali', checked)}
                              className="scale-75"
                            />
                          </div>
                          
                          <div>
                            <Label htmlFor="chat-temperature" className="text-xs block mb-1">
                              Temperature: {chatOptions.temperature.toFixed(2)}
                            </Label>
                            <Input 
                              id="chat-temperature" 
                              type="range"
                              min={0} 
                              max={1} 
                              step={0.01}
                              value={chatOptions.temperature}
                              onChange={(e) => updateChatOption('temperature', parseFloat(e.target.value) || 0)}
                              className="h-6"
                            />
                          </div>
                          
                          <div>
                            <Label htmlFor="chat-k" className="text-xs block mb-1">
                              Results (k): {chatOptions.k}
                            </Label>
                            <Input 
                              id="chat-k" 
                              type="range"
                              min={1} 
                              max={12}
                              value={chatOptions.k}
                              onChange={(e) => updateChatOption('k', parseInt(e.target.value) || 1)}
                              className="h-6"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Source detail dialog */}
            <Dialog 
              open={showSourceDetail} 
              onOpenChange={setShowSourceDetail}
            >
              <DialogContent className="sm:max-w-lg">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Source Document
                  </DialogTitle>
                  <DialogDescription>
                    {selectedSource?.filename || `Document ID: ${selectedSource?.document_id?.substring(0, 10)}...`}
                  </DialogDescription>
                </DialogHeader>
                
                <div className="space-y-4">
                  {selectedSource && (
                    <>
                      <div className="bg-gray-50 p-3 rounded-md max-h-[400px] overflow-auto">
                        {renderContent(selectedSource.content, selectedSource.content_type)}
                      </div>
                      
                      <div className="text-sm">
                        <p className="font-medium mb-1">Metadata:</p>
                        <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
                          {JSON.stringify(selectedSource.metadata, null, 2)}
                        </pre>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2 mt-2">
                        <div className="bg-gray-50 p-2 rounded border flex flex-col items-center">
                          <span className="text-xs text-gray-500">Relevance Score</span>
                          <span className="font-medium text-sm">{selectedSource.score.toFixed(3)}</span>
                        </div>
                        <div className="bg-gray-50 p-2 rounded border flex flex-col items-center">
                          <span className="text-xs text-gray-500">Chunk #</span>
                          <span className="font-medium text-sm">{selectedSource.chunk_number}</span>
                        </div>
                        <div className="bg-gray-50 p-2 rounded border flex flex-col items-center overflow-hidden">
                          <span className="text-xs text-gray-500">Document ID</span>
                          <span className="font-mono text-xs truncate w-full text-center">{selectedSource.document_id.substring(0, 10)}...</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>
                
                <DialogFooter>
                  <Button variant="outline" onClick={() => setShowSourceDetail(false)}>
                    Close
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        )}
    </div>
  );
};

export default NotebookSection;