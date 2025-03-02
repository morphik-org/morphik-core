"use client";

import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { AlertCircle, CheckCircle, FileText, Image, Upload, PlusCircle, Search, MessageSquare, Info, Settings, ChevronDown, ChevronUp } from 'lucide-react';

// API base URL - change this to match your DataBridge server
const API_BASE_URL = 'http://localhost:8000';

interface Document {
  external_id: string;
  filename?: string;
  content_type: string;
  metadata: Record<string, any>;
  system_metadata: Record<string, any>;
  additional_metadata: Record<string, any>;
}

interface SearchResult {
  document_id: string;
  chunk_number: number;
  content: string;
  content_type: string;
  score: number;
  filename?: string;
  metadata: Record<string, any>;
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
  use_colpali: boolean;
}

interface QueryOptions extends SearchOptions {
  max_tokens: number;
  temperature: number;
}

const DataBridgeUI = () => {
  const [activeTab, setActiveTab] = useState('documents');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [chatQuery, setChatQuery] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [uploadType, setUploadType] = useState<'file' | 'text'>('file');
  const [textContent, setTextContent] = useState('');
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  const [metadata, setMetadata] = useState('{}');
  const [rules, setRules] = useState('[]');
  const [useColpali, setUseColpali] = useState(false);
  
  // Advanced options for search
  const [showSearchAdvanced, setShowSearchAdvanced] = useState(false);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: true,
    use_colpali: false
  });

  // Advanced options for chat/query
  const [showChatAdvanced, setShowChatAdvanced] = useState(false);
  const [queryOptions, setQueryOptions] = useState<QueryOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: true,
    use_colpali: false,
    max_tokens: 500,
    temperature: 0.7
  });

  // Auth token - in a real application, you would get this from your auth system
  const authToken = 'YOUR_AUTH_TOKEN';

  // Headers for API requests
  const headers = {
    'Authorization': authToken
  };

  // Fetch documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Fetch all documents
  const fetchDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/documents`, {
        headers
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.statusText}`);
      }
      
      const data = await response.json();
      setDocuments(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Fetch a specific document by ID
  const fetchDocument = async (documentId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
        headers
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch document: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSelectedDocument(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Handle document click
  const handleDocumentClick = (document: Document) => {
    fetchDocument(document.external_id);
  };

  // Handle file upload
  const handleFileUpload = async () => {
    if (!fileToUpload) {
      setError('Please select a file to upload');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('metadata', metadata);
      formData.append('rules', rules);
      
      const url = `${API_BASE_URL}/ingest/file${useColpali ? '?use_colpali=true' : ''}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': authToken
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Failed to upload file: ${response.statusText}`);
      }
      
      const data = await response.json();
      fetchDocuments(); // Refresh document list
      setShowUploadDialog(false);
      setFileToUpload(null);
      setMetadata('{}');
      setRules('[]');
      setUseColpali(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Handle text upload
  const handleTextUpload = async () => {
    if (!textContent.trim()) {
      setError('Please enter text content');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/ingest/text`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: textContent,
          metadata: JSON.parse(metadata || '{}'),
          rules: JSON.parse(rules || '[]'),
          use_colpali: useColpali
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to upload text: ${response.statusText}`);
      }
      
      const data = await response.json();
      fetchDocuments(); // Refresh document list
      setShowUploadDialog(false);
      setTextContent('');
      setMetadata('{}');
      setRules('[]');
      setUseColpali(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Handle search
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setError('Please enter a search query');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
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
          use_colpali: searchOptions.use_colpali
        })
      });
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSearchResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Handle chat
  const handleChat = async () => {
    if (!chatQuery.trim()) {
      setError('Please enter a message');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Add user message to chat
      const userMessage: ChatMessage = { role: 'user', content: chatQuery };
      setChatMessages(prev => [...prev, userMessage]);
      
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Authorization': authToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: chatQuery,
          filters: JSON.parse(queryOptions.filters || '{}'),
          k: queryOptions.k,
          min_score: queryOptions.min_score,
          use_reranking: queryOptions.use_reranking,
          use_colpali: queryOptions.use_colpali,
          max_tokens: queryOptions.max_tokens,
          temperature: queryOptions.temperature
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
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Render content based on content type
  const renderContent = (content: string, contentType: string) => {
    if (contentType.startsWith('image/')) {
      return (
        <div className="flex justify-center p-4 bg-gray-100 rounded-md">
          <img 
            src={content} 
            alt="Document content" 
            className="max-w-full max-h-96 object-contain"
          />
        </div>
      );
    } else if (content.startsWith('data:image/png;base64,') || content.startsWith('data:image/jpeg;base64,')) {
      return (
        <div className="flex justify-center p-4 bg-gray-100 rounded-md">
          <img 
            src={content} 
            alt="Base64 image content" 
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

  // Reset upload dialog
  const resetUploadDialog = () => {
    setUploadType('file');
    setFileToUpload(null);
    setTextContent('');
    setMetadata('{}');
    setRules('[]');
    setUseColpali(false);
  };

  // Update search options
  const updateSearchOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    setSearchOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Update query options
  const updateQueryOption = <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => {
    setQueryOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">DataBridge</h1>
      
      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="search">Search</TabsTrigger>
          <TabsTrigger value="chat">Chat</TabsTrigger>
        </TabsList>
        
        {/* Documents Tab */}
        <TabsContent value="documents">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Your Documents</CardTitle>
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
                        Upload a file or text to your DataBridge repository.
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
                        onClick={uploadType === 'file' ? handleFileUpload : handleTextUpload}
                        disabled={loading}
                      >
                        {loading ? 'Uploading...' : 'Upload'}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
              <CardDescription>
                Manage your uploaded documents and view their metadata.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading && !documents.length ? (
                <div className="text-center py-8">Loading documents...</div>
              ) : documents.length > 0 ? (
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="w-full md:w-2/3">
                    <ScrollArea className="h-[500px]">
                      <div className="border rounded-md">
                        <div className="grid grid-cols-12 bg-gray-100 p-3 font-medium border-b">
                          <div className="col-span-5">Filename</div>
                          <div className="col-span-3">Type</div>
                          <div className="col-span-4">ID</div>
                        </div>
                        <div className="divide-y">
                          {documents.map((doc) => (
                            <div 
                              key={doc.external_id}
                              onClick={() => handleDocumentClick(doc)}
                              className="grid grid-cols-12 p-3 cursor-pointer hover:bg-gray-50"
                            >
                              <div className="col-span-5 flex items-center">
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
                        </div>
                      </div>
                    </ScrollArea>
                  </div>
                  
                  <div className="w-full md:w-1/3">
                    {selectedDocument ? (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Document Details</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ScrollArea className="h-[400px]">
                            <div className="space-y-4">
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
                                    <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
                                      {JSON.stringify(selectedDocument.metadata, null, 2)}
                                    </pre>
                                  </AccordionContent>
                                </AccordionItem>
                                
                                <AccordionItem value="system-metadata">
                                  <AccordionTrigger>System Metadata</AccordionTrigger>
                                  <AccordionContent>
                                    <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
                                      {JSON.stringify(selectedDocument.system_metadata, null, 2)}
                                    </pre>
                                  </AccordionContent>
                                </AccordionItem>
                                
                                <AccordionItem value="additional-metadata">
                                  <AccordionTrigger>Additional Metadata</AccordionTrigger>
                                  <AccordionContent>
                                    <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
                                      {JSON.stringify(selectedDocument.additional_metadata, null, 2)}
                                    </pre>
                                  </AccordionContent>
                                </AccordionItem>
                              </Accordion>
                            </div>
                          </ScrollArea>
                        </CardContent>
                      </Card>
                    ) : (
                      <div className="h-full flex items-center justify-center p-8 border border-dashed rounded-lg">
                        <div className="text-center text-gray-500">
                          <Info className="mx-auto h-12 w-12 mb-2" />
                          <p>Select a document to view details</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 border border-dashed rounded-lg">
                  <Upload className="mx-auto h-12 w-12 mb-2 text-gray-400" />
                  <p className="text-gray-500">No documents found. Upload your first document.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Search Tab */}
        <TabsContent value="search">
          <Card>
            <CardHeader>
              <CardTitle>Search Documents</CardTitle>
              <CardDescription>
                Search across your documents to find relevant information.
              </CardDescription>
            </CardHeader>
            <CardContent>
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
                  <button
                    type="button" 
                    className="flex items-center text-sm text-gray-600 hover:text-gray-900"
                    onClick={() => setShowSearchAdvanced(!showSearchAdvanced)}
                  >
                    <Settings className="mr-1 h-4 w-4" />
                    Advanced Options
                    {showSearchAdvanced ? <ChevronUp className="ml-1 h-4 w-4" /> : <ChevronDown className="ml-1 h-4 w-4" />}
                  </button>
                  
                  {showSearchAdvanced && (
                    <div className="mt-3 p-4 border rounded-md bg-gray-50">
                      <div className="space-y-4">
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
                            checked={searchOptions.use_colpali}
                            onCheckedChange={(checked) => updateSearchOption('use_colpali', checked)}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="mt-6">
                {searchResults.length > 0 ? (
                  <div className="space-y-6">
                    <h3 className="text-lg font-medium">Results ({searchResults.length})</h3>
                    
                    {searchResults.map((result, index) => (
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
        </TabsContent>
        
        {/* Chat Tab */}
        <TabsContent value="chat">
          <Card className="h-[700px] flex flex-col">
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
                              checked={queryOptions.use_colpali}
                              onCheckedChange={(checked) => updateQueryOption('use_colpali', checked)}
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
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DataBridgeUI;