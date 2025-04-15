"use client";

import React, { useState, useEffect } from 'react';
import { Sidebar } from '@/components/ui/sidebar';
import DocumentsSection from '@/components/documents/DocumentsSection';
import SearchSection from '@/components/search/SearchSection';
import ChatSection from '@/components/chat/ChatSection';
import NotebookSection from '@/components/NotebookSection';
import GraphSection from '@/components/GraphSection';
import { Badge } from '@/components/ui/badge';
import { extractTokenFromUri, getApiBaseUrlFromUri } from '@/lib/utils';

// Default API base URL
const DEFAULT_API_BASE_URL = 'http://localhost:8000';

interface MorphikUIProps {
  connectionUri?: string;
  apiBaseUrl?: string;
  isReadOnlyUri?: boolean; // Controls whether the URI can be edited
  onUriChange?: (uri: string) => void; // Callback when URI is changed
}

const MorphikUI: React.FC<MorphikUIProps> = ({ 
  connectionUri,
  apiBaseUrl = DEFAULT_API_BASE_URL,
  isReadOnlyUri = false, // Default to editable URI
  onUriChange
}) => {
  // State to manage connectionUri internally if needed
  const [currentUri, setCurrentUri] = useState(connectionUri);
  
  // Update internal state when prop changes
  useEffect(() => {
    setCurrentUri(connectionUri);
  }, [connectionUri]);
  
  // Handle URI changes from sidebar
  const handleUriChange = (newUri: string) => {
    console.log('MorphikUI: URI changed to:', newUri);
    setCurrentUri(newUri);
    if (onUriChange) {
      onUriChange(newUri);
    }
  };
  const [activeSection, setActiveSection] = useState('documents');
  const [selectedGraphName, setSelectedGraphName] = useState<string | undefined>(undefined);
  
  // Extract auth token and API URL from connection URI if provided
  const authToken = currentUri ? extractTokenFromUri(currentUri) : null;
  
  // Derive API base URL from the URI if provided
  // If URI is empty, this will now connect to localhost by default
  const effectiveApiBaseUrl = getApiBaseUrlFromUri(currentUri, apiBaseUrl);
  
  // Log the effective API URL for debugging
  useEffect(() => {
    console.log('MorphikUI: Using API URL:', effectiveApiBaseUrl);
    console.log('MorphikUI: Auth token present:', !!authToken);
  }, [effectiveApiBaseUrl, authToken]);
  
  return (
    <div className="flex h-screen">
      <Sidebar 
        activeSection={activeSection} 
        onSectionChange={setActiveSection}
        className="h-screen"
        connectionUri={currentUri}
        isReadOnlyUri={isReadOnlyUri}
        onUriChange={handleUriChange}
      />
      
      <div className="flex-1 p-6 flex flex-col h-screen overflow-hidden">
        {/* Documents Section */}
        {activeSection === 'documents' && (
          <DocumentsSection 
            apiBaseUrl={effectiveApiBaseUrl} 
            authToken={authToken} 
          />
        )}
        
        {/* Search Section */}
        {activeSection === 'search' && (
          <SearchSection 
            apiBaseUrl={effectiveApiBaseUrl} 
            authToken={authToken}
          />
        )}
        
        {/* Chat Section */}
        {activeSection === 'chat' && (
          <ChatSection 
            apiBaseUrl={effectiveApiBaseUrl} 
            authToken={authToken}
          />
        )}

        {/* Notebooks Section */}
        {activeSection === 'notebooks' && (
          <NotebookSection 
            apiBaseUrl={effectiveApiBaseUrl}
          />
        )}

        {/* Graphs Section */}
        {activeSection === 'graphs' && (
          <div className="space-y-4">
            <div className="flex justify-end items-center">
              {selectedGraphName && (
                <Badge variant="outline" className="bg-blue-50 px-3 py-1">
                  Current Query Graph: {selectedGraphName}
                </Badge>
              )}
            </div>
            
            <GraphSection 
              apiBaseUrl={effectiveApiBaseUrl}
              authToken={authToken}
              onSelectGraph={(graphName) => setSelectedGraphName(graphName)}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default MorphikUI;