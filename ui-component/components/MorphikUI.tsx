"use client";

import React, { useState, useEffect } from 'react';
import { Sidebar } from '@/components/ui/sidebar';
import DocumentsSection from '@/components/documents/DocumentsSection';
import SearchSection from '@/components/search/SearchSection';
import ChatSection from '@/components/chat/ChatSection';
import GraphSection from '@/components/GraphSection';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AlertSystem } from '@/components/ui/alert-system';
import { extractTokenFromUri, getApiBaseUrlFromUri } from '@/lib/utils';
import { MorphikUIProps } from '@/components/types';
import { ArrowLeft } from 'lucide-react';

// Default API base URL
const DEFAULT_API_BASE_URL = 'http://localhost:8000';

const MorphikUI: React.FC<MorphikUIProps> = ({ 
  connectionUri,
  apiBaseUrl = DEFAULT_API_BASE_URL,
  isReadOnlyUri = false, // Default to editable URI
  onUriChange,
  onBackClick,
  initialFolder = null,
  initialSection = 'documents'
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
  const [activeSection, setActiveSection] = useState(initialSection);
  const [selectedGraphName, setSelectedGraphName] = useState<string | undefined>(undefined);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  
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
    <>
      <div className="flex h-screen">
        <Sidebar 
          activeSection={activeSection} 
          onSectionChange={setActiveSection}
          className="h-screen"
          connectionUri={currentUri}
          isReadOnlyUri={isReadOnlyUri}
          onUriChange={handleUriChange}
          isCollapsed={isSidebarCollapsed}
          setIsCollapsed={setIsSidebarCollapsed}
        />
        
        <div className="flex-1 flex flex-col h-screen overflow-hidden">
          {/* Header with back button */}
          {onBackClick && (
            <div className="bg-background border-b p-3 flex items-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={onBackClick}
                className="mr-2"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back to dashboard
              </Button>
            </div>
          )}
          
          <div className="flex-1 p-6 flex flex-col overflow-hidden">
            {/* Documents Section */}
            {activeSection === 'documents' && (
              <DocumentsSection 
                apiBaseUrl={effectiveApiBaseUrl} 
                authToken={authToken}
                initialFolder={initialFolder}
                setSidebarCollapsed={setIsSidebarCollapsed}
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

            {/* Notebooks Section - Removed */}

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
      </div>
      
      {/* Global alert system - integrated directly in the component */}
      <AlertSystem position="bottom-right" />
    </>
  );
};

export default MorphikUI;