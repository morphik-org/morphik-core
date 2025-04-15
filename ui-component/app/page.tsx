"use client";

import React, { useState } from 'react';
import { Sidebar } from '@/components/ui/sidebar';
import DocumentsSection from '@/components/documents/DocumentsSection';
import SearchSection from '@/components/search/SearchSection';
import ChatSection from '@/components/chat/ChatSection';
import NotebookSection from '@/components/NotebookSection';
import GraphSection from '@/components/GraphSection';
import { Badge } from '@/components/ui/badge';

// API base URL - change this to match your Morphik server
const API_BASE_URL = 'http://localhost:8000';

// Auth token - in a real application, you would get this from your auth system
const AUTH_TOKEN = 'YOUR_AUTH_TOKEN';

const MorphikUI = () => {
  const [activeSection, setActiveSection] = useState('documents');
  
  // State for selected graph name
  const [selectedGraphName, setSelectedGraphName] = useState<string | undefined>(undefined);

  return (
    <div className="flex h-screen">
      <Sidebar 
        activeSection={activeSection} 
        onSectionChange={setActiveSection}
        className="h-screen"
      />
      
      <div className="flex-1 p-6 flex flex-col h-screen overflow-hidden">
        {/* Documents Section */}
        {activeSection === 'documents' && (
          <DocumentsSection apiBaseUrl={API_BASE_URL} authToken={AUTH_TOKEN} />
        )}
        
        {/* Search Section */}
        {activeSection === 'search' && (
          <SearchSection apiBaseUrl={API_BASE_URL} authToken={AUTH_TOKEN} />
        )}
        
        {/* Chat Section */}
        {activeSection === 'chat' && (
          <ChatSection apiBaseUrl={API_BASE_URL} authToken={AUTH_TOKEN} />
        )}

        {/* Notebooks Section */}
        {activeSection === 'notebooks' && (
          <NotebookSection apiBaseUrl={API_BASE_URL} />
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
              apiBaseUrl={API_BASE_URL} 
              onSelectGraph={(graphName) => setSelectedGraphName(graphName)}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default MorphikUI;