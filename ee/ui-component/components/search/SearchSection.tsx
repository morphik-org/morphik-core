"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Search } from 'lucide-react';
import { showAlert } from '@/components/ui/alert-system';
import SearchOptionsDialog from './SearchOptionsDialog';
import SearchResultCard from './SearchResultCard';

import { SearchResult, SearchOptions, Folder } from '@/components/types';

interface SearchSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
}

const SearchSection: React.FC<SearchSectionProps> = ({ apiBaseUrl, authToken }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showSearchAdvanced, setShowSearchAdvanced] = useState(false);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>({
    filters: '{}',
    k: 4,
    min_score: 0,
    use_reranking: false,
    use_colpali: true
  });

  // Update search options
  const updateSearchOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    setSearchOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Fetch folders and reset search results when auth token or API URL changes
  useEffect(() => {
    console.log('SearchSection: Token or API URL changed, resetting results');
    setSearchResults([]);

    // Fetch available folders
    const fetchFolders = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/folders`, {
          headers: authToken ? { 'Authorization': `Bearer ${authToken}` } : {}
        });

        if (response.ok) {
          const folderData = await response.json();
          setFolders(folderData);
        } else {
          console.error('Failed to fetch folders', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching folders:', error);
      }
    };

    if (authToken || apiBaseUrl.includes('localhost')) {
      fetchFolders();
    }
  }, [authToken, apiBaseUrl]);

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

      const response = await fetch(`${apiBaseUrl}/retrieve/chunks`, {
        method: 'POST',
        headers: {
          'Authorization': authToken ? `Bearer ${authToken}` : '',
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

  return (
    <div className="flex-1 flex flex-col h-full p-4">
      <div className="flex-1 flex flex-col">
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
            <SearchOptionsDialog
              showSearchAdvanced={showSearchAdvanced}
              setShowSearchAdvanced={setShowSearchAdvanced}
              searchOptions={searchOptions}
              updateSearchOption={updateSearchOption}
              folders={folders}
            />
          </div>
        </div>

        <div className="mt-6 flex-1 overflow-hidden">
          {searchResults.length > 0 ? (
            <div>
              <h3 className="text-lg font-medium mb-4">Results ({searchResults.length})</h3>

              <ScrollArea className="h-[calc(100vh-320px)]">
                <div className="space-y-6 pr-4">
                  {searchResults.map((result) => (
                    <SearchResultCard key={`${result.document_id}-${result.chunk_number}`} result={result} />
                  ))}
                </div>
              </ScrollArea>
            </div>
          ) : (
            <div className="text-center py-16 border border-dashed rounded-lg">
              <Search className="mx-auto h-12 w-12 mb-2 text-muted-foreground" />
              <p className="text-muted-foreground">
                {searchQuery.trim() ? 'No results found. Try a different query.' : 'Enter a query to search your documents.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchSection;
