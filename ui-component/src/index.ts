'use client';

import MorphikUI from '../components/MorphikUI';
import { extractTokenFromUri, getApiBaseUrlFromUri } from '../lib/utils';
import { showAlert, showUploadAlert, removeAlert } from '../components/ui/alert-system';

export { 
  MorphikUI, 
  extractTokenFromUri, 
  getApiBaseUrlFromUri,
  // Alert system helpers
  showAlert,
  showUploadAlert,
  removeAlert
};

// Export types
export type { MorphikUIProps, Document, SearchResult, ChatMessage, SearchOptions, QueryOptions } from '../components/types';
