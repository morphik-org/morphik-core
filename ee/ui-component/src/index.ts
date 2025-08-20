"use client";

import MorphikUI from "../components/MorphikUI";
// Explicitly named sidebar variants for clarity
export { MorphikSidebarLocal } from "../components/sidebar";
export { MorphikSidebarRemote } from "../components/sidebar-stateful";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "../lib/utils";
import { showAlert, showUploadAlert, removeAlert } from "../components/ui/alert-system";

export {
  MorphikUI,
  extractTokenFromUri,
  getApiBaseUrlFromUri,
  // Alert system helpers
  showAlert,
  showUploadAlert,
  removeAlert,
};

// Export types
export type {
  MorphikUIProps,
  Breadcrumb,
  Document,
  SearchResult,
  ChatMessage,
  SearchOptions,
  QueryOptions,
} from "../components/types";
