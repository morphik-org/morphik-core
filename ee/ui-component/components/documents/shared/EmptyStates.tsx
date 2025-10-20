"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Filter } from "lucide-react";

interface EmptyDocumentsProps {
  onClearFilters?: () => void;
}

export const EmptyDocuments: React.FC<EmptyDocumentsProps> = () => (
  <div className="flex flex-col items-center justify-center p-12 text-center">
    <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-muted-foreground"
      >
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="9" y1="15" x2="15" y2="15"></line>
      </svg>
    </div>
    <p className="text-muted-foreground">No documents found in this view.</p>
    <p className="mt-1 text-xs text-muted-foreground">Try uploading a document or selecting a different folder.</p>
  </div>
);

interface NoMatchingDocumentsProps {
  searchQuery: string;
  hasFilters: boolean;
  onClearFilters: () => void;
}

export const NoMatchingDocuments: React.FC<NoMatchingDocumentsProps> = ({
  searchQuery,
  hasFilters,
  onClearFilters,
}) => (
  <div className="flex flex-col items-center justify-center p-12 text-center">
    <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
      <Filter className="text-muted-foreground" />
    </div>
    <p className="text-muted-foreground">
      No documents match {searchQuery.trim() && "your search"}
      {searchQuery.trim() && hasFilters && " and"}
      {hasFilters && " the current filters"}.
    </p>
    <Button variant="link" className="mt-2" onClick={onClearFilters}>
      Clear {searchQuery.trim() && "search"}
      {searchQuery.trim() && hasFilters && " and"}
      {hasFilters && " filters"}
    </Button>
  </div>
);

interface LoadingDocumentsProps {
  message?: string;
}

export const LoadingDocuments: React.FC<LoadingDocumentsProps> = ({ message = "Loading documents..." }) => (
  <div className="p-8">
    <div className="flex flex-col items-center justify-center">
      <div className="mb-4 h-8 w-8 animate-spin rounded-full border-b-2 border-primary"></div>
      <p className="text-muted-foreground">{message}</p>
    </div>
  </div>
);
