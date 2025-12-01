"use client";

import React, { useState } from "react";
import Image from "next/image";
import { ChevronDown } from "lucide-react";

import { SearchResult } from "@/components/types";

interface SearchResultCardProps {
  result: SearchResult;
  darkMode?: boolean;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({ result, darkMode = false }) => {
  const [showMetadata, setShowMetadata] = useState(false);

  // Helper to render content based on content type
  const renderContent = (content: string, contentType: string) => {
    const isImage = contentType.startsWith("image/");
    const isDataUri = content.startsWith("data:image/");

    const canUseNextImage =
      !isDataUri && (content.startsWith("/") || content.startsWith("http://") || content.startsWith("https://"));

    if (isImage || isDataUri) {
      return (
        <div className={`flex justify-center rounded-md p-4 ${darkMode ? "bg-zinc-800" : "bg-muted"}`}>
          {canUseNextImage ? (
            <Image
              src={content}
              alt="Document content"
              className="max-h-96 max-w-full object-contain"
              width={500}
              height={300}
            />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={content} alt="Document content" className="max-h-96 max-w-full object-contain" />
          )}
        </div>
      );
    }

    return (
      <div className={`whitespace-pre-wrap text-sm ${darkMode ? "text-zinc-300" : "text-foreground"}`}>{content}</div>
    );
  };

  const baseClasses = darkMode ? "rounded-lg border border-zinc-700 bg-zinc-800/50" : "rounded-lg border bg-card";

  return (
    <div className={baseClasses}>
      {/* Header */}
      <div className="p-4 pb-2">
        <h4 className={`font-medium ${darkMode ? "text-zinc-100" : "text-foreground"}`}>
          {result.filename || `Document ${result.document_id.substring(0, 8)}...`}
        </h4>
        <p className={`mt-1 text-xs ${darkMode ? "text-zinc-500" : "text-muted-foreground"}`}>
          Chunk {result.chunk_number} • Score: {result.score.toFixed(2)}
        </p>
      </div>

      {/* Divider */}
      <div className={`mx-4 border-t ${darkMode ? "border-zinc-700" : "border-border"}`} />

      {/* Content */}
      <div className="p-4 pt-3">
        <p className={`mb-2 text-xs font-medium ${darkMode ? "text-zinc-400" : "text-muted-foreground"}`}>Content</p>
        {renderContent(result.content, result.content_type)}
      </div>

      {/* Metadata */}
      <div className={`border-t ${darkMode ? "border-zinc-700" : "border-border"}`}>
        <button
          onClick={() => setShowMetadata(!showMetadata)}
          className={`flex w-full items-center justify-between px-4 py-2.5 text-sm transition-colors ${
            darkMode
              ? "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
        >
          <span>Metadata</span>
          <ChevronDown className={`h-4 w-4 transition-transform ${showMetadata ? "rotate-180" : ""}`} />
        </button>
        {showMetadata && (
          <div className={`px-4 pb-4 ${darkMode ? "bg-zinc-900/50" : "bg-muted/50"}`}>
            <pre
              className={`overflow-x-auto rounded p-2 text-xs ${darkMode ? "bg-zinc-900 text-zinc-400" : "bg-muted text-muted-foreground"}`}
            >
              {JSON.stringify(result.metadata, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchResultCard;
