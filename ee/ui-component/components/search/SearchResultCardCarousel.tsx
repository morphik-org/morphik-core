"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, ChevronDown } from "lucide-react";

import { ChunkGroup } from "@/components/types";

interface SearchResultCardCarouselProps {
  group: ChunkGroup;
  darkMode?: boolean;
}

const SearchResultCardCarousel: React.FC<SearchResultCardCarouselProps> = ({ group, darkMode = false }) => {
  const [currentIndex, setCurrentIndex] = useState(() => {
    const allChunks = group.padding_chunks
      .filter(c => c.chunk_number < group.main_chunk.chunk_number)
      .sort((a, b) => a.chunk_number - b.chunk_number)
      .concat([group.main_chunk])
      .concat(
        group.padding_chunks
          .filter(c => c.chunk_number > group.main_chunk.chunk_number)
          .sort((a, b) => a.chunk_number - b.chunk_number)
      );

    return allChunks.findIndex(
      c => c.document_id === group.main_chunk.document_id && c.chunk_number === group.main_chunk.chunk_number
    );
  });

  const [showMetadata, setShowMetadata] = useState(false);

  const allChunks = group.padding_chunks
    .filter(c => c.chunk_number < group.main_chunk.chunk_number)
    .sort((a, b) => a.chunk_number - b.chunk_number)
    .concat([group.main_chunk])
    .concat(
      group.padding_chunks
        .filter(c => c.chunk_number > group.main_chunk.chunk_number)
        .sort((a, b) => a.chunk_number - b.chunk_number)
    );

  const currentChunk = allChunks[currentIndex];
  const isMainChunk = !currentChunk.is_padding;
  const hasMultipleChunks = allChunks.length > 1;

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

  const nextChunk = () => setCurrentIndex(prev => (prev + 1) % allChunks.length);
  const prevChunk = () => setCurrentIndex(prev => (prev - 1 + allChunks.length) % allChunks.length);

  const baseClasses = darkMode ? "rounded-lg border border-zinc-700 bg-zinc-800/50" : "rounded-lg border bg-card";

  return (
    <div className={baseClasses}>
      {/* Header */}
      <div className="p-4 pb-2">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className={`flex items-center gap-2 font-medium ${darkMode ? "text-zinc-100" : "text-foreground"}`}>
              {currentChunk.filename || `Document ${currentChunk.document_id.substring(0, 8)}...`}
              {isMainChunk && (
                <Badge variant="default" className="text-xs">
                  Match
                </Badge>
              )}
              {!isMainChunk && (
                <Badge variant="secondary" className="text-xs">
                  Context
                </Badge>
              )}
            </h4>
            <p className={`mt-1 text-xs ${darkMode ? "text-zinc-500" : "text-muted-foreground"}`}>
              Chunk {currentChunk.chunk_number} • Score: {currentChunk.score.toFixed(2)}
              {hasMultipleChunks && (
                <span className="ml-2">
                  ({currentIndex + 1} of {allChunks.length})
                </span>
              )}
            </p>
          </div>
          {hasMultipleChunks && (
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={prevChunk}
                className={`h-8 w-8 p-0 ${darkMode ? "border-zinc-700 hover:bg-zinc-700" : ""}`}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={nextChunk}
                className={`h-8 w-8 p-0 ${darkMode ? "border-zinc-700 hover:bg-zinc-700" : ""}`}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className={`mx-4 border-t ${darkMode ? "border-zinc-700" : "border-border"}`} />

      {/* Content */}
      <div className="p-4 pt-3">
        <p className={`mb-2 text-xs font-medium ${darkMode ? "text-zinc-400" : "text-muted-foreground"}`}>Content</p>
        {renderContent(currentChunk.content, currentChunk.content_type)}
      </div>

      {/* Carousel dots */}
      {hasMultipleChunks && (
        <div className="flex justify-center pb-3">
          <div className="flex gap-1">
            {allChunks.map((chunk, index) => (
              <button
                key={`${chunk.document_id}-${chunk.chunk_number}`}
                onClick={() => setCurrentIndex(index)}
                className={`h-2 w-2 rounded-full transition-colors ${
                  index === currentIndex
                    ? chunk.is_padding
                      ? darkMode
                        ? "bg-zinc-500"
                        : "bg-secondary"
                      : "bg-primary"
                    : darkMode
                      ? "bg-zinc-700"
                      : "bg-muted-foreground/30"
                }`}
                aria-label={`Go to chunk ${chunk.chunk_number}`}
              />
            ))}
          </div>
        </div>
      )}

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
              {JSON.stringify(currentChunk.metadata, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchResultCardCarousel;
