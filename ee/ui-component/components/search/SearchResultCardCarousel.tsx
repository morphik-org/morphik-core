"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { ChunkGroup, SearchResult } from "@/components/types";

interface SearchResultCardCarouselProps {
  group: ChunkGroup;
}

const SearchResultCardCarousel: React.FC<SearchResultCardCarouselProps> = ({ group }) => {
  const [currentIndex, setCurrentIndex] = useState(() => {
    // Start with the main chunk (find its index in all_chunks)
    const allChunks = group.padding_chunks
      .filter(c => c.chunk_number < group.main_chunk.chunk_number)
      .sort((a, b) => a.chunk_number - b.chunk_number)
      .concat([group.main_chunk])
      .concat(
        group.padding_chunks
          .filter(c => c.chunk_number > group.main_chunk.chunk_number)
          .sort((a, b) => a.chunk_number - b.chunk_number)
      );

    return allChunks.findIndex(c =>
      c.document_id === group.main_chunk.document_id &&
      c.chunk_number === group.main_chunk.chunk_number
    );
  });

  // Get all chunks in display order (padding before + main + padding after)
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

  // Helper to render content based on content type
  const renderContent = (content: string, contentType: string) => {
    const isImage = contentType.startsWith("image/");
    const isDataUri = content.startsWith("data:image/");

    // Helper: Only allow next/image for paths/URLs that Next can parse
    const canUseNextImage =
      !isDataUri && (content.startsWith("/") || content.startsWith("http://") || content.startsWith("https://"));

    if (isImage || isDataUri) {
      // Use next/image for valid remote / relative paths, fallback to <img> otherwise
      return (
        <div className="flex justify-center rounded-md bg-muted p-4">
          {canUseNextImage ? (
            <Image
              src={content}
              alt="Document content"
              className="max-h-96 max-w-full object-contain"
              width={500}
              height={300}
            />
          ) : (
            // Fallback for data-URIs or other non-standard sources
            // eslint-disable-next-line @next/next/no-img-element
            <img src={content} alt="Document content" className="max-h-96 max-w-full object-contain" />
          )}
        </div>
      );
    }

    // Default (non-image) rendering
    return <div className="whitespace-pre-wrap rounded-md bg-muted p-4 font-mono text-sm">{content}</div>;
  };

  const nextChunk = () => {
    setCurrentIndex((prev) => (prev + 1) % allChunks.length);
  };

  const prevChunk = () => {
    setCurrentIndex((prev) => (prev - 1 + allChunks.length) % allChunks.length);
  };

  return (
    <Card key={`${group.main_chunk.document_id}-${group.main_chunk.chunk_number}`}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base flex items-center gap-2">
              {currentChunk.filename || `Document ${currentChunk.document_id.substring(0, 8)}...`}
              {isMainChunk && <Badge variant="default" className="text-xs">Match</Badge>}
              {!isMainChunk && <Badge variant="secondary" className="text-xs">Context</Badge>}
            </CardTitle>
            <CardDescription>
              Chunk {currentChunk.chunk_number} • Score: {currentChunk.score.toFixed(2)}
              {hasMultipleChunks && (
                <span className="ml-2 text-xs text-muted-foreground">
                  ({currentIndex + 1} of {allChunks.length})
                </span>
              )}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{currentChunk.content_type}</Badge>
            {hasMultipleChunks && (
              <div className="flex gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={prevChunk}
                  disabled={allChunks.length <= 1}
                  className="h-8 w-8 p-0"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={nextChunk}
                  disabled={allChunks.length <= 1}
                  className="h-8 w-8 p-0"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {renderContent(currentChunk.content, currentChunk.content_type)}

        {hasMultipleChunks && (
          <div className="mt-4 flex justify-center">
            <div className="flex gap-1">
              {allChunks.map((chunk, index) => (
                <button
                  key={`${chunk.document_id}-${chunk.chunk_number}`}
                  onClick={() => setCurrentIndex(index)}
                  className={`h-2 w-2 rounded-full transition-colors ${
                    index === currentIndex
                      ? chunk.is_padding
                        ? "bg-secondary"
                        : "bg-primary"
                      : "bg-muted-foreground/30"
                  }`}
                  aria-label={`Go to chunk ${chunk.chunk_number}`}
                />
              ))}
            </div>
          </div>
        )}

        <Accordion type="single" collapsible className="mt-4">
          <AccordionItem value="metadata">
            <AccordionTrigger className="text-sm">Metadata</AccordionTrigger>
            <AccordionContent>
              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                {JSON.stringify(currentChunk.metadata, null, 2)}
              </pre>
            </AccordionContent>
          </AccordionItem>
          {hasMultipleChunks && (
            <AccordionItem value="all-chunks">
              <AccordionTrigger className="text-sm">All Chunks in Group</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2">
                  {allChunks.map((chunk, index) => (
                    <div
                      key={`${chunk.document_id}-${chunk.chunk_number}`}
                      className={`p-2 rounded border cursor-pointer transition-colors ${
                        index === currentIndex
                          ? "border-primary bg-primary/5"
                          : "border-muted hover:border-muted-foreground/50"
                      }`}
                      onClick={() => setCurrentIndex(index)}
                    >
                      <div className="flex items-center justify-between text-sm">
                        <span>Chunk {chunk.chunk_number}</span>
                        <div className="flex gap-2">
                          {!chunk.is_padding && <Badge variant="default" className="text-xs">Match</Badge>}
                          {chunk.is_padding && <Badge variant="secondary" className="text-xs">Context</Badge>}
                          <span className="text-muted-foreground">Score: {chunk.score.toFixed(2)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          )}
        </Accordion>
      </CardContent>
    </Card>
  );
};

export default SearchResultCardCarousel;
