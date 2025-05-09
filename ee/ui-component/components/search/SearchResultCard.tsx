"use client";

import React from "react";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

import { SearchResult } from "@/components/types";

interface SearchResultCardProps {
  result: SearchResult;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({ result }) => {
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

  return (
    <Card key={`${result.document_id}-${result.chunk_number}`}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-base">
              {result.filename || `Document ${result.document_id.substring(0, 8)}...`}
            </CardTitle>
            <CardDescription>
              Chunk {result.chunk_number} • Score: {result.score.toFixed(2)}
            </CardDescription>
          </div>
          <Badge variant="outline">{result.content_type}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        {renderContent(result.content, result.content_type)}

        <Accordion type="single" collapsible className="mt-4">
          <AccordionItem value="metadata">
            <AccordionTrigger className="text-sm">Metadata</AccordionTrigger>
            <AccordionContent>
              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                {JSON.stringify(result.metadata, null, 2)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
};

export default SearchResultCard;
