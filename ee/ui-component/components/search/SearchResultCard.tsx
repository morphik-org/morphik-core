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
    if (contentType.startsWith("image/")) {
      return (
        <div className="flex justify-center rounded-md bg-muted p-4">
          <Image
            src={content}
            alt="Document content"
            className="max-h-96 max-w-full object-contain"
            width={500}
            height={300}
          />
        </div>
      );
    } else if (content.startsWith("data:image/png;base64,") || content.startsWith("data:image/jpeg;base64,")) {
      return (
        <div className="flex justify-center rounded-md bg-muted p-4">
          <Image
            src={content}
            alt="Base64 image content"
            className="max-h-96 max-w-full object-contain"
            width={500}
            height={300}
          />
        </div>
      );
    } else {
      return <div className="whitespace-pre-wrap rounded-md bg-muted p-4 font-mono text-sm">{content}</div>;
    }
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
