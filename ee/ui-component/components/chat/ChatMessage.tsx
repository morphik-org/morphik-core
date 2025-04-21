"use client";

import React from 'react';
import Image from 'next/image';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Source } from '@/components/types';

// Define our own props interface to avoid empty interface error
interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ role, content, sources }) => {
  // Helper to render content based on content type
  const renderContent = (content: string, contentType: string) => {
    if (contentType.startsWith('image/')) {
      return (
        <div className="flex justify-center p-4 bg-muted rounded-md">
          <Image
            src={content}
            alt="Document content"
            className="max-w-full max-h-96 object-contain"
            width={500}
            height={300}
          />
        </div>
      );
    } else if (content.startsWith('data:image/png;base64,') || content.startsWith('data:image/jpeg;base64,')) {
      return (
        <div className="flex justify-center p-4 bg-muted rounded-md">
          <Image
            src={content}
            alt="Base64 image content"
            className="max-w-full max-h-96 object-contain"
            width={500}
            height={300}
          />
        </div>
      );
    } else {
      return (
        <div className="bg-muted p-4 rounded-md whitespace-pre-wrap font-mono text-sm">
          {content}
        </div>
      );
    }
  };

  return (
    <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-3/4 p-3 rounded-lg ${
          role === 'user'
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted'
        }`}
      >
        <div className="whitespace-pre-wrap">{content}</div>

        {sources && sources.length > 0 && role === 'assistant' && (
          <Accordion type="single" collapsible className="mt-4">
            <AccordionItem value="sources">
              <AccordionTrigger className="text-xs">Sources ({sources.length})</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2">
                  {sources.map((source, index) => (
                    <div key={`${source.document_id}-${source.chunk_number}-${index}`} className="bg-background p-2 rounded text-xs border">
                      <div className="pb-2">
                        <div className="flex justify-between items-start">
                          <div>
                            <span className="font-medium">
                              {source.filename || `Document ${source.document_id.substring(0, 8)}...`}
                            </span>
                            <span className="text-muted-foreground ml-1">
                              Chunk {source.chunk_number} {source.score !== undefined && `• Score: ${source.score.toFixed(2)}`}
                            </span>
                          </div>
                          {source.content_type && (
                            <Badge variant="outline" className="text-[10px]">
                              {source.content_type}
                            </Badge>
                          )}
                        </div>
                      </div>

                      {source.content && (
                        renderContent(source.content, source.content_type || 'text/plain')
                      )}

                      <Accordion type="single" collapsible className="mt-3">
                        <AccordionItem value="metadata">
                          <AccordionTrigger className="text-[10px]">Metadata</AccordionTrigger>
                          <AccordionContent>
                            <pre className="bg-muted p-1 rounded text-[10px] overflow-x-auto">
                              {JSON.stringify(source.metadata, null, 2)}
                            </pre>
                          </AccordionContent>
                        </AccordionItem>
                      </Accordion>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        )}
      </div>
    </div>
  );
};

export default ChatMessageComponent;
