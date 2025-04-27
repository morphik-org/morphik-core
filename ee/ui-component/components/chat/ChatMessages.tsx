import React from 'react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Spin } from './icons';
import Image from 'next/image';
import { Source } from '@/components/types';

// Define interface for the UIMessage - matching what our useMorphikChat hook returns
export interface UIMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: Date;
  experimental_customData?: { sources: Source[] };
}

export interface MessageProps {
  chatId: string;
  message: UIMessage;
  isLoading?: boolean;
  setMessages: (messages: UIMessage[]) => void;
  reload: () => void;
  isReadonly: boolean;
}

export function ThinkingMessage() {
  return (
    <div className="flex items-center justify-center h-12 text-center text-xs text-muted-foreground">
      <Spin className="mr-2 animate-spin" />
      Thinking...
    </div>
  );
}

// Helper to render source content based on content type
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

export function PreviewMessage({ message }: Pick<MessageProps, 'message'>) {
  const sources = message.experimental_customData?.sources;

  return (
    <div className="px-4 py-3 flex group relative">
      <div className={`flex flex-col w-full ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
        <div className="flex items-start gap-4 w-full max-w-3xl">

          <div className={`flex-1 space-y-2 overflow-hidden ${message.role === 'user' ? '' : ''}`}>
            <div className={`p-4 rounded-xl ${
              message.role === 'user'
                ? 'bg-primary text-primary-foreground ml-auto'
                : 'bg-muted'
            }`}>
              <div className="prose prose-sm dark:prose-invert break-words">
                {message.content}
              </div>
            </div>

            {sources && sources.length > 0 && message.role === 'assistant' && (
              <Accordion type="single" collapsible className="mt-2 border rounded-xl overflow-hidden">
                <AccordionItem value="sources" className="border-0">
                  <AccordionTrigger className="px-4 py-2 text-sm font-medium">
                    Sources ({sources.length})
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-3">
                    <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                      {sources.map((source, index) => (
                        <div key={`${source.document_id}-${source.chunk_number}-${index}`}
                             className="bg-background rounded-md border overflow-hidden">
                          <div className="p-3 border-b">
                            <div className="flex justify-between items-start">
                              <div>
                                <span className="font-medium text-sm">
                                  {source.filename || `Document ${source.document_id.substring(0, 8)}...`}
                                </span>
                                <div className="text-xs text-muted-foreground mt-0.5">
                                  Chunk {source.chunk_number} {source.score !== undefined && `• Score: ${source.score.toFixed(2)}`}
                                </div>
                              </div>
                              {source.content_type && (
                                <Badge variant="outline" className="text-[10px]">
                                  {source.content_type}
                                </Badge>
                              )}
                            </div>
                          </div>

                          {source.content && (
                            <div className="px-3 py-2">
                              {renderContent(source.content, source.content_type || 'text/plain')}
                            </div>
                          )}

                          <Accordion type="single" collapsible className="border-t">
                            <AccordionItem value="metadata" className="border-0">
                              <AccordionTrigger className="px-3 py-2 text-xs">
                                Metadata
                              </AccordionTrigger>
                              <AccordionContent className="px-3 pb-3">
                                <pre className="bg-muted p-2 rounded text-xs overflow-x-auto">
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
      </div>
    </div>
  );
}
