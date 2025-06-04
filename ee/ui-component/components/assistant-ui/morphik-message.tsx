"use client";

import { forwardRef } from "react";
import { MessagePrimitive } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Copy, RefreshCw, Edit } from "lucide-react";

interface Source {
  id: string;
  filename: string;
  content: string;
  score?: number;
  page_number?: number;
  chunk_index?: number;
  metadata?: Record<string, any>;
}

interface MorphikMessageContentProps {
  className?: string;
}

const MorphikMessageContent = forwardRef<HTMLDivElement, MorphikMessageContentProps>(
  ({ className, ...props }, ref) => {
    return (
      <MessagePrimitive.Content
        ref={ref}
        className={cn("prose prose-sm max-w-none dark:prose-invert", className)}
        {...props}
      />
    );
  }
);
MorphikMessageContent.displayName = "MorphikMessageContent";

interface MorphikMessageSourcesProps {
  className?: string;
}

const MorphikMessageSources = forwardRef<HTMLDivElement, MorphikMessageSourcesProps>(
  ({ className, ...props }, ref) => {
    // Access sources from global state (as set by our runtime)
    const sources = (globalThis as any).lastQuerySources as Source[] | undefined;

    if (!sources || sources.length === 0) {
      return null;
    }

    return (
      <div ref={ref} className={cn("mt-4", className)} {...props}>
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="sources" className="border rounded-lg">
            <AccordionTrigger className="px-4 py-2 hover:no-underline">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Sources</span>
                <Badge variant="secondary" className="text-xs">
                  {sources.length}
                </Badge>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-4 pb-4">
              <div className="space-y-3">
                {sources.map((source, index) => (
                  <div key={source.id || index} className="border rounded-lg p-3 bg-muted/50">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium truncate">{source.filename}</h4>
                        <div className="flex items-center gap-2 mt-1">
                          {source.score && (
                            <Badge variant="outline" className="text-xs">
                              Score: {source.score.toFixed(3)}
                            </Badge>
                          )}
                          {source.page_number && (
                            <Badge variant="outline" className="text-xs">
                              Page {source.page_number}
                            </Badge>
                          )}
                          {source.chunk_index !== undefined && (
                            <Badge variant="outline" className="text-xs">
                              Chunk {source.chunk_index}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                    {source.content && (
                      <div className="text-xs text-muted-foreground bg-background/50 rounded p-2 border">
                        <p className="line-clamp-3">{source.content}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    );
  }
);
MorphikMessageSources.displayName = "MorphikMessageSources";

interface MorphikMessageProps {
  className?: string;
}

const MorphikMessage = forwardRef<HTMLDivElement, MorphikMessageProps>(
  ({ className, ...props }, ref) => {
    return (
      <MessagePrimitive.Root
        ref={ref}
        className={cn(
          "group relative mb-6 flex items-start gap-3",
          "data-[role=user]:flex-row-reverse data-[role=user]:justify-start",
          className
        )}
        {...props}
      >
        {/* Avatar */}
        <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full border bg-background shadow">
          <MessagePrimitive.If role="user">
            <div className="h-4 w-4 bg-primary rounded-full" />
          </MessagePrimitive.If>
          <MessagePrimitive.If role="assistant">
            <div className="h-4 w-4 bg-muted-foreground rounded-full" />
          </MessagePrimitive.If>
        </div>

        {/* Message Content */}
        <div className="flex-1 space-y-2 overflow-hidden">
          <div className="flex items-center gap-2">
            <MessagePrimitive.If role="user">
              <span className="text-sm font-medium">You</span>
            </MessagePrimitive.If>
            <MessagePrimitive.If role="assistant">
              <span className="text-sm font-medium">Morphik</span>
            </MessagePrimitive.If>
            <MessagePrimitive.If editing={false}>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
                <MessagePrimitive.Copy asChild>
                  <Button variant="ghost" size="icon" className="h-6 w-6">
                    <Copy className="h-3 w-3" />
                  </Button>
                </MessagePrimitive.Copy>
                <MessagePrimitive.If role="assistant">
                  <MessagePrimitive.Reload asChild>
                    <Button variant="ghost" size="icon" className="h-6 w-6">
                      <RefreshCw className="h-3 w-3" />
                    </Button>
                  </MessagePrimitive.Reload>
                </MessagePrimitive.If>
                <MessagePrimitive.Edit asChild>
                  <Button variant="ghost" size="icon" className="h-6 w-6">
                    <Edit className="h-3 w-3" />
                  </Button>
                </MessagePrimitive.Edit>
              </div>
            </MessagePrimitive.If>
          </div>

          <MessagePrimitive.If editing={false}>
            <MorphikMessageContent />
            <MessagePrimitive.If role="assistant">
              <MorphikMessageSources />
            </MessagePrimitive.If>
          </MessagePrimitive.If>

          <MessagePrimitive.If editing>
            <MessagePrimitive.Editor className="min-h-[100px] w-full resize-none border-0 bg-transparent p-0 text-sm focus:outline-none" />
          </MessagePrimitive.If>
        </div>
      </MessagePrimitive.Root>
    );
  }
);
MorphikMessage.displayName = "MorphikMessage";

export { MorphikMessage, MorphikMessageContent, MorphikMessageSources };
