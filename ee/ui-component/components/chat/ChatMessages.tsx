import React from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Spin } from "./icons";
import Image from "next/image";
import { Source } from "@/components/types";

// Define interface for the UIMessage - matching what our useMorphikChat hook returns
export interface UIMessage {
  id: string;
  role: "user" | "assistant";
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
    <div className="flex h-12 items-center justify-center text-center text-xs text-muted-foreground">
      <Spin className="mr-2 animate-spin" />
      Thinking...
    </div>
  );
}

// Helper to render source content based on content type
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

export function PreviewMessage({ message }: Pick<MessageProps, "message">) {
  const sources = message.experimental_customData?.sources;

  return (
    <div className="group relative flex px-4 py-3">
      <div className={`flex w-full flex-col ${message.role === "user" ? "items-end" : "items-start"}`}>
        <div className="flex w-full max-w-3xl items-start gap-4">
          <div className={`flex-1 space-y-2 overflow-hidden ${message.role === "user" ? "" : ""}`}>
            <div
              className={`rounded-xl p-4 ${
                message.role === "user" ? "ml-auto bg-primary text-primary-foreground" : "bg-muted"
              }`}
            >
              <div className="prose prose-sm dark:prose-invert break-words">{message.content}</div>
            </div>

            {sources && sources.length > 0 && message.role === "assistant" && (
              <Accordion type="single" collapsible className="mt-2 overflow-hidden rounded-xl border">
                <AccordionItem value="sources" className="border-0">
                  <AccordionTrigger className="px-4 py-2 text-sm font-medium">
                    Sources ({sources.length})
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-3">
                    <div className="max-h-[400px] space-y-3 overflow-y-auto pr-2">
                      {sources.map((source, index) => (
                        <div
                          key={`${source.document_id}-${source.chunk_number}-${index}`}
                          className="overflow-hidden rounded-md border bg-background"
                        >
                          <div className="border-b p-3">
                            <div className="flex items-start justify-between">
                              <div>
                                <span className="text-sm font-medium">
                                  {source.filename || `Document ${source.document_id.substring(0, 8)}...`}
                                </span>
                                <div className="mt-0.5 text-xs text-muted-foreground">
                                  Chunk {source.chunk_number}{" "}
                                  {source.score !== undefined && `• Score: ${source.score.toFixed(2)}`}
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
                              {renderContent(source.content, source.content_type || "text/plain")}
                            </div>
                          )}

                          <Accordion type="single" collapsible className="border-t">
                            <AccordionItem value="metadata" className="border-0">
                              <AccordionTrigger className="px-3 py-2 text-xs">Metadata</AccordionTrigger>
                              <AccordionContent className="px-3 pb-3">
                                <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
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
