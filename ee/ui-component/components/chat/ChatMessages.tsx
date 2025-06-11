import React from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Copy, Check, Spin } from "./icons";
import Image from "next/image";
import { Source } from "@/components/types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

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

// Copy button component
function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 w-8 p-0"
      onClick={handleCopy}
      title={copied ? "Copied!" : "Copy message"}
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
    </Button>
  );
}

export function PreviewMessage({ message }: Pick<MessageProps, "message">) {
  const sources = message.experimental_customData?.sources;

  return (
    <div className="group relative flex px-4 py-3">
      <div className={` w-full `}>
        <div className={`mx-auto flex w-full flex-col max-w-4xl gap-4 ${message.role === "user" ? "items-end" : "items-start"}`}>
          <div className={`space-y-2 overflow-hidden max-w-3xl ${message.role === "user" ? "" : ""}`}>
            <div
              className={`relative rounded-xl p-4  ${
                message.role === "user" ? "ml-auto bg-primary text-primary-foreground " : "bg-muted pr-10"
              }`}
            >
              {message.role === "assistant" && (
                <div className="absolute right-2 top-2">
                  <CopyButton content={message.content} />
                </div>
              )}
              <div className="prose prose-sm dark:prose-invert max-w-none break-words">
                {message.role === "assistant" ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code(props) {
                        const { children, className, ...rest } = props;
                        const inline = !className?.includes("language-");
                        const match = /language-(\w+)/.exec(className || "");

                        if (!inline && match) {
                          const language = match[1];
                          return (
                            <div className="my-4 overflow-hidden rounded-md">
                              <SyntaxHighlighter style={oneDark} language={language} PreTag="div" className="!my-0">
                                {String(children).replace(/\n$/, "")}
                              </SyntaxHighlighter>
                            </div>
                          );
                        } else if (!inline) {
                          return (
                            <div className="my-4 overflow-hidden rounded-md">
                              <SyntaxHighlighter style={oneDark} language="text" PreTag="div" className="!my-0">
                                {String(children).replace(/\n$/, "")}
                              </SyntaxHighlighter>
                            </div>
                          );
                        }
                        return (
                          <code className="rounded bg-muted px-1 py-0.5 font-mono text-sm" {...rest}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  message.content // User messages rendered as plain text within prose-styled div
                )}
              </div>
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
