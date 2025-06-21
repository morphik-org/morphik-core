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
    <div className="mx-auto max-w-4xl py-6">
      <div className="flex items-center text-sm text-muted-foreground">
        <Spin className="mr-2 h-4 w-4 animate-spin" />
        <span>Thinking...</span>
      </div>
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
      className="h-8 w-8 p-0 text-muted-foreground transition-colors hover:text-foreground"
      onClick={handleCopy}
      title={copied ? "Copied!" : "Copy message"}
    >
      {copied ? <Check className="h-4 w-4 text-green-600 dark:text-green-400" /> : <Copy className="h-4 w-4" />}
    </Button>
  );
}

export function PreviewMessage({ message }: Pick<MessageProps, "message">) {
  const sources = message.experimental_customData?.sources;

  return (
    <div className="group relative py-4">
      <div className="mx-auto w-full max-w-4xl">
        {message.role === "user" ? (
          // User message - full width with grey background and thin black border
          <div className="w-full rounded-lg border border-black/10 bg-gray-100 p-4 dark:border-white/10 dark:bg-zinc-900">
            <div className="text-[15px] leading-relaxed">{message.content}</div>
          </div>
        ) : (
          // Assistant message - no border, just markdown
          <div className="relative w-full text-[15px]">
            <div className="absolute -right-2 -top-2">
              <CopyButton content={message.content} />
            </div>
            <div className="w-full">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ children }) => <p className="mb-4 leading-7 last:mb-0">{children}</p>,
                  h1: ({ children }) => <h1 className="mb-4 text-2xl font-semibold">{children}</h1>,
                  h2: ({ children }) => <h2 className="mb-3 text-xl font-semibold">{children}</h2>,
                  h3: ({ children }) => <h3 className="mb-2 text-lg font-semibold">{children}</h3>,
                  ul: ({ children }) => <ul className="mb-4 list-disc space-y-1 pl-6">{children}</ul>,
                  ol: ({ children }) => <ol className="mb-4 list-decimal space-y-1 pl-6">{children}</ol>,
                  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      className="text-primary underline-offset-2 hover:underline"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {children}
                    </a>
                  ),
                  blockquote: ({ children }) => (
                    <blockquote className="my-4 border-l-4 border-gray-300 pl-4 italic text-muted-foreground dark:border-gray-600">
                      {children}
                    </blockquote>
                  ),
                  code(props) {
                    const { children, className, ...rest } = props;
                    const inline = !className?.includes("language-");
                    const match = /language-(\w+)/.exec(className || "");

                    if (!inline && match) {
                      const language = match[1];
                      return (
                        <div className="my-4 overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800">
                          <SyntaxHighlighter
                            style={oneDark}
                            language={language}
                            PreTag="div"
                            className="!my-0 !bg-zinc-900"
                            customStyle={{
                              padding: "1rem",
                              fontSize: "0.875rem",
                              lineHeight: "1.5",
                            }}
                          >
                            {String(children).replace(/\n$/, "")}
                          </SyntaxHighlighter>
                        </div>
                      );
                    } else if (!inline) {
                      return (
                        <div className="my-4 overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800">
                          <SyntaxHighlighter
                            style={oneDark}
                            language="text"
                            PreTag="div"
                            className="!my-0 !bg-zinc-900"
                            customStyle={{
                              padding: "1rem",
                              fontSize: "0.875rem",
                              lineHeight: "1.5",
                            }}
                          >
                            {String(children).replace(/\n$/, "")}
                          </SyntaxHighlighter>
                        </div>
                      );
                    }
                    return (
                      <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-sm dark:bg-zinc-800" {...rest}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Sources for assistant messages */}
            {sources && sources.length > 0 && message.role === "assistant" && (
              <Accordion
                type="single"
                collapsible
                className="mt-4 overflow-hidden rounded-lg border border-border/50 bg-muted/30"
              >
                <AccordionItem value="sources" className="border-0">
                  <AccordionTrigger className="px-4 py-3 text-sm font-medium hover:bg-muted/50">
                    <span className="flex items-center gap-2">
                      <span className="text-muted-foreground">📚</span>
                      Sources ({sources.length})
                    </span>
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
        )}
      </div>
    </div>
  );
}
