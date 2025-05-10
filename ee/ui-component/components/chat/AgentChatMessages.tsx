import React, { useEffect, useState } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { PreviewMessage, UIMessage } from "./ChatMessages";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";

// Add custom scrollbar styles
const scrollbarStyles = `
  .custom-scrollbar::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }

  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
  }

  .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(155, 155, 155, 0.5);
    border-radius: 3px;
  }

  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background-color: rgba(155, 155, 155, 0.7);
  }
`;

// Base interface for display objects
export interface BaseDisplayObject {
  source: string; // Source ID that links to the source
}

// Text display object interface
export interface TextDisplayObject extends BaseDisplayObject {
  type: 'text';
  content: string; // Markdown content
}

// Image display object interface
export interface ImageDisplayObject extends BaseDisplayObject {
  type: 'image';
  content: string; // Base64 encoded image
  caption: string; // Text describing the image
}

// Union type for all display object types
export type DisplayObject = TextDisplayObject | ImageDisplayObject;

// Source object interface
export interface SourceObject {
  sourceId: string;
  documentName: string;
  documentId: string;
  content?: string; // Content from the source
  contentType?: 'text' | 'image'; // Type of content
}

// Define interface for the Tool Call
export interface ToolCall {
  tool_name: string;
  tool_args: unknown;
  tool_result: unknown;
}

// Extended interface for UIMessage with tool history
export interface AgentUIMessage extends UIMessage {
  experimental_agentData?: {
    tool_history: ToolCall[];
    displayObjects?: DisplayObject[];
    sources?: SourceObject[];
  };
  isLoading?: boolean;
}

export interface AgentMessageProps {
  message: AgentUIMessage;
}

const thinkingPhrases = [
  { text: "Grokking the universe", emoji: "🌌" },
  { text: "Consulting the AI elders", emoji: "🧙‍♂️" },
  { text: "Mining for insights", emoji: "⛏️" },
  { text: "Pondering deeply", emoji: "🤔" },
  { text: "Connecting neural pathways", emoji: "🧠" },
  { text: "Brewing thoughts", emoji: "☕️" },
  { text: "Quantum computing...", emoji: "⚛️" },
  { text: "Traversing knowledge graphs", emoji: "🕸️" },
  { text: "Summoning wisdom", emoji: "✨" },
  { text: "Processing in parallel", emoji: "💭" },
  { text: "Analyzing patterns", emoji: "🔍" },
  { text: "Consulting documentation", emoji: "📚" },
  { text: "Debugging the matrix", emoji: "🐛" },
  { text: "Loading creativity modules", emoji: "🎨" },
];

const ThinkingMessage = () => {
  const [currentPhrase, setCurrentPhrase] = useState(thinkingPhrases[0]);
  const [dots, setDots] = useState("");

  useEffect(() => {
    // Rotate through phrases every 2 seconds
    const phraseInterval = setInterval(() => {
      setCurrentPhrase(prev => {
        const currentIndex = thinkingPhrases.findIndex(p => p.text === prev.text);
        const nextIndex = (currentIndex + 1) % thinkingPhrases.length;
        return thinkingPhrases[nextIndex];
      });
    }, 2000);

    // Animate dots every 500ms
    const dotsInterval = setInterval(() => {
      setDots(prev => (prev.length >= 3 ? "" : prev + "."));
    }, 500);

    return () => {
      clearInterval(phraseInterval);
      clearInterval(dotsInterval);
    };
  }, []);

  return (
    <div className="flex flex-col space-y-4 p-4">
      {/* Thinking Message */}
      <div className="flex items-center justify-start space-x-3 text-muted-foreground">
        <span className="animate-bounce text-xl">{currentPhrase.emoji}</span>
        <span className="text-sm font-medium">
          {currentPhrase.text}
          {dots}
        </span>
      </div>

      {/* Skeleton Loading */}
      <div className="space-y-3">
        <div className="flex space-x-2">
          <div className="h-4 w-4/12 animate-pulse rounded-md bg-muted"></div>
          <div className="h-4 w-3/12 animate-pulse rounded-md bg-muted"></div>
        </div>
        <div className="flex space-x-2">
          <div className="h-4 w-6/12 animate-pulse rounded-md bg-muted"></div>
          <div className="h-4 w-2/12 animate-pulse rounded-md bg-muted"></div>
        </div>
        <div className="h-4 w-8/12 animate-pulse rounded-md bg-muted"></div>
      </div>
    </div>
  );
};

// Helper to render JSON content with syntax highlighting
const renderJson = (obj: unknown) => {
  return (
    <pre className="max-h-[300px] overflow-auto whitespace-pre-wrap rounded-md bg-muted p-4 font-mono text-sm">
      {JSON.stringify(obj, null, 2)}
    </pre>
  );
};

// Markdown content renderer component
const MarkdownContent: React.FC<{ content: string }> = ({ content }) => {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="my-4 w-full overflow-x-auto">
              <table className="w-full border-collapse border border-border">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
          tbody: ({ children }) => <tbody className="divide-y divide-border">{children}</tbody>,
          tr: ({ children }) => <tr className="divide-x divide-border">{children}</tr>,
          th: ({ children }) => <th className="p-3 text-left font-semibold">{children}</th>,
          td: ({ children }) => <td className="p-3">{children}</td>,
          h1: ({ children }) => <h1 className="mb-4 text-2xl font-bold">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-3 text-xl font-bold">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-2 text-lg font-bold">{children}</h3>,
          p: ({ children }) => <p className="mb-4 leading-relaxed">{children}</p>,
          ul: ({ children }) => <ul className="mb-4 list-disc space-y-2 pl-6">{children}</ul>,
          ol: ({ children }) => <ol className="mb-4 list-decimal space-y-2 pl-6">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="my-4 border-l-4 border-gray-300 pl-4 italic dark:border-gray-600">
              {children}
            </blockquote>
          ),
          code({ className, children }) {
            const match = /language-(\w+)/.exec(className || "");
            const language = match ? match[1] : "";
            const isInline = !className;

            if (!isInline && language) {
              return (
                <div className="my-4 overflow-hidden rounded-md">
                  <SyntaxHighlighter style={oneDark} language={language} PreTag="div" className="!my-0">
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                </div>
              );
            }

            return isInline ? (
              <code className="rounded bg-muted px-1.5 py-0.5 text-sm">{children}</code>
            ) : (
              <div className="my-4 overflow-hidden rounded-md">
                <SyntaxHighlighter style={oneDark} language="text" PreTag="div" className="!my-0">
                  {String(children).replace(/\n$/, "")}
                </SyntaxHighlighter>
              </div>
            );
          },
          a: ({ href, children }) => (
            <a href={href} className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          strong: ({ children }) => <strong className="font-bold">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          hr: () => <hr className="my-8 border-t border-gray-200 dark:border-gray-700" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

// Helper function to ensure image content is properly formatted
const formatImageSource = (content: string): string => {
  // If already a data URI, use as is
  if (content.startsWith('data:image/')) {
    return content;
  }

  // If it looks like base64 data without a prefix, add a PNG prefix
  if (/^[A-Za-z0-9+/=]+$/.test(content.substring(0, 20))) {
    return `data:image/png;base64,${content}`;
  }

  // Otherwise assume it's a URL
  return content;
};

// Component to render display objects
const DisplayObjectRenderer: React.FC<{ object: DisplayObject; isInSourceView?: boolean }> = ({
  object,
  isInSourceView = false
}) => {
  if (object.type === 'text') {
    return (
      <div className={isInSourceView ? "my-1" : "my-2"}>
        <MarkdownContent content={object.content} />
        {!isInSourceView && (
          <div className="mt-1 text-xs text-muted-foreground">
            Source: {object.source}
          </div>
        )}
      </div>
    );
  } else if (object.type === 'image') {
    // Try to determine if the content has the image format information
    const hasImagePrefix = object.content.startsWith('data:image');

    return (
      <div className={isInSourceView ? "my-1" : "my-2"}>
        <div className="overflow-hidden rounded-md">
          <img
            src={hasImagePrefix ? object.content : `data:image/png;base64,${object.content}`}
            alt={object.caption || 'Image'}
            className="max-w-full h-auto"
            onError={(e) => {
              // Fallback chain: try JPEG if PNG fails
              const target = e.target as HTMLImageElement;
              if (!hasImagePrefix && target.src.includes('image/png')) {
                target.src = `data:image/jpeg;base64,${object.content}`;
              }
            }}
          />
        </div>
        {object.caption && (
          <div className="mt-1 text-sm text-muted-foreground">{object.caption}</div>
        )}
        {!isInSourceView && (
          <div className="mt-1 text-xs text-muted-foreground">
            Source: {object.source}
          </div>
        )}
      </div>
    );
  }
  return null;
};

// Helper function to detect image format from base64
const detectImageFormat = (base64String: string): string => {
  // Check the first few bytes of the base64 to determine image format
  // See: https://en.wikipedia.org/wiki/List_of_file_signatures
  try {
    if (!base64String || base64String.length < 10) return 'png';

    // Decode the first few bytes of the base64 string
    const firstBytes = atob(base64String.substring(0, 20));

    // Check for common image signatures
    if (firstBytes.startsWith('\xFF\xD8\xFF')) return 'jpeg';
    if (firstBytes.startsWith('\x89PNG\r\n\x1A\n')) return 'png';
    if (firstBytes.startsWith('GIF87a') || firstBytes.startsWith('GIF89a')) return 'gif';
    if (firstBytes.startsWith('RIFF') && firstBytes.substring(8, 12) === 'WEBP') return 'webp';

    // Default to PNG if signature not recognized
    return 'png';
  } catch (e) {
    // If any error in detection, default to PNG
    return 'png';
  }
};

// Component to render sources as tags with dropdown content
const SourcesRenderer: React.FC<{ sources: SourceObject[]; displayObjects?: DisplayObject[] }> = ({
  sources,
  displayObjects = []
}) => {
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [animation, setAnimation] = useState<'open' | 'close' | null>(null);
  const [visibleContent, setVisibleContent] = useState<string | null>(null);

  useEffect(() => {
    // Set animation state when source is selected
    if (selectedSource) {
      setAnimation('open');
      setVisibleContent(selectedSource);
    }
  }, [selectedSource]);

  if (!sources || sources.length === 0) return null;

  // Toggle source selection with animation
  const toggleSource = (sourceId: string) => {
    if (selectedSource === sourceId) {
      // Close animation
      setAnimation('close');
      setTimeout(() => {
        setSelectedSource(null);
        setVisibleContent(null);
        setAnimation(null);
      }, 200);
    } else {
      // Just update selected source, effect will handle animation
      setSelectedSource(sourceId);
    }
  };

  // Render source content
  const renderSourceContent = (source: SourceObject) => {
    if (!source.content) {
      return (
        <div className="text-xs text-muted-foreground">
          No content available for this source.
        </div>
      );
    }

    // Render based on contentType
    if (source.contentType === 'image') {
      const content = source.content;

      // Handle different image formats similar to ChatMessages.tsx
      if (content.startsWith('data:image/') ||
          // If it looks like base64, we'll try as PNG
          /^[A-Za-z0-9+/=]+$/.test(content.substring(0, 20))) {

        // Format similar to how ChatMessages.tsx does it
        const imageUrl = content.startsWith('data:image/')
          ? content
          : `data:image/png;base64,${content}`;

        return (
          <div className="flex justify-center rounded-md bg-muted p-4">
            <img
              src={imageUrl}
              alt={`Image from ${source.documentName}`}
              className="max-h-96 max-w-full object-contain"
            />
          </div>
        );
      } else {
        // If not recognizable as an image format, show as text
        return (
          <div className="whitespace-pre-wrap rounded-md bg-muted p-4 font-mono text-sm">
            {content}
          </div>
        );
      }
    } else {
      // Default to text/markdown with scrollable area
      return (
        <div className="max-h-[300px] overflow-y-auto pr-1 custom-scrollbar">
          <MarkdownContent content={source.content} />
        </div>
      );
    }
  };

  return (
    <div className="mt-3">
      {/* Inject scrollbar styles */}
      <style>{scrollbarStyles}</style>

      <div className="mb-2 flex flex-wrap gap-2">
        {sources.map((source) => (
          <Badge
            key={source.sourceId}
            variant={selectedSource === source.sourceId ? "default" : "outline"}
            className="cursor-pointer px-3 py-1 text-xs hover:bg-primary/10"
            onClick={() => toggleSource(source.sourceId)}
          >
            {source.documentName}
          </Badge>
        ))}
      </div>

      {/* Display selected source content */}
      {visibleContent && (
        <div
          className={`mt-3 overflow-hidden rounded-md border bg-card shadow-sm transition-all duration-200 ease-in-out ${
            animation === 'open' ? 'max-h-[400px] opacity-100 p-3' :
            animation === 'close' ? 'max-h-0 opacity-0 p-0' : 'p-3'
          }`}
        >
          <div className="mb-2 flex items-center justify-between">
            <div className="text-xs font-medium">
              Source: {sources.find(s => s.sourceId === visibleContent)?.documentName}
            </div>
            <button
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
              onClick={() => toggleSource(visibleContent)}
            >
              Close
            </button>
          </div>

          {/* Show source content if available */}
          <div className="rounded-sm bg-muted/50 p-2">
            {renderSourceContent(sources.find(s => s.sourceId === visibleContent)!)}
          </div>
        </div>
      )}
    </div>
  );
};

export function AgentPreviewMessage({ message }: AgentMessageProps) {
  const displayObjects = message.experimental_agentData?.displayObjects;
  const sources = message.experimental_agentData?.sources;

  // If this is a loading state, show the thinking message
  if (message.isLoading) {
    return <ThinkingMessage />;
  }

  // For user messages, render standard message
  if (message.role === "user") {
    return <PreviewMessage message={message} />;
  }

  // For assistant messages with no display objects, show regular message
  if (!displayObjects || displayObjects.length === 0) {
    return <PreviewMessage message={message} />;
  }

  // Show only display objects for assistant messages that have them
  return (
    <div className="group relative flex px-4 py-3">
      <div className="flex w-full flex-col items-start">
        <div className="flex w-full max-w-3xl items-start gap-4">
          <div className="flex-1 space-y-2 overflow-hidden">
            <div className="rounded-xl bg-muted p-4">
              <div className="space-y-3">
                {displayObjects.map((obj, idx) => (
                  <DisplayObjectRenderer key={idx} object={obj} />
                ))}
              </div>

              {/* Render sources if available */}
              {sources && sources.length > 0 && (
                <div className="mt-4 border-t border-border pt-3">
                  <SourcesRenderer sources={sources} displayObjects={displayObjects} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
