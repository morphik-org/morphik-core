import React, { useEffect, useState } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { PreviewMessage, UIMessage } from "./ChatMessages";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import { Citation, ImageCitation, TextCitation } from "@/types/agent-response";

// Define interface for the Tool Call
interface ToolCall {
  name?: string;
  args?: any;
  response?: any;
  tool_name?: string;
  tool_args?: any;
  tool_result?: any;
}

// Extended interface for UIMessage with tool history
interface AgentResponseData {
  body?: string;
  mode?: 'plain' | 'rich';
  display_instructions?: string;
  display_elements?: any[];
  text?: string; 
  citations?: any[];
  tool_history?: any[];
}

interface AgentRichResponse extends AgentResponseData {
  mode: 'rich';
  body: string;
  citations: any[];
  text?: string; 
}

export interface AgentUIMessage extends UIMessage {
  experimental_agentData?: {
    tool_history: ToolCall[];
  };
  isLoading?: boolean;
  // content remains string as per UIMessage
  // richResponse will hold the new structured data
  richResponse?: AgentResponseData;
  displayElements?: any[];
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

// Updated MarkdownContent Props to include citations
interface MarkdownContentProps {
  content: string | object;
  citations?: Citation[]; 
}

// Markdown content renderer component
const MarkdownContent: React.FC<MarkdownContentProps> = ({ content, citations }) => {
  const contentString = typeof content === 'string' ? content : JSON.stringify(content);
  // console.log("MarkdownContent received body:", contentString); 
  // console.log("MarkdownContent received citations:", citations); 

  const processNode = (node: React.ReactNode): React.ReactNode => {
    if (typeof node === 'string') {
      const parts: React.ReactNode[] = []; 
      let lastIndex = 0;
      const regex = /\[ref:([a-zA-Z0-9_-]+)\]/g;
      let match;

      while ((match = regex.exec(node)) !== null) {
        if (match.index > lastIndex) {
          parts.push(node.substring(lastIndex, match.index));
        }

        const citationId = match[1];
        // console.log(`Found placeholder for citation ID: ${citationId}`); 
        const citation = citations?.find(c => c.id === citationId);

        if (citation) {
          // Check if a text citation actually contains an image data URI
          let renderAsImage = citation.type === 'image';
          let imageUrlForRender = "";
          let imageAltText = `Referenced Image: ${citation.id} from ${citation.sourceDocId}`;

          if (citation.type === 'image') {
            const imgCitation = citation as ImageCitation;
            imageUrlForRender = imgCitation.imageUrl;
            imageAltText = imgCitation.snippet || `Image from ${imgCitation.sourceDocId} (ID: ${imgCitation.id})`;
          } else if (citation.type === 'text') { 
            const textSnippet = (citation as TextCitation).snippet;
            if (textSnippet && textSnippet.startsWith("data:image")) {
              renderAsImage = true;
              imageUrlForRender = textSnippet; 
              imageAltText = `Cited Image Data (Ref: ${citation.id})`;
              // console.log(`Citation ${citationId} is type text but snippet is an image. Rendering as image.`); 
            }
          }
          if (renderAsImage) {
            if (!imageUrlForRender) {
              parts.push(`[Missing Image Data: ${citationId}]`);
            } else {
              parts.push(
                <span key={`citation-${citationId}-${match.index}`} className="inline-flex flex-col items-center align-middle mx-1 my-1 p-1 border rounded max-w-xs">
                  <img
                    src={imageUrlForRender}
                    alt={imageAltText}
                    className="max-w-full h-auto rounded border"
                    style={{ display: 'block' }}
                  />
                  {(citation.type === 'image' && (citation as ImageCitation).snippet) &&
                    <p className="text-xs italic text-muted-foreground text-center mt-1 px-1">{imageAltText}</p>}
                </span>
              );
            }
          } else { 
            const textCitation = citation as TextCitation;
            const displaySnippet = textCitation.snippet?.substring(0, 70) + (textCitation.snippet && textCitation.snippet.length > 70 ? "..." : "");
            const titleText = `Ref: ${textCitation.id} | SourceDoc: ${textCitation.sourceDocId} | ChunkId: ${textCitation.chunkId}` +
                            (textCitation.reasoning ? `\nReasoning: ${textCitation.reasoning}` : "");
            parts.push(
              <span
                key={`citation-${citationId}-${match.index}`}
                className="inline-block align-middle mx-1 px-1.5 py-0.5 border border-dashed border-primary/50 rounded bg-primary/10 text-xs cursor-help"
                title={titleText}
              >
                📝 <span className="italic">{displaySnippet}</span> (Ref: {citationId})
              </span>
            );
          }
        } else {
          // console.warn(`Citation ID ${citationId} found in text but not in citations array.`); 
          parts.push(match[0]); 
        }
        lastIndex = regex.lastIndex;
      }

      if (lastIndex < node.length) {
        parts.push(node.substring(lastIndex));
      }
      return parts.length > 0 ? <>{parts}</> : node;
    }

    if (React.isValidElement(node) && node.props.children) {
      return React.cloneElement(node, { ...node.props, children: React.Children.map(node.props.children, processNode) });
    }

    return node;
  };

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
          p: ({ node, ...props }) => <p {...props}>{React.Children.map(props.children, child => processNode(child))}</p>,
          li: ({ node, ...props }) => <li {...props}>{React.Children.map(props.children, child => processNode(child))}</li>,
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
              <code className="rounded bg-muted px-1.5 py-0.5 text-sm">{String(children)}</code>
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
        {contentString}
      </ReactMarkdown>
    </div>
  );
};

// New component to render individual citations
const CitationView: React.FC<{ citation: Citation }> = ({ citation }) => {
  const renderCitationContent = () => {
    if (citation.type === 'image') {
      const imgCitation = citation as ImageCitation;
      return (
        <div className="mt-2">
          <img src={imgCitation.imageUrl} alt={imgCitation.snippet || `Image ${imgCitation.id}`} className="max-w-xs rounded border" />
          {imgCitation.bbox && (
            <p className="text-xs text-muted-foreground">Bounding box: {JSON.stringify(imgCitation.bbox)}</p>
          )}
          {imgCitation.snippet && <p className="mt-1 text-xs italic text-muted-foreground">{imgCitation.snippet}</p>}
        </div>
      );
    }
    // Default to text citation
    const textCitation = citation as TextCitation;
    return (
      <div className="mt-2">
        <p className="text-xs text-muted-foreground">Source: Doc {textCitation.sourceDocId}, Chunk {textCitation.chunkId}</p>
        <blockquote className="mt-1 border-l-2 pl-2 text-xs italic">
          {textCitation.snippet}
        </blockquote>
      </div>
    );
  };

  return (
    <div className="mt-2 rounded-md border bg-background/50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold">
          Citation ID: {citation.id} ({citation.type})
        </span>
        {citation.grounded !== undefined && (
          <Badge variant={citation.grounded ? "default" : "secondary"} className="text-[10px]">
            {citation.grounded ? "Grounded" : "Not Grounded"}
          </Badge>
        )}
      </div>
      {renderCitationContent()}
    </div>
  );
};

export function AgentPreviewMessage({ message }: AgentMessageProps) {
  const toolHistory = message.experimental_agentData?.tool_history;

  if (message.isLoading) {
    return <ThinkingMessage />;
  }

  // Handle User Messages separately if PreviewMessage is primarily for them
  if (message.role === 'user') {
    return <PreviewMessage message={message} />;
  }

  // Handle Assistant Messages
  // Check if it's a rich response
  if (message.richResponse && (message.richResponse.text || message.richResponse.body)) {
    const { text, body, citations } = message.richResponse;
    return (
      <div className="group relative flex px-4 py-3">
        <div className="flex-shrink-0 mr-3">{/* Placeholder for an avatar or icon */}</div>
        <div className="flex-1 space-y-2 overflow-hidden">
          <div className="rounded-xl bg-muted p-4">
            <MarkdownContent content={text || body || ''} citations={citations || []} />
          </div>
          {message.displayElements && message.displayElements.length > 0 && (
            <div className="mt-2 space-y-4">
              {message.displayElements.map((element, index) => (
                <div key={index} className="border-l-4 border-blue-500 pl-3">
                  {element.type === 'text' && (
                    <div className="text-sm text-gray-700">
                      <MarkdownContent content={element.element.content} citations={[]} />
                    </div>
                  )}
                  {element.type === 'image' && (
                    <div className="text-sm">
                      <p className="font-semibold">Image: {element.element.description}</p>
                      {element.element.bounding_box && (
                        <div className="mt-1 text-xs text-gray-500">
                          Bounding Box: {JSON.stringify(element.element.bounding_box.bbox)}
                          {element.element.bounding_box.image_number && (
                            <span> (Image #{element.element.bounding_box.image_number})</span>
                          )}
                        </div>
                      )}
                      {/* Placeholder for actual image rendering */}
                      <div className="mt-2 h-48 w-full bg-gray-200 rounded-md flex items-center justify-center">
                        <span className="text-gray-500">Image Placeholder</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {toolHistory && toolHistory.length > 0 && (
            <div className="mt-2 text-xs text-muted-foreground">
              {toolHistory.length === 1
                ? "1 tool was used to produce this response."
                : `${toolHistory.length} tools were used to produce this response.`}
              <Accordion type="single" collapsible className="w-full mt-1">
                <AccordionItem value="tool-history">
                  <AccordionTrigger>View Tool History</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {toolHistory.map((toolCall, index) => (
                        <div key={index} className="p-2 bg-background rounded border text-xs">
                          <div className="font-medium">Tool: {toolCall.name || toolCall.tool_name || 'Unknown'}</div>
                          {(toolCall.args || toolCall.tool_args) && (
                            <div className="mt-1">
                              <span className="font-medium">Arguments:</span>
                              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                                {JSON.stringify(toolCall.args || toolCall.tool_args, null, 2)}
                              </pre>
                            </div>
                          )}
                          {(toolCall.response || toolCall.tool_result) && (
                            <div className="mt-1">
                              <span className="font-medium">Response:</span>
                              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                                {JSON.stringify(toolCall.response || toolCall.tool_result, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          )}
        </div>
      </div>
    );
  } else {
    // Plain assistant message (not rich, but may have message.content and/or toolHistory)
    return (
      <div className="group relative flex px-4 py-3">
        <div className="flex-shrink-0 mr-3">{/* Placeholder for an avatar or icon */}</div>
        <div className="flex-1 space-y-2 overflow-hidden">
          {message.content && (
            <div className="rounded-xl bg-muted p-4">
              <MarkdownContent content={message.content} citations={[]} />
            </div>
          )}
          {message.displayElements && message.displayElements.length > 0 && (
            <div className="mt-2 space-y-4">
              {message.displayElements.map((element, index) => (
                <div key={index} className="border-l-4 border-blue-500 pl-3">
                  {element.type === 'text' && (
                    <div className="text-sm text-gray-700">
                      <MarkdownContent content={element.element.content} citations={[]} />
                    </div>
                  )}
                  {element.type === 'image' && (
                    <div className="text-sm">
                      <p className="font-semibold">Image: {element.element.description}</p>
                      {element.element.bounding_box && (
                        <div className="mt-1 text-xs text-gray-500">
                          Bounding Box: {JSON.stringify(element.element.bounding_box.bbox)}
                          {element.element.bounding_box.image_number && (
                            <span> (Image #{element.element.bounding_box.image_number})</span>
                          )}
                        </div>
                      )}
                      {/* Placeholder for actual image rendering */}
                      <div className="mt-2 h-48 w-full bg-gray-200 rounded-md flex items-center justify-center">
                        <span className="text-gray-500">Image Placeholder</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {toolHistory && toolHistory.length > 0 && (
            <div className="mt-2 text-xs text-muted-foreground">
              {toolHistory.length === 1
                ? "1 tool was used to produce this response."
                : `${toolHistory.length} tools were used to produce this response.`}
              <Accordion type="single" collapsible className="w-full mt-1">
                <AccordionItem value="tool-history">
                  <AccordionTrigger>View Tool History</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {toolHistory.map((toolCall, index) => (
                        <div key={index} className="p-2 bg-background rounded border text-xs">
                          <div className="font-medium">Tool: {toolCall.name || toolCall.tool_name || 'Unknown'}</div>
                          {(toolCall.args || toolCall.tool_args) && (
                            <div className="mt-1">
                              <span className="font-medium">Arguments:</span>
                              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                                {JSON.stringify(toolCall.args || toolCall.tool_args, null, 2)}
                              </pre>
                            </div>
                          )}
                          {(toolCall.response || toolCall.tool_result) && (
                            <div className="mt-1">
                              <span className="font-medium">Response:</span>
                              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                                {JSON.stringify(toolCall.response || toolCall.tool_result, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          )}
        </div>
      </div>
    );
  }
}
