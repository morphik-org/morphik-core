import React, { useEffect, useState } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { PreviewMessage, UIMessage } from "./ChatMessages";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import { AgentResponseData, Citation, ImageCitation, TextCitation } from "@/types/agent-response";

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
  };
  isLoading?: boolean;
  // content remains string as per UIMessage
  // richResponse will hold the new structured data
  richResponse?: AgentResponseData;
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
  citations?: Citation[]; // Optional for now, but will be used for inline rendering
}

// Markdown content renderer component
const MarkdownContent: React.FC<MarkdownContentProps> = ({ content, citations }) => {
  const contentString = typeof content === 'string' ? content : JSON.stringify(content);
  // console.log("MarkdownContent received body:", contentString); // DEBUG: Log the raw body
  // console.log("MarkdownContent received citations:", citations); // DEBUG: Log the citations array

  const processNode = (node: React.ReactNode): React.ReactNode => {
    if (typeof node === 'string') {
      const parts: React.ReactNode[] = []; // Explicitly type parts
      let lastIndex = 0;
      const regex = /\[ref:([a-zA-Z0-9_-]+)\]/g;
      let match;

      while ((match = regex.exec(node)) !== null) {
        if (match.index > lastIndex) {
          parts.push(node.substring(lastIndex, match.index));
        }

        const citationId = match[1];
        // console.log(`Found placeholder for citation ID: ${citationId}`); // DEBUG
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
          } else if (citation.type === 'text') { // Text citation
            const textSnippet = (citation as TextCitation).snippet;
            if (textSnippet && textSnippet.startsWith("data:image")) {
              renderAsImage = true;
              imageUrlForRender = textSnippet; // The snippet itself is the data URI
              imageAltText = `Cited Image Data (Ref: ${citation.id})`;
              // console.log(`Citation ${citationId} is type text but snippet is an image. Rendering as image.`); // DEBUG
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
          } else { // Is definitely a text citation (and not an image data URI in snippet)
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
          // console.warn(`Citation ID ${citationId} found in text but not in citations array.`); // DEBUG
          parts.push(match[0]); // Keep original placeholder if citation not found
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
    // Assuming PreviewMessage is designed to take the whole message object for users
    // and does not require children from AgentPreviewMessage for user roles.
    // If PreviewMessage is just a styled bubble, it might take children,
    // but the error suggested it doesn't when uiMessage prop is used.
    // Let's assume it handles user message content internally or via a simpler prop.
    return <PreviewMessage message={message} />;
  }

  // Handle Assistant Messages
  if (message.role === 'assistant') {
    if (message.richResponse && message.richResponse.mode === 'rich') {
      const { body, citations } = message.richResponse;
      return (
        <div className="group relative flex px-4 py-3">
          <div className="flex w-full flex-col items-start">
            <div className="flex w-full max-w-3xl items-start gap-4">
              <div className="flex-1 space-y-2 overflow-hidden">
                <div className="rounded-xl bg-muted p-4">
                  <MarkdownContent content={body} citations={citations} />
                </div>

                {citations && citations.length > 0 && (
                  <div className="mt-4 space-y-3 rounded-xl border p-3">
                    <h4 className="text-sm font-semibold">Citations ({citations.length})</h4>
                    <div className="max-h-[400px] space-y-3 overflow-y-auto pr-2">
                      {citations.map((citation) => (
                        <CitationView key={citation.id} citation={citation} />
                      ))}
                    </div>
                  </div>
                )}

                {toolHistory && toolHistory.length > 0 && (
                   <Accordion type="single" collapsible className="mt-4 overflow-hidden rounded-xl border">
                     <AccordionItem value="tools" className="border-0">
                       <AccordionTrigger className="px-4 py-2 text-sm font-medium">
                         Tool Calls ({toolHistory.length})
                       </AccordionTrigger>
                       <AccordionContent className="px-4 pb-3">
                         <div className="max-h-[400px] space-y-3 overflow-y-auto pr-2">
                           {toolHistory.map((tool, index) => (
                             <div
                               key={`${tool.tool_name}-${index}`}
                               className="overflow-hidden rounded-md border bg-background"
                             >
                               <div className="border-b p-3">
                                 <div className="flex items-start justify-between">
                                   <div>
                                     <span className="text-sm font-medium">{tool.tool_name}</span>
                                   </div>
                                   <Badge variant="outline" className="text-[10px]">
                                     Tool Call #{index + 1}
                                   </Badge>
                                 </div>
                               </div>
                               <Accordion type="multiple" className="border-t">
                                 <AccordionItem value="args" className="border-0">
                                   <AccordionTrigger className="px-3 py-2 text-xs">Arguments</AccordionTrigger>
                                   <AccordionContent className="px-3 pb-3">{renderJson(tool.tool_args)}</AccordionContent>
                                 </AccordionItem>
                                 <AccordionItem value="result" className="border-t">
                                   <AccordionTrigger className="px-3 py-2 text-xs">Result</AccordionTrigger>
                                   <AccordionContent className="px-3 pb-3">{renderJson(tool.tool_result)}</AccordionContent>
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
    } else {
      // Plain assistant message (not rich, but may have message.content and/or toolHistory)
      return (
        <div className="group relative flex px-4 py-3">
          <div className="flex-shrink-0 mr-3">{/* Placeholder for an avatar or icon */}</div>
          <div className="flex-1 space-y-2 overflow-hidden">
            {message.content && (
              <div className="rounded-xl bg-muted p-4">
                <MarkdownContent content={message.content} citations={undefined} /> {/* No rich citations here */}
              </div>
            )}
            {toolHistory && toolHistory.length > 0 && (
              <Accordion type="single" collapsible className="mt-2 overflow-hidden rounded-xl border">
                <AccordionItem value="tools" className="border-0">
                  <AccordionTrigger className="px-4 py-2 text-sm font-medium">
                    Tool Calls ({toolHistory.length})
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-3">
                    <div className="max-h-[400px] space-y-3 overflow-y-auto pr-2">
                      {toolHistory.map((tool, index) => (
                        <div
                          key={`${tool.tool_name}-${index}`}
                          className="overflow-hidden rounded-md border bg-background"
                        >
                          <div className="border-b p-3">
                            <div className="flex items-start justify-between">
                              <div>
                                <span className="text-sm font-medium">{tool.tool_name}</span>
                              </div>
                              <Badge variant="outline" className="text-[10px]">
                                Tool Call #{index + 1}
                              </Badge>
                            </div>
                          </div>
                          <Accordion type="multiple" className="border-t">
                            <AccordionItem value="args" className="border-0">
                              <AccordionTrigger className="px-3 py-2 text-xs">Arguments</AccordionTrigger>
                              <AccordionContent className="px-3 pb-3">{renderJson(tool.tool_args)}</AccordionContent>
                            </AccordionItem>
                            <AccordionItem value="result" className="border-t">
                              <AccordionTrigger className="px-3 py-2 text-xs">Result</AccordionTrigger>
                              <AccordionContent className="px-3 pb-3">{renderJson(tool.tool_result)}</AccordionContent>
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
    }
  }

  // Fallback for any other unhandled message types or roles (should not happen with current types)
  return <PreviewMessage message={message} />;
}
