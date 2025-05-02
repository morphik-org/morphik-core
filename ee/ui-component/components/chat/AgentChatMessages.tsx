import React, { useEffect, useState } from 'react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { PreviewMessage, UIMessage } from './ChatMessages';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

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
  const [dots, setDots] = useState('');

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
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
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
          {currentPhrase.text}{dots}
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
    <pre className="bg-muted p-4 rounded-md whitespace-pre-wrap font-mono text-sm overflow-auto max-h-[300px]">
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
              <table className="w-full border-collapse border border-border">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-muted">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-border">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="divide-x divide-border">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="p-3 text-left font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="p-3">
              {children}
            </td>
          ),
          h1: ({ children }) => <h1 className="text-2xl font-bold mb-4">{children}</h1>,
          h2: ({ children }) => <h2 className="text-xl font-bold mb-3">{children}</h2>,
          h3: ({ children }) => <h3 className="text-lg font-bold mb-2">{children}</h3>,
          p: ({ children }) => <p className="mb-4 leading-relaxed">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-6 mb-4 space-y-2">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-6 mb-4 space-y-2">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic my-4">
              {children}
            </blockquote>
          ),
          code({ className, children }) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            const isInline = !className;

            if (!isInline && language) {
              return (
                <div className="rounded-md overflow-hidden my-4">
                  <SyntaxHighlighter
                    style={oneDark}
                    language={language}
                    PreTag="div"
                    className="!my-0"
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              );
            }

            return isInline ? (
              <code className="bg-muted px-1.5 py-0.5 rounded text-sm">{children}</code>
            ) : (
              <div className="rounded-md overflow-hidden my-4">
                <SyntaxHighlighter
                  style={oneDark}
                  language="text"
                  PreTag="div"
                  className="!my-0"
                >
                  {String(children).replace(/\n$/, '')}
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

export function AgentPreviewMessage({ message }: AgentMessageProps) {
  const toolHistory = message.experimental_agentData?.tool_history;

  // If this is a loading state, show the thinking message
  if (message.isLoading) {
    return <ThinkingMessage />;
  }

  // If no tool history, render regular message
  if (!toolHistory || toolHistory.length === 0) {
    return <PreviewMessage message={message} />;
  }

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
              {message.role === 'assistant' ? (
                <MarkdownContent content={message.content} />
              ) : (
                <div className="prose prose-sm dark:prose-invert break-words">
                  {message.content}
                </div>
              )}
            </div>

            {message.role === 'assistant' && toolHistory.length > 0 && (
              <Accordion type="single" collapsible className="mt-2 border rounded-xl overflow-hidden">
                <AccordionItem value="tools" className="border-0">
                  <AccordionTrigger className="px-4 py-2 text-sm font-medium">
                    Tool Calls ({toolHistory.length})
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-3">
                    <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                      {toolHistory.map((tool, index) => (
                        <div key={`${tool.tool_name}-${index}`} className="bg-background rounded-md border overflow-hidden">
                          <div className="p-3 border-b">
                            <div className="flex justify-between items-start">
                              <div>
                                <span className="font-medium text-sm">
                                  {tool.tool_name}
                                </span>
                              </div>
                              <Badge variant="outline" className="text-[10px]">
                                Tool Call #{index + 1}
                              </Badge>
                            </div>
                          </div>

                          <Accordion type="multiple" className="border-t">
                            <AccordionItem value="args" className="border-0">
                              <AccordionTrigger className="px-3 py-2 text-xs">
                                Arguments
                              </AccordionTrigger>
                              <AccordionContent className="px-3 pb-3">
                                {renderJson(tool.tool_args)}
                              </AccordionContent>
                            </AccordionItem>

                            <AccordionItem value="result" className="border-t">
                              <AccordionTrigger className="px-3 py-2 text-xs">
                                Result
                              </AccordionTrigger>
                              <AccordionContent className="px-3 pb-3">
                                {renderJson(tool.tool_result)}
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
