import React, { useEffect, useState, memo } from "react";
import { PreviewMessage, UIMessage } from "./ChatMessages";

// Base interface for display objects
export interface BaseDisplayObject {
  source: string; // Source ID that links to the source
}

// Text display object interface
export interface TextDisplayObject extends BaseDisplayObject {
  type: "text";
  content: string; // Markdown content
}

// Image display object interface
export interface ImageDisplayObject extends BaseDisplayObject {
  type: "image";
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
  contentType?: "text" | "image"; // Type of content
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

const ThinkingMessage = memo(function ThinkingMessage() {
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
});

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

  // For assistant messages, always use PreviewMessage for consistency
  // Convert agent data to regular message format if needed
  if (message.role === "assistant") {
    // If we have display objects, combine them into the content
    let combinedContent = message.content;

    if (displayObjects && displayObjects.length > 0) {
      const textContent = displayObjects
        .filter(obj => obj.type === "text")
        .map(obj => obj.content)
        .join("\n\n");

      // If we have additional text content from display objects, append it
      if (textContent && textContent !== message.content) {
        combinedContent = message.content ? `${message.content}\n\n${textContent}` : textContent;
      }
    }

    // Convert agent sources to regular sources format for consistency
    const convertedSources = sources?.map(source => ({
      document_id: source.documentId,
      filename: source.documentName,
      content: source.content || "",
      content_type: source.contentType === "image" ? "image/png" : "text/plain",
      chunk_number: 1, // Default since agent sources don't have chunk numbers
      score: undefined,
      metadata: {},
    }));

    // Create a modified message with converted data
    const modifiedMessage = {
      ...message,
      content: combinedContent,
      experimental_customData: convertedSources?.length ? { sources: convertedSources } : undefined,
    };

    return <PreviewMessage message={modifiedMessage} />;
  }

  // Fallback to standard message
  return <PreviewMessage message={message} />;
}
