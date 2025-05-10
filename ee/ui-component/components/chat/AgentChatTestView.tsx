"use client";

import React from "react";
import { AgentPreviewMessage, AgentUIMessage, DisplayObject, SourceObject, ToolCall } from "./AgentChatMessages";
import { generateUUID } from "@/lib/utils";

// Sample base64 image (small red square)
const sampleBase64Image = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==";

// Sample tool call
const sampleToolCall: ToolCall = {
  tool_name: "web_search",
  tool_args: { query: "sample search query" },
  tool_result: { results: ["Sample result 1", "Sample result 2"] }
};

// Sample display objects
const sampleDisplayObjects: DisplayObject[] = [
  {
    type: 'text',
    content: "This is a **markdown** text display object with _formatting_.",
    source: "source-123"
  },
  {
    type: 'image',
    content: sampleBase64Image,
    caption: "A sample image caption",
    source: "source-456"
  }
];

// Sample sources
const sampleSources: SourceObject[] = [
  {
    sourceId: "source-123",
    documentName: "Sample Document 1",
    documentId: "doc-abc"
  },
  {
    sourceId: "source-456",
    documentName: "Sample Image Document",
    documentId: "doc-xyz"
  }
];

// Create sample messages
const createSampleMessages = (): AgentUIMessage[] => {
  return [
    // User message
    {
      id: generateUUID(),
      role: "user",
      content: "Show me information about data structures",
      createdAt: new Date(Date.now() - 60000)
    },
    // Assistant message with display objects and sources
    {
      id: generateUUID(),
      role: "assistant",
      content: "Here's information about data structures. I've included both text and visual examples.",
      createdAt: new Date(),
      experimental_agentData: {
        tool_history: [sampleToolCall],
        displayObjects: sampleDisplayObjects,
        sources: sampleSources
      }
    }
  ];
};

const AgentChatTestView: React.FC = () => {
  const sampleMessages = createSampleMessages();

  return (
    <div className="mx-auto max-w-4xl p-4">
      <h1 className="mb-6 text-xl font-bold">Agent Chat Test View</h1>
      <div className="rounded-lg border p-4">
        {sampleMessages.map(message => (
          <AgentPreviewMessage key={message.id} message={message} />
        ))}
      </div>
    </div>
  );
};

export default AgentChatTestView;
