import React from "react";
import { useChatSessions } from "@/hooks/useChatSessions";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { RotateCw, Plus, ChevronsLeft, ChevronsRight } from "lucide-react";
// import { DisplayObject } from "./AgentChatMessages"; // Potentially for a more robust type

interface ChatSidebarProps {
  apiBaseUrl: string;
  authToken: string | null;
  onSelect: (chatId: string | undefined) => void;
  activeChatId?: string;
  collapsed: boolean;
  onToggle: () => void;
}

// Define types for message preview generation
interface DisplayObjectPreview {
  type: string;
  content?: string;
}

interface AgentDataPreview {
  display_objects?: DisplayObjectPreview[];
}

interface MessagePreviewContent {
  content?: string;
  agent_data?: AgentDataPreview;
  // Include other properties from session.lastMessage if necessary for context
}

// Function to generate a better preview for agent messages
const generateMessagePreview = (content: string, lastMessage?: MessagePreviewContent): string => {
  if (!content && !lastMessage?.agent_data?.display_objects) return "(no message)";
  if (!content && lastMessage?.agent_data?.display_objects) content = ""; // Ensure content is not null if we have display objects

  // Check if this is an agent message with agent_data
  if (lastMessage?.agent_data?.display_objects && Array.isArray(lastMessage.agent_data.display_objects)) {
    const displayObjects = lastMessage.agent_data.display_objects;

    // Find the first text display object
    const textObject = displayObjects.find((obj: DisplayObjectPreview) => obj.type === "text" && obj.content);

    if (textObject && textObject.content) {
      let textContent = textObject.content;
      // Remove markdown formatting for preview
      textContent = textContent.replace(/#{1,6}\s+/g, "");
      textContent = textContent.replace(/\*\*(.*?)\*\*/g, "$1");
      textContent = textContent.replace(/\*(.*?)\*/g, "$1");
      textContent = textContent.replace(/`(.*?)`/g, "$1");
      textContent = textContent.replace(/\n+/g, " ");
      return textContent.trim().slice(0, 35) || "Agent response (text)"; // ensure not empty string
    }

    // If no text objects, show a generic agent response message
    return "Agent response (media)"; // Differentiated for clarity
  }

  // For regular text messages, avoid showing raw JSON
  const trimmedContent = content.trim();
  if (trimmedContent.startsWith("[") || trimmedContent.startsWith("{")) {
    try {
      const parsed = JSON.parse(trimmedContent);

      if (Array.isArray(parsed)) {
        const textObjects = parsed.filter((obj: DisplayObjectPreview) => obj.type === "text" && obj.content);
        if (textObjects.length > 0 && textObjects[0].content) {
          let textContent = textObjects[0].content;
          textContent = textContent.replace(/#{1,6}\s+/g, "");
          textContent = textContent.replace(/\*\*(.*?)\*\*/g, "$1");
          textContent = textContent.replace(/\*(.*?)\*/g, "$1");
          textContent = textContent.replace(/`(.*?)`/g, "$1");
          textContent = textContent.replace(/\n+/g, " ");
          return textContent.trim().slice(0, 35) || "Agent response (parsed text)";
        }
        return "Agent response (parsed media)";
      }

      if (parsed.content && typeof parsed.content === "string") {
        return parsed.content.slice(0, 35) || "Agent response (parsed content)";
      }

      return "Agent response (JSON)";
    } catch (_e) {
      console.log("Error parsing JSON:", _e);
      // Prefixed 'e' with an underscore
      if (trimmedContent.length < 100 && !trimmedContent.includes('"type"')) {
        return content.slice(0, 35);
      }
      return "Agent response (error)";
    }
  }

  // for regular chat 
  content = content.replace(/#{1,6}\s+/g, "");
  content = content.replace(/\*\*(.*?)\*\*/g, "$1");
  content = content.replace(/\*(.*?)\*/g, "$1");
  content = content.replace(/`(.*?)`/g, "$1");
  content = content.replace(/\n+/g, " ");
  return content.trim().slice(0, 35) || "chat response (text)";
};

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  apiBaseUrl,
  authToken,
  onSelect,
  activeChatId,
  collapsed,
  onToggle,
}) => {
  const { sessions, isLoading, reload } = useChatSessions({ apiBaseUrl, authToken });

  if (collapsed) {
    return (
      <div className="flex w-8 flex-col items-center border-r bg-muted/40">
        <Button variant="ghost" size="icon" className="mt-2" onClick={onToggle} title="Expand">
          <ChevronsRight className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex w-60 flex-col border-r bg-muted/40">
      <div className="flex h-12 items-center justify-between px-3 text-xs font-medium">
        <span className="text-base">Conversations</span>
        <div className="flex items-center justify-center">
          <Button variant="ghost" size="icon" onClick={() => onSelect(undefined)} title="New chat">
            <Plus className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={() => reload()} title="Refresh chats">
            <RotateCw className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={onToggle} title="Collapse sidebar">
            <ChevronsLeft className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <ul className="p-1">
          {isLoading && <li className="py-1 text-xs text-center">Loading…</li>}
          {!isLoading && sessions.length === 0 && (
            <li className="px-2 py-1 text-xs text-muted-foreground text-center">No chats yet</li>
          )}
          {sessions.map((session) => (
            <li key={session.chatId} className="mb-1">
              <button
                onClick={() => onSelect(session.chatId)}
                className={cn(
                  "w-full rounded px-2 py-1 text-left text-sm hover:bg-accent/60",
                  activeChatId === session.chatId && "bg-accent text-accent-foreground"
                )}
              >
                <div className="truncate">
                  {generateMessagePreview(
                    session.lastMessage?.content || "",
                    session.lastMessage === null ? undefined : session.lastMessage
                  )}
                </div>
                <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
                  {new Date(session.updatedAt || session.createdAt || Date.now()).toLocaleString()}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </ScrollArea>
    </div>
  );
};
