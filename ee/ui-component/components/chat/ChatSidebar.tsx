import React from "react";
import { useChatSessions } from "@/hooks/useChatSessions";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { RotateCw, Plus, ChevronsLeft, ChevronsRight } from "lucide-react";

interface ChatSidebarProps {
  apiBaseUrl: string;
  authToken: string | null;
  onSelect: (chatId: string | undefined) => void;
  activeChatId?: string;
  collapsed: boolean;
  onToggle: () => void;
}

// Function to generate a better preview for agent messages
const generateMessagePreview = (content: string, lastMessage?: any): string => {
  if (!content) return "(no message)";

  // Check if this is an agent message with agent_data
  if (lastMessage?.agent_data?.display_objects && Array.isArray(lastMessage.agent_data.display_objects)) {
    const displayObjects = lastMessage.agent_data.display_objects;

    // Find the first text display object
    const textObject = displayObjects.find((obj: any) => obj.type === "text" && obj.content);

    if (textObject) {
      let textContent = textObject.content;
      // Remove markdown formatting for preview
      textContent = textContent.replace(/#{1,6}\s+/g, ''); // Remove headers
      textContent = textContent.replace(/\*\*(.*?)\*\*/g, '$1'); // Remove bold
      textContent = textContent.replace(/\*(.*?)\*/g, '$1'); // Remove italic
      textContent = textContent.replace(/`(.*?)`/g, '$1'); // Remove code
      textContent = textContent.replace(/\n+/g, ' '); // Replace newlines with spaces
      return textContent.trim().slice(0, 50);
    }

    // If no text objects, show a generic agent response message
    return "Agent response";
  }

  // For regular text messages, avoid showing raw JSON
  // First check if the content looks like it might be JSON
  const trimmedContent = content.trim();
  if (trimmedContent.startsWith('[') || trimmedContent.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmedContent);

      // If it's an array of display objects, extract text content
      if (Array.isArray(parsed)) {
        const textObjects = parsed.filter((obj: any) => obj.type === "text" && obj.content);
        if (textObjects.length > 0) {
          // Get the first text object's content and clean it up
          let textContent = textObjects[0].content;
          // Remove markdown formatting for preview
          textContent = textContent.replace(/#{1,6}\s+/g, ''); // Remove headers
          textContent = textContent.replace(/\*\*(.*?)\*\*/g, '$1'); // Remove bold
          textContent = textContent.replace(/\*(.*?)\*/g, '$1'); // Remove italic
          textContent = textContent.replace(/`(.*?)`/g, '$1'); // Remove code
          textContent = textContent.replace(/\n+/g, ' '); // Replace newlines with spaces
          return textContent.trim().slice(0, 50);
        }

        // If it's an array but no text objects, it's likely display objects
        return "Agent response with media";
      }

      // If it's a single object with content
      if (parsed.content && typeof parsed.content === 'string') {
        return parsed.content.slice(0, 50);
      }

      // If it's any other JSON structure, show generic message
      return "Agent response";
    } catch (e) {
      // If JSON parsing fails, it might be a normal message that just starts with [ or {
      // Only show first 50 chars if it doesn't look like a full JSON structure
      if (trimmedContent.length < 100 && !trimmedContent.includes('"type"')) {
        return content.slice(0, 50);
      }
      // Otherwise, it's probably malformed JSON from an agent response
      return "Agent response";
    }
  }

  // For regular text messages
  return content.slice(0, 50);
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
        <span>Conversations</span>
        <div className="flex items-center gap-1">
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
          {isLoading && <li className="px-2 py-1 text-xs">Loading…</li>}
          {!isLoading && sessions.length === 0 && (
            <li className="px-2 py-1 text-xs text-muted-foreground">No chats yet</li>
          )}
          {sessions.map(s => (
            <li key={s.chatId} className="mb-1">
              <button
                onClick={() => onSelect(s.chatId)}
                className={cn(
                  "w-full rounded px-2 py-1 text-left text-sm hover:bg-accent/60",
                  activeChatId === s.chatId && "bg-accent text-accent-foreground"
                )}
              >
                <div className="truncate">{generateMessagePreview(s.lastMessage?.content || "", s.lastMessage)}</div>
                <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
                  {new Date(s.updatedAt || s.createdAt || Date.now()).toLocaleString()}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </ScrollArea>
    </div>
  );
};
