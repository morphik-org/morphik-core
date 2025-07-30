import React, { useState } from "react";
import { useChatSessions } from "@/hooks/useChatSessions";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { RotateCw, Plus, ChevronsLeft, ChevronsRight, Search, MoreVertical, Edit3, Check, X } from "lucide-react";
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

export const ChatSidebar: React.FC<ChatSidebarProps> = React.memo(function ChatSidebar({
  apiBaseUrl,
  authToken,
  onSelect,
  activeChatId,
  collapsed,
  onToggle,
}) {
  const { sessions, isLoading, reload } = useChatSessions({ apiBaseUrl, authToken });
  const [searchQuery, setSearchQuery] = useState("");
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [openDropdownId, setOpenDropdownId] = useState<string | null>(null);

  // Filter sessions based on search query and ensure they're sorted by most recent
  const filteredSessions = sessions
    .filter(session => {
      const title =
        session.title ||
        generateMessagePreview(
          session.lastMessage?.content || "",
          session.lastMessage === null ? undefined : session.lastMessage
        );
      return title.toLowerCase().includes(searchQuery.toLowerCase());
    })
    .sort((a, b) => {
      // Sort by updatedAt in descending order (most recent first)
      const dateA = new Date(a.updatedAt || a.createdAt || 0).getTime();
      const dateB = new Date(b.updatedAt || b.createdAt || 0).getTime();
      return dateB - dateA;
    });

  const handleEditTitle = async (chatId: string, newTitle: string) => {
    try {
      const response = await fetch(`${apiBaseUrl}/chats/${chatId}/title?title=${encodeURIComponent(newTitle)}`, {
        method: "PATCH",
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });
      if (response.ok) {
        reload();
        setEditingChatId(null);
      }
    } catch (error) {
      console.error("Failed to update chat title:", error);
    }
  };

  if (collapsed) {
    return (
      <div className="flex h-full w-10 flex-col items-center border-r bg-muted/40">
        <Button variant="ghost" size="icon" className="mt-2" onClick={onToggle} title="Expand">
          <ChevronsRight className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-full w-80 flex-col border-r bg-muted/40">
      <div className="flex h-12 items-center justify-between px-3 text-xs font-medium">
        <span className="text-sm text-muted-foreground">Chats</span>
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
      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="h-8 pl-8 text-sm"
          />
        </div>
      </div>
      <ScrollArea className="flex-1">
        <ul className="p-1">
          {isLoading ? (
            <>
              {/* Skeleton loading items */}
              {Array.from({ length: 5 }).map((_, i) => (
                <li key={i} className="mb-1">
                  <div className="animate-pulse rounded px-2 py-1">
                    <div className="mb-1 h-4 w-3/4 rounded bg-muted"></div>
                    <div className="h-3 w-1/2 rounded bg-muted/60"></div>
                  </div>
                </li>
              ))}
            </>
          ) : sessions.length === 0 ? (
            <li className="px-2 py-1 text-center text-xs text-muted-foreground">No chats yet</li>
          ) : (
            filteredSessions.map(session => {
              const fullTitle =
                session.title ||
                generateMessagePreview(
                  session.lastMessage?.content || "",
                  session.lastMessage === null ? undefined : session.lastMessage
                );

              const displayTitle = fullTitle.length > 30 ? fullTitle.slice(0, 30) + "..." : fullTitle;

              return (
                <li key={session.chatId} className="mb-1">
                  <div className="relative flex items-center px-1">
                    <button
                      onClick={() => onSelect(session.chatId)}
                      className={cn(
                        "group flex-1 rounded px-2 py-1 text-left text-sm hover:bg-accent/60",
                        activeChatId === session.chatId && "bg-accent text-accent-foreground"
                      )}
                    >
                      {editingChatId === session.chatId ? (
                        <div className="flex items-center gap-1">
                          <Input
                            type="text"
                            value={editingTitle}
                            onChange={e => setEditingTitle(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === "Enter") {
                                handleEditTitle(session.chatId, editingTitle);
                              } else if (e.key === "Escape") {
                                setEditingChatId(null);
                              }
                            }}
                            onClick={e => e.stopPropagation()}
                            className="h-6 flex-1 px-1 text-sm"
                            autoFocus
                          />
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={e => {
                              e.stopPropagation();
                              handleEditTitle(session.chatId, editingTitle);
                            }}
                          >
                            <Check className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={e => {
                              e.stopPropagation();
                              setEditingChatId(null);
                            }}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      ) : (
                        <div className="truncate">{displayTitle}</div>
                      )}
                    </button>

                    {/* Dropdown menu - fixed position aligned with collapse button, with background */}
                    {editingChatId !== session.chatId && (
                      <div className="absolute right-3 top-1/2 -translate-y-1/2 opacity-0 transition-opacity group-hover:opacity-100">
                        <DropdownMenu
                          open={openDropdownId === session.chatId}
                          onOpenChange={open => {
                            setOpenDropdownId(open ? session.chatId : null);
                          }}
                        >
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 shrink-0 border border-border/50 bg-background/80 backdrop-blur-sm hover:bg-accent/60"
                              onClick={e => {
                                e.stopPropagation();
                              }}
                            >
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-48">
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation();
                                setEditingChatId(session.chatId);
                                setEditingTitle(fullTitle);
                                setOpenDropdownId(null);
                              }}
                            >
                              <Edit3 className="mr-2 h-4 w-4" />
                              Edit title
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    )}
                  </div>
                </li>
              );
            })
          )}
        </ul>
      </ScrollArea>
    </div>
  );
});
