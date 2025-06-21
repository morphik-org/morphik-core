import React, { useState, useMemo } from "react";
import { useChatSessions } from "@/hooks/useChatSessions";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { RotateCw, Plus, ChevronsLeft, ChevronsRight, MoreVertical, Pencil, Trash2, Search, X } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  const [searchQuery, setSearchQuery] = useState("");
  const { sessions, isLoading, reload, renameChat, deleteChat, searchChats } = useChatSessions({ 
    apiBaseUrl, 
    authToken,
    searchQuery 
  });
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [newChatName, setNewChatName] = useState("");

  // Local filtering for immediate response
  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions;
    
    const query = searchQuery.toLowerCase();
    return sessions.filter(session => {
      // Search in chat name (highest priority)
      if (session.name?.toLowerCase().includes(query)) return true;
      
      // Search in last message content
      if (session.lastMessage?.content?.toLowerCase().includes(query)) return true;
      
      // Search in agent data if available
      if (session.lastMessage?.agent_data?.display_objects) {
        const hasMatch = session.lastMessage.agent_data.display_objects.some(obj => 
          obj.type === "text" && obj.content?.toLowerCase().includes(query)
        );
        if (hasMatch) return true;
      }
      
      // Fallback to message preview search
      const preview = session.name || generateMessagePreview(
        session.lastMessage?.content || "",
        session.lastMessage === null ? undefined : session.lastMessage
      );
      return preview.toLowerCase().includes(query);
    });
  }, [sessions, searchQuery]);

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
  };

  const handleClearSearch = () => {
    setSearchQuery("");
  };

  const handleRenameClick = (chatId: string, currentName: string | null | undefined) => {
    console.log("Rename clicked for:", chatId, currentName);
    setSelectedChatId(chatId);
    setNewChatName(currentName || "");
    setRenameDialogOpen(true);
  };

  const handleDeleteClick = (chatId: string) => {
    console.log("Delete clicked for:", chatId);
    setSelectedChatId(chatId);
    setDeleteDialogOpen(true);
  };

  const handleRenameConfirm = async () => {
    if (selectedChatId && newChatName.trim()) {
      const success = await renameChat(selectedChatId, newChatName.trim());
      if (success) {
        setRenameDialogOpen(false);
        setSelectedChatId(null);
        setNewChatName("");
      }
    }
  };

  const handleDeleteConfirm = async () => {
    if (selectedChatId) {
      const success = await deleteChat(selectedChatId);
      if (success) {
        setDeleteDialogOpen(false);
        setSelectedChatId(null);
        // If we deleted the active chat, clear the selection
        if (selectedChatId === activeChatId) {
          onSelect(undefined);
        }
      }
    }
  };

  if (collapsed) {
    return (
      <div className="flex w-10 flex-col items-center border-r bg-muted/40">
        <Button variant="ghost" size="icon" className="mt-2" onClick={onToggle} title="Expand">
          <ChevronsRight className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex w-72 flex-col border-r bg-muted/40">
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
      
      {/* Search Input */}
      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search"
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="pl-9 pr-9 h-8 bg-background border-input"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-1 top-1/2 h-6 w-6 -translate-y-1/2 hover:bg-muted"
              onClick={handleClearSearch}
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        <ul className="p-1">
          {isLoading && <li className="py-1 text-center text-xs text-muted-foreground">Loading…</li>}
          {!isLoading && sessions.length === 0 && (
            <li className="px-2 py-1 text-center text-xs text-muted-foreground">No chats yet</li>
          )}
          {!isLoading && searchQuery && filteredSessions.length === 0 && (
            <li className="px-2 py-1 text-center text-xs text-muted-foreground">No chats found</li>
          )}
          {filteredSessions.map(session => (
            <li 
              key={session.chatId} 
              className="mb-1 group"
            >
              <div className={cn(
                "flex items-center gap-1 rounded hover:bg-accent/60",
                activeChatId === session.chatId && "bg-accent text-accent-foreground"
              )}>
                <button
                  onClick={() => onSelect(session.chatId)}
                  className="flex-1 px-2 py-1 text-left text-sm min-w-0"
                >
                  <div className="truncate">
                    {session.name || generateMessagePreview(
                      session.lastMessage?.content || "",
                      session.lastMessage === null ? undefined : session.lastMessage
                    )}
                  </div>
                  <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
                    {new Date(session.updatedAt || session.createdAt || Date.now()).toLocaleString()}
                  </div>
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger className="h-7 w-7 mr-1 rounded-md flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-accent flex-shrink-0">
                    <MoreVertical className="h-3 w-3" />
                  </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleRenameClick(session.chatId, session.name)}>
                    <Pencil className="mr-2 h-3 w-3" />
                    Rename
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => handleDeleteClick(session.chatId)}
                    className="text-destructive"
                  >
                    <Trash2 className="mr-2 h-3 w-3" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              </div>
            </li>
          ))}
        </ul>
      </ScrollArea>

      {/* Rename Dialog */}
      <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Chat</DialogTitle>
            <DialogDescription>
              Enter a new name for this chat conversation.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input
                id="name"
                value={newChatName}
                onChange={(e) => setNewChatName(e.target.value)}
                className="col-span-3"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleRenameConfirm();
                  }
                }}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleRenameConfirm}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Chat</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this chat? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDeleteConfirm}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
