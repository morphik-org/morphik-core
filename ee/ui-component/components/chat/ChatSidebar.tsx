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
                <div className="truncate">{s.lastMessage?.content?.slice(0, 30) || "(no message)"}</div>
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
