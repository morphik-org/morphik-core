"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { FileText, Search, MessageSquare, ChevronLeft, ChevronRight } from "lucide-react"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {
  activeSection: string
  onSectionChange: (section: string) => void
}

export function Sidebar({ className, activeSection, onSectionChange, ...props }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = React.useState(false)

  return (
    <div
      className={cn(
        "relative flex flex-col border-r bg-background transition-all duration-300",
        isCollapsed ? "w-16" : "w-64",
        className
      )}
      {...props}
    >
      <div className="flex items-center justify-between p-4 border-b">
        {!isCollapsed && <h2 className="text-lg font-semibold">DataBridge</h2>}
        <Button
          variant="ghost"
          size="icon"
          className="ml-auto"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="space-y-1 p-2">
          <Button
            variant={activeSection === "documents" ? "secondary" : "ghost"}
            className={cn(
              "w-full justify-start",
              isCollapsed && "justify-center"
            )}
            onClick={() => onSectionChange("documents")}
          >
            <FileText className="h-4 w-4" />
            {!isCollapsed && <span className="ml-2">Documents</span>}
          </Button>
          
          <Button
            variant={activeSection === "search" ? "secondary" : "ghost"}
            className={cn(
              "w-full justify-start",
              isCollapsed && "justify-center"
            )}
            onClick={() => onSectionChange("search")}
          >
            <Search className="h-4 w-4" />
            {!isCollapsed && <span className="ml-2">Search</span>}
          </Button>
          
          <Button
            variant={activeSection === "chat" ? "secondary" : "ghost"}
            className={cn(
              "w-full justify-start",
              isCollapsed && "justify-center"
            )}
            onClick={() => onSectionChange("chat")}
          >
            <MessageSquare className="h-4 w-4" />
            {!isCollapsed && <span className="ml-2">Chat</span>}
          </Button>
        </div>
      </ScrollArea>
    </div>
  )
} 