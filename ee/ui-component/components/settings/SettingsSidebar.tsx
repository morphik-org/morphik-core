import React from "react";
import { Button } from "@/components/ui/button";
import { Key, Bot, ChevronLeft } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface SettingsSidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  onBackClick?: () => void;
  collapsed: boolean;
  onToggle: () => void;
}

export const SettingsSidebar: React.FC<SettingsSidebarProps> = ({
  activeTab,
  onTabChange,
  onBackClick,
  collapsed,
  onToggle,
}) => {
  if (collapsed) {
    return (
      <TooltipProvider>
        <div className="flex h-full w-10 flex-col items-center border-r bg-muted/40 py-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                aria-label="API Keys"
                className="mt-1 rounded p-1 hover:bg-accent/40"
                onClick={() => {
                  onTabChange("api-keys");
                  onToggle();
                }}
              >
                <Key className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">API Keys</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                aria-label="Custom Models"
                className="mt-2 rounded p-1 hover:bg-accent/40"
                onClick={() => {
                  onTabChange("models");
                  onToggle();
                }}
              >
                <Bot className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">Custom Models</TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>
    );
  }

  return (
    <div className="flex h-full w-80 flex-col border-r bg-muted/40">
      <div className="flex h-12 items-center justify-between px-3 text-xs font-medium">
        <span className="text-sm text-muted-foreground">Settings</span>
        {onBackClick && (
          <Button variant="ghost" size="sm" onClick={onBackClick} className="w-full justify-start">
            <ChevronLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
        )}
      </div>

      <div className="p-4">
        <nav className="mt-4 space-y-2">
          <button
            onClick={() => onTabChange("api-keys")}
            className={`flex w-full items-center gap-2 rounded-lg px-3 py-3 text-sm font-medium transition-colors ${
              activeTab === "api-keys" ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
            }`}
          >
            <Key className="h-4 w-4" />
            API Keys
          </button>
          <button
            onClick={() => onTabChange("models")}
            className={`flex w-full items-center gap-2 rounded-lg px-3 py-3 text-sm font-medium transition-colors ${
              activeTab === "models" ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
            }`}
          >
            <Bot className="h-4 w-4" />
            Custom Models
          </button>
        </nav>
      </div>
    </div>
  );
};
