import React from "react";
import { Button } from "@/components/ui/button";
import { Key, Bot, ChevronLeft } from "lucide-react";

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
      <div className="flex h-full w-10 flex-col items-center border-r bg-muted/40">
        <Button variant="ghost" size="icon" className="mt-2" onClick={onToggle} title="Expand">
          {/* Add expand icon here if needed */}
        </Button>
      </div>
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
