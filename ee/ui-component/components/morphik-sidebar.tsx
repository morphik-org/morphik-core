/*
 * MorphikSidebar - LOCAL DEVELOPMENT VERSION
 *
 * This sidebar is used by the local ui-component dev server via:
 * layout.tsx → ConnectedSidebar → MorphikSidebar
 *
 * Features:
 * - URL-based navigation (traditional Next.js routing)
 * - URI editing capability for local development
 * - localStorage persistence for connection settings
 * - Used for standalone development and testing
 *
 * Safe to modify for local dev features - does NOT affect cloud UI!
 * The cloud UI uses morphik-sidebar-stateful.tsx instead.
 */
"use client";

import * as React from "react";
import { BaseSidebar } from "@/components/base-sidebar";
import { createUrlNavigation } from "@/lib/navigation-utils";
import { normalizeToMorphikUri } from "@/lib/utils";

interface MorphikSidebarProps {
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
  showEditableUri?: boolean;
  connectionUri?: string | null;
  onUriChange?: (newUri: string) => void;
  showChatView?: boolean;
  onChatViewChange?: (show: boolean) => void;
  activeChatId?: string;
  onChatSelect?: (id: string | undefined) => void;
  showSettingsView?: boolean;
  onSettingsViewChange?: (show: boolean) => void;
  activeSettingsTab?: string;
  onSettingsTabChange?: (tab: string) => void;
}

export function MorphikSidebar({
  userProfile,
  onLogout,
  onProfileNavigate,
  onUpgradeClick,
  showEditableUri = true,
  connectionUri,
  onUriChange,
  showChatView = false,
  onChatViewChange,
  activeChatId,
  onChatSelect,
  showSettingsView = false,
  onSettingsViewChange,
  activeSettingsTab = "api-keys",
  onSettingsTabChange,
}: MorphikSidebarProps) {
  const handleChatClick = React.useCallback(() => {
    if (typeof window !== "undefined") {
      sessionStorage.removeItem("chatViewManuallyHidden");
    }
    onChatViewChange?.(true);
  }, [onChatViewChange]);

  const handleSettingsClick = React.useCallback(() => {
    if (typeof window !== "undefined") {
      sessionStorage.removeItem("settingsViewManuallyHidden");
    }
    onSettingsViewChange?.(true);
  }, [onSettingsViewChange]);

  const navigation = createUrlNavigation(handleChatClick, handleSettingsClick);

  const handleUriChange = React.useCallback(
    (newUri: string) => {
      if (onUriChange) {
        const normalizedUri = normalizeToMorphikUri(newUri);
        onUriChange(normalizedUri);
      }
    },
    [onUriChange]
  );

  return (
    <BaseSidebar
      userProfile={userProfile}
      onLogout={onLogout}
      onProfileNavigate={onProfileNavigate}
      onUpgradeClick={onUpgradeClick}
      showChatView={showChatView}
      onChatViewChange={onChatViewChange}
      activeChatId={activeChatId}
      onChatSelect={onChatSelect}
      showSettingsView={showSettingsView}
      onSettingsViewChange={onSettingsViewChange}
      activeSettingsTab={activeSettingsTab}
      onSettingsTabChange={onSettingsTabChange}
      navigation={navigation}
      showEditableUri={showEditableUri}
      connectionUri={connectionUri}
      onUriChange={handleUriChange}
      collapsible="icon"
    />
  );
}
