/*
 * MorphikSidebarRemote - REMOTE UI VERSION
 *
 * This sidebar is used by the cloud UI (morphik-cloud-ui) via:
 * TrackedMorphikUI → MorphikUI → MorphikSidebarStateful
 *
 * Features:
 * - Section-based navigation (onSectionChange callbacks)
 * - Single-page app experience with dynamic content switching
 * - No URI editing capability (cloud manages connections)
 * - Used in prod
 *
 * DO NOT add URI editing functionality here - it will affect cloud users!
 * For local development URI editing, see morphik-sidebar.tsx instead.
 */
"use client";

import * as React from "react";
import { BaseSidebar } from "@/components/sidebar-base";
import { createSectionNavigation } from "@/lib/navigation-utils";

interface MorphikSidebarStatefulProps {
  currentSection: string;
  onSectionChange: (section: string) => void;
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
  logoLight?: string;
  logoDark?: string;
  showChatView?: boolean;
  onChatViewChange?: (show: boolean) => void;
  activeChatId?: string;
  onChatSelect?: (id: string | undefined) => void;
  showSettingsView?: boolean;
  onSettingsViewChange?: (show: boolean) => void;
  activeSettingsTab?: string;
  onSettingsTabChange?: (tab: string) => void;
}

export function MorphikSidebarRemote({
  currentSection,
  onSectionChange,
  userProfile,
  onLogout,
  onProfileNavigate,
  onUpgradeClick,
  logoLight,
  logoDark,
  showChatView = false,
  onChatViewChange,
  activeChatId,
  onChatSelect,
  showSettingsView = false,
  onSettingsViewChange,
  activeSettingsTab = "api-keys",
  onSettingsTabChange,
}: MorphikSidebarStatefulProps) {
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

  const navigation = createSectionNavigation(onSectionChange, handleChatClick, currentSection, handleSettingsClick);

  const logoClickHandler = React.useCallback(() => {
    // In cloud UI, navigate to dashboard when logo is clicked
    if (typeof window !== "undefined" && window.location.pathname.includes("/")) {
      const pathSegments = window.location.pathname.split("/").filter(Boolean);
      // If we're in an app context (e.g., /app_id/documents), go to dashboard
      if (pathSegments.length > 0 && !["login", "signup", "dashboard"].includes(pathSegments[0])) {
        window.location.href = "/organization/applications";
        return;
      }
    }
    // Otherwise, go to documents
    onSectionChange("documents");
  }, [onSectionChange]);

  return (
    <BaseSidebar
      userProfile={userProfile}
      onLogout={onLogout}
      onProfileNavigate={onProfileNavigate}
      onUpgradeClick={onUpgradeClick}
      logoLight={logoLight}
      logoDark={logoDark}
      showChatView={showChatView}
      onChatViewChange={onChatViewChange}
      activeChatId={activeChatId}
      onChatSelect={onChatSelect}
      showSettingsView={showSettingsView}
      onSettingsViewChange={onSettingsViewChange}
      activeSettingsTab={activeSettingsTab}
      onSettingsTabChange={onSettingsTabChange}
      navigation={navigation}
      logoClickHandler={logoClickHandler}
      collapsible="icon"
    />
  );
}
