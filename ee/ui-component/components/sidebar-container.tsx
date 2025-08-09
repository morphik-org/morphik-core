"use client";

import React from "react";
import { useMorphik } from "@/contexts/morphik-context";
import { MorphikSidebar } from "@/components/sidebar";
import { usePathname } from "next/navigation";
import { useChatContext } from "@/components/chat/chat-context";

// Chat context moved to components/chat/chat-context.tsx

export function SidebarContainer() {
  const { connectionUri, updateConnectionUri, userProfile, onLogout, onProfileNavigate, onUpgradeClick } = useMorphik();
  const pathname = usePathname();
  const {
    showChatView,
    setShowChatView,
    activeChatId,
    setActiveChatId,
    showSettingsView,
    setShowSettingsView,
    activeSettingsTab,
    setActiveSettingsTab,
  } = useChatContext();

  // Ensure chat view is shown when on chat page
  React.useEffect(() => {
    if (pathname === "/chat") {
      // Clear any stale "manually hidden" state when navigating to chat from other pages
      // Only respect the manually hidden state if we're already on the chat page
      const wasOnChatPage = sessionStorage.getItem("lastPage") === "/chat";
      const hasManuallyHidden = sessionStorage.getItem("chatViewManuallyHidden") === "true";

      if (!wasOnChatPage) {
        // Coming from another page - always show chat view
        sessionStorage.removeItem("chatViewManuallyHidden");
        setShowChatView(true);
      } else if (!hasManuallyHidden && !showChatView) {
        // Already on chat page and not manually hidden - show chat view
        setShowChatView(true);
      }

      // Track that we're on chat page
      sessionStorage.setItem("lastPage", "/chat");
    } else if (typeof window !== "undefined") {
      // Track current page for next navigation
      sessionStorage.setItem("lastPage", window.location.pathname);
    }
  }, [pathname, showChatView, setShowChatView]);

  // Ensure settings view is shown when on settings page
  React.useEffect(() => {
    if (pathname === "/settings") {
      // Clear any stale "manually hidden" state when navigating to settings from other pages
      const wasOnSettingsPage = sessionStorage.getItem("lastPage") === "/settings";
      const hasManuallyHidden = sessionStorage.getItem("settingsViewManuallyHidden") === "true";

      if (!wasOnSettingsPage) {
        // Coming from another page - always show settings view
        sessionStorage.removeItem("settingsViewManuallyHidden");
        setShowSettingsView(true);
      } else if (!hasManuallyHidden && !showSettingsView) {
        // Already on settings page and not manually hidden - show settings view
        setShowSettingsView(true);
      }

      // Track that we're on settings page
      sessionStorage.setItem("lastPage", "/settings");
    }
  }, [pathname, showSettingsView, setShowSettingsView]);

  // Track when user manually hides chat view
  const handleChatViewChange = React.useCallback(
    (show: boolean) => {
      if (typeof window !== "undefined" && !show) {
        sessionStorage.setItem("chatViewManuallyHidden", "true");
      } else if (typeof window !== "undefined" && show) {
        sessionStorage.removeItem("chatViewManuallyHidden");
      }
      setShowChatView(show);
    },
    [setShowChatView]
  );

  // Track when user manually hides settings view
  const handleSettingsViewChange = React.useCallback(
    (show: boolean) => {
      if (typeof window !== "undefined" && !show) {
        sessionStorage.setItem("settingsViewManuallyHidden", "true");
      } else if (typeof window !== "undefined" && show) {
        sessionStorage.removeItem("settingsViewManuallyHidden");
      }
      setShowSettingsView(show);
    },
    [setShowSettingsView]
  );

  return (
    <MorphikSidebar
      showEditableUri={true}
      connectionUri={connectionUri}
      onUriChange={updateConnectionUri}
      userProfile={userProfile}
      onLogout={onLogout}
      onProfileNavigate={onProfileNavigate}
      onUpgradeClick={onUpgradeClick}
      showChatView={showChatView}
      onChatViewChange={handleChatViewChange}
      activeChatId={activeChatId}
      onChatSelect={setActiveChatId}
      showSettingsView={showSettingsView}
      onSettingsViewChange={handleSettingsViewChange}
      activeSettingsTab={activeSettingsTab}
      onSettingsTabChange={setActiveSettingsTab}
    />
  );
}
