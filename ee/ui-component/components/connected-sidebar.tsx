"use client";

import React, { useState, createContext, useContext, useCallback } from "react";
import { useMorphik } from "@/contexts/morphik-context";
import { MorphikSidebar } from "@/components/morphik-sidebar";

// Create a context for chat and settings state sharing
interface ChatContextType {
  activeChatId?: string;
  setActiveChatId: (id: string | undefined) => void;
  showChatView: boolean;
  setShowChatView: (show: boolean) => void;
  showSettingsView: boolean;
  setShowSettingsView: (show: boolean) => void;
  activeSettingsTab: string;
  setActiveSettingsTab: (tab: string) => void;
}

const ChatContext = createContext<ChatContextType | null>(null);

export function useChatContext() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [showChatView, setShowChatView] = useState(false);
  const [activeChatId, setActiveChatId] = useState<string | undefined>();
  const [showSettingsView, setShowSettingsView] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState("api-keys");

  // Memoize the setter functions to prevent unnecessary re-renders
  const setActiveChatIdMemo = useCallback((id: string | undefined) => {
    setActiveChatId(prev => (prev !== id ? id : prev));
  }, []);

  const setShowChatViewMemo = useCallback((show: boolean) => {
    setShowChatView(prev => (prev !== show ? show : prev));
  }, []);

  const setShowSettingsViewMemo = useCallback((show: boolean) => {
    setShowSettingsView(prev => (prev !== show ? show : prev));
  }, []);

  const setActiveSettingsTabMemo = useCallback((tab: string) => {
    setActiveSettingsTab(prev => (prev !== tab ? tab : prev));
  }, []);

  // Memoize the context value to prevent unnecessary re-renders
  const contextValue = React.useMemo(
    () => ({
      activeChatId,
      setActiveChatId: setActiveChatIdMemo,
      showChatView,
      setShowChatView: setShowChatViewMemo,
      showSettingsView,
      setShowSettingsView: setShowSettingsViewMemo,
      activeSettingsTab,
      setActiveSettingsTab: setActiveSettingsTabMemo,
    }),
    [
      activeChatId,
      showChatView,
      showSettingsView,
      activeSettingsTab,
      setActiveChatIdMemo,
      setShowChatViewMemo,
      setShowSettingsViewMemo,
      setActiveSettingsTabMemo,
    ]
  );

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
}

export function ConnectedSidebar() {
  const { connectionUri, updateConnectionUri, userProfile, onLogout, onProfileNavigate, onUpgradeClick } = useMorphik();
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
    if (typeof window !== "undefined" && window.location.pathname === "/chat") {
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
  }, [showChatView, setShowChatView]); // Run when showChatView changes

  // Ensure settings view is shown when on settings page
  React.useEffect(() => {
    if (typeof window !== "undefined" && window.location.pathname === "/settings") {
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
  }, [showSettingsView, setShowSettingsView]); // Run when showSettingsView changes

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
