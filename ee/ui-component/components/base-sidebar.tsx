"use client";

import * as React from "react";
import Image from "next/image";
import { IconArrowRight, IconArrowLeft, IconLink } from "@tabler/icons-react";

import { NavUser } from "@/components/nav-user";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { SettingsSidebar } from "@/components/settings/SettingsSidebar";
import { useMorphik } from "@/contexts/morphik-context";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarGroup,
  SidebarGroupContent,
} from "@/components/ui/sidebar-new";
import { NavigationStrategy, externalNavItems } from "@/lib/navigation-utils";

interface BaseSidebarProps extends React.ComponentProps<typeof Sidebar> {
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
  showChatView?: boolean;
  onChatViewChange?: (show: boolean) => void;
  activeChatId?: string;
  onChatSelect?: (id: string | undefined) => void;
  showSettingsView?: boolean;
  onSettingsViewChange?: (show: boolean) => void;
  activeSettingsTab?: string;
  onSettingsTabChange?: (tab: string) => void;
  navigation: NavigationStrategy;

  // Connection-related props (for local dev)
  showEditableUri?: boolean;
  connectionUri?: string | null;
  onUriChange?: (newUri: string) => void;

  // Logo handling
  logoClickHandler?: () => void;
  logoComponent?: React.ReactNode;
}

export function BaseSidebar({
  userProfile,
  onLogout,
  onProfileNavigate,
  onUpgradeClick,
  showChatView = false,
  onChatViewChange,
  activeChatId,
  onChatSelect,
  showSettingsView = false,
  onSettingsViewChange,
  activeSettingsTab = "api-keys",
  onSettingsTabChange,
  navigation,
  showEditableUri = false,
  connectionUri,
  onUriChange,
  logoClickHandler,
  logoComponent,
  ...props
}: BaseSidebarProps) {
  const [mounted, setMounted] = React.useState(false);
  const [uriInput, setUriInput] = React.useState(connectionUri || "");
  const { apiBaseUrl, authToken } = useMorphik();

  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    setUriInput(connectionUri || "");
  }, [connectionUri]);

  const handleUriSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onUriChange && uriInput.trim()) {
      const normalizeToMorphikUri = (uri: string) => {
        if (uri.startsWith("http://") || uri.startsWith("https://")) {
          return uri;
        }
        return `http://${uri}`;
      };
      const normalizedUri = normalizeToMorphikUri(uriInput.trim());
      onUriChange(normalizedUri);
    }
  };

  const userData = {
    name: userProfile?.name || "Morphik User",
    email: userProfile?.email || "user@morphik.ai",
    avatar: userProfile?.avatar || "/assets/placeholder-user.jpg",
  };

  const defaultLogoComponent = (
    <>
      {mounted && (
        <>
          <Image
            src="/morphik-logos/morphik_logo_white_on_transparent_no_text.png"
            alt="Morphik Logo"
            width={26}
            height={26}
            className="hidden h-[26px] w-[26px] object-contain group-data-[state=collapsed]:h-[26px] group-data-[state=collapsed]:w-[26px] dark:block"
            priority
          />
          <Image
            src="/morphik-logos/morphik_logo_black_on_transparent_no_text.png"
            alt="Morphik Logo"
            width={26}
            height={26}
            className="block h-[26px] w-[26px] object-contain group-data-[state=collapsed]:h-[26px] group-data-[state=collapsed]:w-[26px] dark:hidden"
            priority
          />
        </>
      )}
      {!mounted && (
        <div className="h-[26px] w-[26px] group-data-[state=collapsed]:h-[26px] group-data-[state=collapsed]:w-[26px]" />
      )}
    </>
  );

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            {logoComponent ? (
              <SidebarMenuButton asChild>{logoComponent}</SidebarMenuButton>
            ) : logoClickHandler ? (
              <SidebarMenuButton
                size="lg"
                className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                onClick={logoClickHandler}
              >
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground group-data-[state=collapsed]:size-[26px] group-data-[state=collapsed]:p-0">
                  {defaultLogoComponent}
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">Morphik</span>
                  <span className="truncate text-xs">
                    {userProfile?.tier ? userProfile.tier.charAt(0).toUpperCase() + userProfile.tier.slice(1) : ""}
                  </span>
                </div>
              </SidebarMenuButton>
            ) : (
              <SidebarMenuButton asChild>
                <a
                  href="/"
                  className="flex items-center group-data-[collapsible=icon]:justify-center group-data-[state=collapsed]:p-0"
                >
                  {defaultLogoComponent}
                </a>
              </SidebarMenuButton>
            )}
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent className="relative flex flex-col">
        {/* Chat view content - always rendered, controlled by opacity and pointer-events */}
        <div
          className={`absolute inset-0 flex flex-col transition-all duration-300 ease-out ${
            showChatView
              ? "pointer-events-auto z-10 translate-x-0 opacity-100"
              : "pointer-events-none z-0 translate-x-2 opacity-0"
          }`}
        >
          {/* Back button */}
          <SidebarGroup>
            <SidebarGroupContent className="px-2 py-1">
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 text-sm"
                onClick={() => onChatViewChange?.(false)}
              >
                <IconArrowLeft className="h-4 w-4" />
                Back to Menu
              </Button>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* Chat sidebar content */}
          <div className="min-h-0 flex-1">
            <div className="h-full [&>div>div:first-child]:!px-2 [&>div>div:first-child]:!py-1 [&>div>div:last-child]:!px-2 [&>div]:!w-full [&>div]:!border-r-0 [&>div]:!bg-transparent [&_button]:!px-2 [&_button]:!py-1.5 [&_li>div]:!px-1 [&_ul]:!p-2">
              <ChatSidebar
                apiBaseUrl={apiBaseUrl}
                authToken={authToken}
                activeChatId={activeChatId}
                onSelect={chatId => {
                  onChatSelect?.(chatId);
                }}
                collapsed={false}
                onToggle={() => {}}
              />
            </div>
          </div>
        </div>

        {/* Settings view content - always rendered, controlled by opacity and pointer-events */}
        <div
          className={`absolute inset-0 flex flex-col transition-all duration-300 ease-out ${
            showSettingsView
              ? "pointer-events-auto z-10 translate-x-0 opacity-100"
              : "pointer-events-none z-0 translate-x-2 opacity-0"
          }`}
        >
          {/* Back button */}
          <SidebarGroup>
            <SidebarGroupContent className="px-2 py-1">
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 text-sm"
                onClick={() => onSettingsViewChange?.(false)}
              >
                <IconArrowLeft className="h-4 w-4" />
                Back to Menu
              </Button>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* Settings sidebar content */}
          <div className="min-h-0 flex-1">
            <div className="h-full [&>div>div:first-child]:!px-2 [&>div>div:first-child]:!py-1 [&>div>div:last-child]:!px-2 [&>div]:!w-full [&>div]:!border-r-0 [&>div]:!bg-transparent [&_button]:!px-2 [&_button]:!py-1.5 [&_li>div]:!px-1 [&_ul]:!p-2">
              <SettingsSidebar
                activeTab={activeSettingsTab}
                onTabChange={tab => onSettingsTabChange?.(tab)}
                collapsed={false}
                onToggle={() => {}}
              />
            </div>
          </div>
        </div>

        {/* Main navigation content - always rendered, controlled by opacity and pointer-events */}
        <div
          className={`absolute inset-0 flex flex-col transition-all duration-300 ease-out ${
            !showChatView && !showSettingsView
              ? "pointer-events-auto z-10 translate-x-0 opacity-100"
              : "pointer-events-none z-0 -translate-x-2 opacity-0"
          }`}
        >
          <div>
            {/* URI editor for local dev */}
            {showEditableUri && (
              <SidebarGroup>
                <SidebarGroupContent className="px-2 py-1">
                  <form onSubmit={handleUriSubmit} className="space-y-1">
                    <div className="flex gap-1">
                      <Input
                        id="connection-uri"
                        type="text"
                        placeholder="localhost:8000"
                        value={uriInput}
                        onChange={e => setUriInput(e.target.value)}
                        className="h-7 text-xs"
                      />
                      <Button type="submit" size="sm" variant="ghost" className="h-7 w-7 p-0">
                        <IconLink className="h-3 w-3" />
                      </Button>
                    </div>
                  </form>
                </SidebarGroupContent>
              </SidebarGroup>
            )}

            {/* Main navigation */}
            <SidebarGroup>
              <SidebarGroupContent className="flex flex-col gap-2">
                <SidebarMenu>
                  {navigation.mainItems.map(item => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        tooltip={item.title}
                        onClick={() => navigation.onItemClick(item)}
                        isActive={
                          navigation.type === "section" && "section" in item
                            ? navigation.currentActive === item.section
                            : navigation.type === "url" && "url" in item
                              ? typeof window !== "undefined" && window.location.pathname === item.url
                              : false
                        }
                      >
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </div>

          {/* Secondary nav stays at bottom */}
          <div className="mt-auto">
            <SidebarGroup>
              <SidebarGroupContent className="flex flex-col gap-2">
                <SidebarMenu>
                  {navigation.secondaryItems.map(item => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        tooltip={item.title}
                        onClick={() => navigation.onItemClick(item)}
                        isActive={
                          navigation.type === "section" && "section" in item
                            ? navigation.currentActive === item.section
                            : navigation.type === "url" && "url" in item
                              ? typeof window !== "undefined" && window.location.pathname === item.url
                              : false
                        }
                      >
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}

                  {/* External links */}
                  {externalNavItems.map(item => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton tooltip={item.title} asChild>
                        <a href={item.url} target="_blank" rel="noopener noreferrer">
                          {item.icon && <item.icon />}
                          <span>{item.title}</span>
                        </a>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            {/* Upgrade button */}
            {onUpgradeClick && (userProfile?.tier === "free" || !userProfile?.tier) && (
              <div className="mx-2 mb-2 mt-2">
                <Button className="w-full justify-between" variant="outline" size="default" onClick={onUpgradeClick}>
                  <div className="flex items-center gap-2">
                    <span>Upgrade to</span>
                    <Badge variant="secondary" className="text-xs">
                      PRO
                    </Badge>
                  </div>
                  <IconArrowRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </SidebarContent>

      <SidebarFooter>
        <NavUser user={userData} onLogout={onLogout} onProfileNavigate={onProfileNavigate} />
      </SidebarFooter>
    </Sidebar>
  );
}
