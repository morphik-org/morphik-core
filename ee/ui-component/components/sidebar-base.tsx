"use client";

/*
 * AppSidebar (BaseSidebar)
 *
 * High-level sidebar composition used by both Local and Cloud variants.
 * - Renders navigation entries and special Chat/Settings side panels
 * - Delegates collapse/expand to SidebarProvider (primitives)
 * - Receives a NavigationStrategy to decide URL vs Section navigation
 */

import * as React from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { IconArrowRight, IconArrowLeft, IconPlus } from "@tabler/icons-react";

import { NavUser } from "@/components/nav-user";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { SettingsSidebar } from "@/components/settings/SettingsSidebar";
import { useMorphik } from "@/contexts/morphik-context";
import { ConnectionInput } from "@/components/ConnectionInput";
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
  useSidebar,
} from "@/components/ui/sidebar-components";
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
  logoLight?: string;
  logoDark?: string;
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
  logoLight,
  logoDark,
  ...props
}: BaseSidebarProps) {
  const { apiBaseUrl, authToken, isLocal } = useMorphik();
  const { state, setOpen, toggleSidebar } = useSidebar();
  const router = useRouter();
  const pathname = usePathname() || "/";

  const userData = {
    name: userProfile?.name || "Morphik User",
    email: userProfile?.email || "user@morphik.ai",
    avatar: userProfile?.avatar || "/assets/placeholder-user.jpg",
  };

  const defaultLogoComponent = (
    <>
      <Image
        src={logoDark || "/morphik-logos/morphik_logo_white_on_transparent_no_text.png"}
        alt="Morphik Logo"
        width={24}
        height={24}
        className="hidden h-6 w-6 object-contain dark:block"
        priority
      />
      <Image
        src={logoLight || "/morphik-logos/morphik_logo_black_on_transparent_no_text.png"}
        alt="Morphik Logo"
        width={24}
        height={24}
        className="block h-6 w-6 object-contain dark:hidden"
        priority
      />
    </>
  );

  // Auto-expand only on ENTERING chat or settings, not after user collapses
  const prevViewsRef = React.useRef({
    showChatView,
    showSettingsView,
  });
  React.useEffect(() => {
    const enteredChat = !prevViewsRef.current.showChatView && showChatView;
    const enteredSettings = !prevViewsRef.current.showSettingsView && showSettingsView;
    if (enteredChat || enteredSettings) {
      setOpen(true);
    }
    prevViewsRef.current = { showChatView, showSettingsView };
  }, [showChatView, showSettingsView, setOpen]);

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
                <div className="flex h-[26px] w-[26px] shrink-0 items-center justify-center overflow-visible rounded-lg">
                  {defaultLogoComponent}
                </div>
                {userProfile?.tier && (
                  <Badge variant="secondary" className="ml-2 text-[10px] group-data-[collapsible=icon]:hidden">
                    {userProfile.tier.charAt(0).toUpperCase() + userProfile.tier.slice(1)}
                  </Badge>
                )}
              </SidebarMenuButton>
            ) : (
              <SidebarMenuButton asChild>
                <Link
                  href="/"
                  className="flex items-center group-data-[collapsible=icon]:justify-center group-data-[state=collapsed]:p-0"
                >
                  {defaultLogoComponent}
                </Link>
              </SidebarMenuButton>
            )}
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent className="relative flex flex-col">
        {showChatView ? (
          <div className="min-h-0 flex-1">
            {/* Back to main menu */}
            <SidebarGroup>
              <SidebarGroupContent className="px-2 py-1">
                {state === "collapsed" ? (
                  <div className="flex w-full items-center justify-center">
                    <Button
                      variant="ghost"
                      size="icon"
                      className=""
                      onClick={() => onChatViewChange?.(false)}
                      title="Back"
                    >
                      <IconArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start gap-2 text-sm"
                    onClick={() => onChatViewChange?.(false)}
                  >
                    <IconArrowLeft className="h-4 w-4" />
                    Back to Menu
                  </Button>
                )}
              </SidebarGroupContent>
            </SidebarGroup>
            {/* Collapsed rail actions for chat */}
            {state === "collapsed" && (
              <SidebarGroup>
                <SidebarGroupContent className="flex items-center justify-center px-2 py-1">
                  <Button variant="ghost" size="icon" title="New chat" onClick={() => onChatSelect?.(undefined)}>
                    <IconPlus className="h-4 w-4" />
                  </Button>
                </SidebarGroupContent>
              </SidebarGroup>
            )}
            <div className="h-full [&>div>div:first-child]:!px-2 [&>div>div:first-child]:!py-1 [&>div>div:last-child]:!px-2 [&>div]:!w-full [&>div]:!border-r-0 [&>div]:!bg-transparent [&_button]:!px-2 [&_button]:!py-1.5 [&_li>div]:!px-1 [&_ul]:!p-2">
              <ChatSidebar
                apiBaseUrl={apiBaseUrl}
                authToken={authToken}
                activeChatId={activeChatId}
                onSelect={chatId => onChatSelect?.(chatId)}
                collapsed={state === "collapsed"}
                onToggle={toggleSidebar}
              />
            </div>
          </div>
        ) : showSettingsView ? (
          <div className="min-h-0 flex-1">
            {/* Back to main menu */}
            <SidebarGroup>
              <SidebarGroupContent className="px-2 py-1">
                {state === "collapsed" ? (
                  <div className="flex w-full items-center justify-center">
                    <Button variant="ghost" size="icon" onClick={() => onSettingsViewChange?.(false)} title="Back">
                      <IconArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start gap-2 text-sm"
                    onClick={() => onSettingsViewChange?.(false)}
                  >
                    <IconArrowLeft className="h-4 w-4" />
                    Back to Menu
                  </Button>
                )}
              </SidebarGroupContent>
            </SidebarGroup>
            <div className="h-full [&>div>div:first-child]:!px-2 [&>div>div:first-child]:!py-1 [&>div>div:last-child]:!px-2 [&>div]:!w-full [&>div]:!border-r-0 [&>div]:!bg-transparent [&_button]:!px-2 [&_button]:!py-1.5 [&_li>div]:!px-1 [&_ul]:!p-2">
              <SettingsSidebar
                activeTab={activeSettingsTab}
                onTabChange={tab => onSettingsTabChange?.(tab)}
                collapsed={state === "collapsed"}
                onToggle={toggleSidebar}
              />
            </div>
          </div>
        ) : (
          <>
            {/* URI editor for local dev */}
            {showEditableUri && (
              <SidebarGroup>
                <SidebarGroupContent className="px-2 py-1 group-data-[collapsible=icon]:hidden">
                  <ConnectionInput
                    value={connectionUri || ""}
                    onChange={onUriChange}
                    onClear={() => onUriChange?.("")}
                    placeholder="localhost:8000"
                  />
                  {connectionUri && isLocal && (
                    <p className="mt-1 text-[10px] text-muted-foreground">Local connection (clears on restart)</p>
                  )}
                </SidebarGroupContent>
              </SidebarGroup>
            )}
            {/* Main navigation */}
            <SidebarGroup>
              <SidebarGroupContent className="flex flex-col gap-2">
                <SidebarMenu>
                  {navigation.mainItems.map(item => (
                    <SidebarMenuItem key={item.title}>
                      {navigation.type === "url" && "url" in item ? (
                        item.isSpecial && item.title === "Chat" ? (
                          <SidebarMenuButton
                            tooltip={item.title}
                            onClick={() => {
                              if (pathname !== "/chat") router.push("/chat");
                              setOpen(true);
                              onChatViewChange?.(true);
                            }}
                            isActive={pathname === "/chat"}
                          >
                            {item.icon && <item.icon />}
                            <span>{item.title}</span>
                          </SidebarMenuButton>
                        ) : item.isSpecial && item.title === "Settings" ? (
                          <SidebarMenuButton
                            tooltip={item.title}
                            onClick={() => {
                              if (pathname !== "/settings") router.push("/settings");
                              setOpen(true);
                              onSettingsViewChange?.(true);
                            }}
                            isActive={pathname === "/settings"}
                          >
                            {item.icon && <item.icon />}
                            <span>{item.title}</span>
                          </SidebarMenuButton>
                        ) : (
                          <SidebarMenuButton tooltip={item.title} asChild>
                            <Link href={item.url}>
                              {item.icon && <item.icon />}
                              <span>{item.title}</span>
                            </Link>
                          </SidebarMenuButton>
                        )
                      ) : (
                        <SidebarMenuButton
                          tooltip={item.title}
                          onClick={() => navigation.onItemClick(item)}
                          isActive={
                            navigation.type === "section" && "section" in item
                              ? navigation.currentActive === item.section
                              : false
                          }
                        >
                          {item.icon && <item.icon />}
                          <span>{item.title}</span>
                        </SidebarMenuButton>
                      )}
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
            {/* Secondary nav */}
            <div className="mt-auto">
              <SidebarGroup>
                <SidebarGroupContent className="flex flex-col gap-2">
                  <SidebarMenu>
                    {navigation.secondaryItems.map(item => (
                      <SidebarMenuItem key={item.title}>
                        {navigation.type === "url" && "url" in item ? (
                          item.isSpecial && item.title === "Settings" ? (
                            <SidebarMenuButton
                              tooltip={item.title}
                              onClick={() => {
                                if (pathname !== "/settings") router.push("/settings");
                                setOpen(true);
                                onSettingsViewChange?.(true);
                              }}
                              isActive={pathname === "/settings"}
                            >
                              {item.icon && <item.icon />}
                              <span>{item.title}</span>
                            </SidebarMenuButton>
                          ) : (
                            <SidebarMenuButton tooltip={item.title} asChild>
                              <Link href={item.url}>
                                {item.icon && <item.icon />}
                                <span>{item.title}</span>
                              </Link>
                            </SidebarMenuButton>
                          )
                        ) : (
                          <SidebarMenuButton
                            tooltip={item.title}
                            onClick={() => navigation.onItemClick(item)}
                            isActive={
                              navigation.type === "section" && "section" in item
                                ? navigation.currentActive === item.section
                                : false
                            }
                          >
                            {item.icon && <item.icon />}
                            <span>{item.title}</span>
                          </SidebarMenuButton>
                        )}
                      </SidebarMenuItem>
                    ))}
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
              {onUpgradeClick && (userProfile?.tier === "free" || !userProfile?.tier) && (
                <div className="mx-2 mb-2 mt-2">
                  <Button
                    className="w-full justify-between group-data-[collapsible=icon]:justify-center"
                    variant="outline"
                    size="default"
                    onClick={onUpgradeClick}
                    title="Upgrade to PRO"
                  >
                    <div className="flex items-center gap-2 group-data-[collapsible=icon]:hidden">
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
          </>
        )}
      </SidebarContent>

      <SidebarFooter>
        <NavUser user={userData} onLogout={onLogout} onProfileNavigate={onProfileNavigate} />
      </SidebarFooter>
    </Sidebar>
  );
}
