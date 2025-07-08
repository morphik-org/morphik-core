"use client";

import * as React from "react";
import Image from "next/image";
import {
  IconFiles,
  IconSearch,
  IconMessage,
  IconShare,
  IconPlugConnected,
  IconFileText,
  IconSettings,
  IconFileAnalytics,
  IconGitBranch,
  IconBook,
  IconMessageCircle,
} from "@tabler/icons-react";

import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar-new";

const data = {
  user: {
    name: "Morphik User",
    email: "user@morphik.ai",
    avatar: "/assets/placeholder-user.jpg",
  },
  navMain: [
    {
      title: "Documents",
      url: "/documents",
      icon: IconFiles,
    },
    {
      title: "PDF Viewer",
      url: "/pdf",
      icon: IconFileText,
    },
    {
      title: "Search",
      url: "/search",
      icon: IconSearch,
    },
    {
      title: "Chat",
      url: "/chat",
      icon: IconMessage,
    },
    {
      title: "Knowledge Graph",
      url: "/graphs",
      icon: IconShare,
    },
    {
      title: "Workflows",
      url: "/workflows",
      icon: IconGitBranch,
    },
    {
      title: "Connections",
      url: "/connections",
      icon: IconPlugConnected,
    },
  ],
  navSecondary: [
    {
      title: "Settings",
      url: "/settings",
      icon: IconSettings,
    },
    {
      title: "Logs",
      url: "/logs",
      icon: IconFileAnalytics,
    },
    {
      title: "Documentation",
      url: "https://docs.morphik.ai",
      icon: IconBook,
    },
    {
      title: "Send Feedback",
      url: "mailto:founders@morphik.ai",
      icon: IconMessageCircle,
    },
  ],
};

interface MorphikSidebarProps extends React.ComponentProps<typeof Sidebar> {
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
}

export function MorphikSidebar({ userProfile, onLogout, onProfileNavigate, ...props }: MorphikSidebarProps) {
  const [mounted, setMounted] = React.useState(false);

  // Ensure component is mounted before rendering theme-dependent content
  React.useEffect(() => {
    setMounted(true);
  }, []);

  // Just need to check if component is mounted for theme-dependent rendering

  // Merge user profile with defaults
  const userData = {
    name: userProfile?.name || data.user.name,
    email: userProfile?.email || data.user.email,
    avatar: userProfile?.avatar || data.user.avatar,
  };

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild className="data-[slot=sidebar-menu-button]:!p-2">
              <a href="/" className="flex items-center">
                {mounted && (
                  <>
                    <Image
                      src="/morphikwhite.png"
                      alt="Morphik Logo"
                      width={120}
                      height={32}
                      className="hidden h-8 w-auto object-contain dark:block"
                      priority
                    />
                    <Image
                      src="/morphikblack.png"
                      alt="Morphik Logo"
                      width={120}
                      height={32}
                      className="block h-8 w-auto object-contain dark:hidden"
                      priority
                    />
                  </>
                )}
                {!mounted && <div className="h-8 w-[120px]" />}
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={userData} onLogout={onLogout} onProfileNavigate={onProfileNavigate} />
      </SidebarFooter>
    </Sidebar>
  );
}
