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

type NavSecondaryItem =
  | {
      title: string;
      section: string;
      icon: React.ComponentType;
    }
  | {
      title: string;
      url: string;
      icon: React.ComponentType;
    };

import { NavUser } from "@/components/nav-user";
import { Badge } from "@/components/ui/badge";
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

const navMainItems = [
  {
    title: "Documents",
    section: "documents" as const,
    icon: IconFiles,
  },
  {
    title: "PDF Viewer",
    section: "pdf" as const,
    icon: IconFileText,
  },
  {
    title: "Search",
    section: "search" as const,
    icon: IconSearch,
  },
  {
    title: "Chat",
    section: "chat" as const,
    icon: IconMessage,
  },
  {
    title: "Knowledge Graphs",
    section: "graphs" as const,
    icon: IconShare,
  },
  {
    title: "Workflows",
    section: "workflows" as const,
    icon: IconGitBranch,
  },
  {
    title: "Connectors",
    section: "connections" as const,
    icon: IconPlugConnected,
  },
];

const navSecondaryItems: NavSecondaryItem[] = [
  {
    title: "Settings",
    section: "settings" as const,
    icon: IconSettings,
  },
  {
    title: "Logs",
    section: "logs" as const,
    icon: IconFileAnalytics,
  },
  {
    title: "Documentation",
    url: "https://docs.morphik.com",
    icon: IconBook,
  },
  {
    title: "Send Feedback",
    url: "mailto:founders@morphik.ai",
    icon: IconMessageCircle,
  },
];

interface MorphikSidebarStatefulProps extends React.ComponentProps<typeof Sidebar> {
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
  logoLight?: string;
  logoDark?: string;
}

export function MorphikSidebarStateful({
  currentSection,
  onSectionChange,
  userProfile,
  onLogout,
  onProfileNavigate,
  logoLight = "/morphikblack.png",
  logoDark = "/morphikwhite.png",
  ...props
}: MorphikSidebarStatefulProps) {
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  const userData = {
    name: userProfile?.name || "Morphik User",
    email: userProfile?.email || "user@morphik.ai",
    avatar: userProfile?.avatar || "/assets/placeholder-user.jpg",
  };

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              className="flex items-center gap-2 data-[slot=sidebar-menu-button]:!p-2"
              onClick={() => {
                // In cloud UI, navigate to dashboard when logo is clicked
                if (typeof window !== "undefined" && window.location.pathname.includes("/")) {
                  const pathSegments = window.location.pathname.split("/").filter(Boolean);
                  // If we're in an app context (e.g., /app_id/documents), go to dashboard
                  if (pathSegments.length > 0 && !["login", "signup", "dashboard"].includes(pathSegments[0])) {
                    window.location.href = "/dashboard";
                    return;
                  }
                }
                // Otherwise, go to documents
                onSectionChange("documents");
              }}
            >
              {mounted && (
                <>
                  <Image
                    src={logoDark}
                    alt="Morphik Logo"
                    width={120}
                    height={32}
                    className="hidden h-8 w-auto object-contain dark:block"
                    priority
                  />
                  <Image
                    src={logoLight}
                    alt="Morphik Logo"
                    width={120}
                    height={32}
                    className="block h-8 w-auto object-contain dark:hidden"
                    priority
                  />
                </>
              )}
              {userProfile?.tier && (
                <Badge variant="secondary" className="text-xs">
                  {userProfile.tier.charAt(0).toUpperCase() + userProfile.tier.slice(1)}
                </Badge>
              )}
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent className="flex flex-col gap-2">
            <SidebarMenu>
              {navMainItems.map(item => (
                <SidebarMenuItem key={item.section}>
                  <SidebarMenuButton
                    tooltip={item.title}
                    onClick={() => onSectionChange(item.section)}
                    isActive={currentSection === item.section}
                  >
                    {item.icon && <item.icon />}
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup className="mt-auto">
          <SidebarGroupContent className="flex flex-col gap-2">
            <SidebarMenu>
              {navSecondaryItems.map(item => (
                <SidebarMenuItem key={item.title}>
                  {"section" in item ? (
                    <SidebarMenuButton
                      tooltip={item.title}
                      onClick={() => onSectionChange(item.section)}
                      isActive={currentSection === item.section}
                    >
                      {item.icon && <item.icon />}
                      <span>{item.title}</span>
                    </SidebarMenuButton>
                  ) : (
                    <SidebarMenuButton tooltip={item.title} asChild>
                      <a href={item.url} target="_blank" rel="noopener noreferrer">
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                      </a>
                    </SidebarMenuButton>
                  )}
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={userData} onLogout={onLogout} onProfileNavigate={onProfileNavigate} />
      </SidebarFooter>
    </Sidebar>
  );
}
