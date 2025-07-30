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

export interface BaseNavItem {
  title: string;
  icon: React.ComponentType;
  isSpecial?: boolean;
}

export interface UrlNavItem extends BaseNavItem {
  type: "url";
  url: string;
}

export interface SectionNavItem extends BaseNavItem {
  type: "section";
  section: string;
}

export type NavItem = UrlNavItem | SectionNavItem;

export interface NavigationStrategy {
  type: "url" | "section";
  mainItems: NavItem[];
  secondaryItems: NavItem[];
  onItemClick: (item: NavItem) => void;
  currentActive?: string;
}

export const baseMainNavItems: Omit<BaseNavItem, "type">[] = [
  {
    title: "Documents",
    icon: IconFiles,
  },
  {
    title: "PDF Viewer",
    icon: IconFileText,
  },
  {
    title: "Search",
    icon: IconSearch,
  },
  {
    title: "Chat",
    icon: IconMessage,
    isSpecial: true,
  },
  {
    title: "Knowledge Graphs",
    icon: IconShare,
  },
  {
    title: "Workflows",
    icon: IconGitBranch,
  },
  {
    title: "Connectors",
    icon: IconPlugConnected,
  },
];

export const baseSecondaryNavItems: Omit<BaseNavItem, "type">[] = [
  {
    title: "Settings",
    icon: IconSettings,
    isSpecial: true,
  },
  {
    title: "Logs",
    icon: IconFileAnalytics,
  },
];

export const externalNavItems = [
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

export const createUrlNavigation = (onChatClick: () => void, onSettingsClick?: () => void): NavigationStrategy => ({
  type: "url",
  mainItems: baseMainNavItems.map((item, index) => ({
    ...item,
    type: "url" as const,
    url:
      index === 0
        ? "/documents"
        : index === 1
          ? "/pdf"
          : index === 2
            ? "/search"
            : index === 3
              ? "/chat"
              : index === 4
                ? "/graphs"
                : index === 5
                  ? "/workflows"
                  : "/connections",
  })),
  secondaryItems: baseSecondaryNavItems.map((item, index) => ({
    ...item,
    type: "url" as const,
    url: index === 0 ? "/settings" : "/logs",
  })),
  onItemClick: item => {
    if (item.isSpecial && "url" in item && item.url === "/chat") {
      onChatClick();
      window.location.href = item.url;
    } else if (item.isSpecial && "url" in item && item.url === "/settings" && onSettingsClick) {
      onSettingsClick();
      window.location.href = item.url;
    } else if ("url" in item) {
      window.location.href = item.url;
    }
  },
});

export const createSectionNavigation = (
  onSectionChange: (section: string) => void,
  onChatClick: () => void,
  currentSection?: string,
  onSettingsClick?: () => void
): NavigationStrategy => ({
  type: "section",
  mainItems: baseMainNavItems.map((item, index) => ({
    ...item,
    type: "section" as const,
    section:
      index === 0
        ? "documents"
        : index === 1
          ? "pdf"
          : index === 2
            ? "search"
            : index === 3
              ? "chat"
              : index === 4
                ? "graphs"
                : index === 5
                  ? "workflows"
                  : "connections",
  })),
  secondaryItems: baseSecondaryNavItems.map((item, index) => ({
    ...item,
    type: "section" as const,
    section: index === 0 ? "settings" : "logs",
  })),
  currentActive: currentSection,
  onItemClick: item => {
    if (item.isSpecial && "section" in item && item.section === "chat") {
      onChatClick();
      onSectionChange(item.section);
    } else if (item.isSpecial && "section" in item && item.section === "settings" && onSettingsClick) {
      onSettingsClick();
      onSectionChange(item.section);
    } else if ("section" in item) {
      onSectionChange(item.section);
    }
  },
});
