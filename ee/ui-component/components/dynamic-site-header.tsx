"use client";

import { usePathname } from "next/navigation";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar-components";
import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { useHeader } from "@/contexts/header-context";
import { ThemeToggle } from "@/components/theme-toggle";

interface Breadcrumb {
  label: string;
  href?: string;
  onClick?: (e: React.MouseEvent) => void;
}

export interface DynamicSiteHeaderProps {
  customBreadcrumbs?: Breadcrumb[];
  rightContent?: React.ReactNode;
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
}

export function DynamicSiteHeader({
  customBreadcrumbs: propBreadcrumbs,
  rightContent: propRightContent,
}: DynamicSiteHeaderProps = {}) {
  const pathname = usePathname();
  const { customBreadcrumbs: contextBreadcrumbs, rightContent: contextRightContent } = useHeader();

  // Use context values if available, otherwise use props, otherwise generate from pathname
  const breadcrumbs = contextBreadcrumbs || propBreadcrumbs || generateBreadcrumbs(pathname || "");
  const rightContent = contextRightContent || propRightContent;

  return (
    <header className="group-has-data-[collapsible=icon]/sidebar-wrapper:h-[var(--header-height)] flex h-[var(--header-height)] shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mx-2 data-[orientation=vertical]:h-4" />

        {/* Breadcrumbs */}
        <nav className="flex items-center space-x-1 text-sm">
          {breadcrumbs.map((crumb, index) => (
            <div key={index} className="flex items-center">
              {index > 0 && <ChevronRight className="mx-1 h-4 w-4 text-muted-foreground" />}
              {crumb.href || crumb.onClick ? (
                crumb.onClick ? (
                  <button
                    onClick={crumb.onClick}
                    className="text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {crumb.label}
                  </button>
                ) : (
                  <Link href={crumb.href!} className="text-muted-foreground transition-colors hover:text-foreground">
                    {crumb.label}
                  </Link>
                )
              ) : (
                <span className="font-medium text-foreground">{crumb.label}</span>
              )}
            </div>
          ))}
        </nav>

        {/* Right side content */}
        <div className="ml-auto flex items-center gap-2">
          {rightContent}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

function generateBreadcrumbs(pathname: string): Breadcrumb[] {
  const segments = pathname.split("/").filter(Boolean);

  // If we're at root, just show Home
  if (segments.length === 0) {
    return [{ label: "Home" }];
  }

  const breadcrumbs: Breadcrumb[] = [{ label: "Home", href: "/" }];

  // Add the current section
  const section = segments[0];
  const sectionLabel = getSectionLabel(section);
  breadcrumbs.push({ label: sectionLabel });

  return breadcrumbs;
}

function getSectionLabel(section: string): string {
  const labels: Record<string, string> = {
    documents: "Documents",
    search: "Search",
    chat: "Chat",
    graphs: "Knowledge Graph",
    workflows: "Workflows",
    connections: "Connections",
    settings: "Settings",
    logs: "Logs",
    pdf: "PDF Viewer",
  };

  return labels[section] || section.charAt(0).toUpperCase() + section.slice(1);
}
