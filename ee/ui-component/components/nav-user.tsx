"use client";

import * as React from "react";
import { IconCreditCard, IconLogout, IconUserCircle, IconChevronUp, IconChevronDown } from "@tabler/icons-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar-new";
import { cn } from "@/lib/utils";

interface NavUserProps {
  user: {
    name: string;
    email: string;
    avatar: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing") => void;
}

export function NavUser({ user, onLogout, onProfileNavigate }: NavUserProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          size="lg"
          onClick={() => setIsOpen(!isOpen)}
          className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
        >
          <Avatar className="h-6 w-6 rounded-lg grayscale">
            <AvatarImage src={user.avatar} alt={user.name} />
            <AvatarFallback className="rounded-lg">M</AvatarFallback>
          </Avatar>
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="truncate font-medium">{user.name}</span>
            <span className="truncate text-xs text-muted-foreground">{user.email}</span>
          </div>
          {isOpen ? <IconChevronUp className="ml-auto h-4 w-4" /> : <IconChevronDown className="ml-auto h-4 w-4" />}
        </SidebarMenuButton>
      </SidebarMenuItem>

      {/* Expandable menu items */}
      <div className={cn("overflow-hidden transition-all duration-200", isOpen ? "max-h-48" : "max-h-0")}>
        <SidebarMenuItem>
          <SidebarMenuButton
            onClick={e => {
              e.preventDefault();
              if (onProfileNavigate) {
                onProfileNavigate("account");
              } else {
                window.location.href = "/account-settings";
              }
            }}
          >
            <IconUserCircle className="h-5 w-5" />
            <span>Account</span>
          </SidebarMenuButton>
        </SidebarMenuItem>

        <SidebarMenuItem>
          <SidebarMenuButton
            onClick={e => {
              e.preventDefault();
              if (onProfileNavigate) {
                onProfileNavigate("billing");
              } else {
                window.location.href = "/billing";
              }
            }}
          >
            <IconCreditCard className="h-5 w-5" />
            <span>Billing</span>
          </SidebarMenuButton>
        </SidebarMenuItem>

        <SidebarMenuItem>
          <SidebarMenuButton
            onClick={e => {
              e.preventDefault();
              if (onLogout) {
                onLogout();
              } else {
                console.warn("No logout handler provided");
              }
            }}
          >
            <IconLogout className="h-5 w-5" />
            <span>Log out</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </div>
    </SidebarMenu>
  );
}
