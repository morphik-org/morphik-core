import { FolderSummary } from "../components/types";

export interface FolderTreeNode extends FolderSummary {
  children: FolderTreeNode[];
}

export interface FlattenedFolderNode extends FolderSummary {
  path: string;
  depthLevel: number;
}

export const normalizeFolderPathValue = (value?: string | null): string => {
  const trimmed = (value ?? "").trim();
  if (!trimmed) {
    return "";
  }

  const withoutEdges = trimmed.replace(/^\/+|\/+$/g, "");
  if (!withoutEdges) {
    return "/";
  }

  const segments = withoutEdges.split("/").filter(Boolean);
  return `/${segments.join("/")}`;
};

export const normalizeFolderPath = (folder: { full_path?: string | null; name?: string }): string => {
  return normalizeFolderPathValue(folder.full_path ?? folder.name ?? "");
};

export const buildFolderTree = (folders: FolderSummary[]): FolderTreeNode[] => {
  const pathMap = new Map<string, FolderTreeNode>();

  folders.forEach(folder => {
    const path = normalizeFolderPath(folder);
    const depth = folder.depth ?? (path === "/" ? 0 : path.split("/").filter(Boolean).length);
    pathMap.set(path, {
      ...folder,
      full_path: path,
      depth,
      children: [],
    });
  });

  const roots: FolderTreeNode[] = [];

  pathMap.forEach(node => {
    const parts = node.full_path ? node.full_path.split("/").filter(Boolean) : [];
    parts.pop();
    const parentPath = parts.length > 0 ? `/${parts.join("/")}` : "";

    if (parentPath && pathMap.has(parentPath)) {
      pathMap.get(parentPath)!.children.push(node);
    } else {
      roots.push(node);
    }
  });

  const sortNodes = (nodes: FolderTreeNode[]) => {
    nodes.sort((a, b) => (a.name ?? a.full_path ?? "").localeCompare(b.name ?? b.full_path ?? ""));
    nodes.forEach(child => sortNodes(child.children));
  };

  sortNodes(roots);
  return roots;
};

export const flattenFolderTree = (nodes: FolderTreeNode[], depth = 0): FlattenedFolderNode[] => {
  const flattened: FlattenedFolderNode[] = [];

  nodes.forEach(node => {
    flattened.push({
      ...node,
      path: normalizeFolderPath(node),
      depthLevel: depth,
    });

    if (node.children.length) {
      flattened.push(...flattenFolderTree(node.children, depth + 1));
    }
  });

  return flattened;
};
