# Migration to assistant-ui

This document outlines the migration from custom Morphik chat components to assistant-ui.

## Overview

We've successfully migrated the Morphik chat system to use assistant-ui while preserving all existing functionality:

- ✅ **Custom Runtime**: `useMorphikRuntime` integrates Morphik's backend with assistant-ui
- ✅ **Enhanced Messages**: `MorphikMessage` component displays sources and metadata
- ✅ **Preserved Features**: All advanced settings, streaming, agent mode, etc.
- ✅ **Backward Compatibility**: Original components remain available

## New Components

### `AssistantChatSection`
- Modern replacement for `ChatSection` using assistant-ui
- Preserves all Morphik-specific features:
  - Document/folder filtering
  - Advanced query options
  - Agent mode toggle
  - Streaming responses
  - Source display

### `useMorphikRuntime`
- Custom runtime adapter for assistant-ui
- Handles Morphik API integration
- Manages query options and source enrichment
- Supports both streaming and non-streaming responses

### `MorphikMessage`
- Custom message component for assistant-ui
- Displays Morphik sources in accordion format
- Shows document metadata (score, page, chunk)
- Includes copy, edit, and refresh actions

## Usage

### Basic Chat
```tsx
import { AssistantChatSection } from "@/components/chat";

<AssistantChatSection
  apiBaseUrl="http://localhost:8000"
  authToken={token}
  onChatSubmit={(query, options) => console.log(query, options)}
/>
```

### With Custom Runtime
```tsx
import { useMorphikRuntime } from "@/hooks/useMorphikRuntime";
import { AssistantRuntimeProvider, Thread } from "@assistant-ui/react";

const { runtime } = useMorphikRuntime({
  chatId: "chat-123",
  apiBaseUrl: "http://localhost:8000",
  authToken: token,
  streamResponse: true,
});

<AssistantRuntimeProvider runtime={runtime}>
  <Thread>
    {/* Your chat UI */}
  </Thread>
</AssistantRuntimeProvider>
```

## Migration Benefits

1. **Modern UI Framework**: Built on assistant-ui's robust foundation
2. **Better Composability**: Modular components for easier customization
3. **Enhanced Features**: Built-in message editing, branching, and more
4. **Future-Proof**: Easy to extend with assistant-ui's ecosystem
5. **Maintained Compatibility**: All existing Morphik features preserved

## Test Page

A test page is available at `/test/assistant-chat` to demo the new implementation.

## Backward Compatibility

The original `ChatSection` component remains available. You can gradually migrate by:

1. Testing with `AssistantChatSection`
2. Updating imports when ready
3. Removing old components when no longer needed

## Next Steps

- Test the new implementation thoroughly
- Update documentation and examples
- Consider additional assistant-ui features (plugins, tools, etc.)
- Remove deprecated components when fully migrated
