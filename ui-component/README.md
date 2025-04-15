# @morphik/ui

A modern React-based UI for Morphik, built with Next.js and Tailwind CSS. This component provides a user-friendly interface for:
- Document management and uploads
- Interactive chat with your knowledge base
- Real-time document processing feedback
- Query testing and prototyping

## Installation

```bash
npm install @morphik/ui
```

## Usage

```jsx
import { MorphikUI } from '@morphik/ui';

export default function YourApp() {
  return (
    <MorphikUI 
      connectionUri="your-connection-uri" 
      apiBaseUrl="http://your-api-base-url" 
      isReadOnlyUri={false}
      onUriChange={(uri) => console.log('URI changed:', uri)}
    />
  );
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `connectionUri` | `string` | `undefined` | Connection URI for Morphik API |
| `apiBaseUrl` | `string` | `"http://localhost:8000"` | Base URL for API requests |
| `isReadOnlyUri` | `boolean` | `false` | Controls whether the URI can be edited |
| `onUriChange` | `(uri: string) => void` | `undefined` | Callback when URI is changed |

## Prerequisites

- Node.js 18 or later
- npm or yarn package manager
- A running Morphik server

## Development Quick Start

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

4. Connect to your Morphik server using a URI from the `/local/generate_uri` endpoint

## Features

- **Document Management**
  - Upload various file types (PDF, TXT, MD, MP3)
  - View and manage uploaded documents
  - Real-time processing status
  - Collapsible document panel

- **Chat Interface**
  - Real-time chat with your knowledge base
  - Support for long messages
  - Message history
  - Markdown rendering

- **Connection Management**
  - Easy server connection
  - Connection status indicator
  - Automatic reconnection
  - Error handling

## Development

The UI is built with:
- [Next.js 14](https://nextjs.org)
- [Tailwind CSS](https://tailwindcss.com)
- [shadcn/ui](https://ui.shadcn.com)
- [React](https://reactjs.org)

### Project Structure
```
ui-component/
├── app/              # Next.js app directory
├── components/       # Reusable UI components
├── lib/             # Utility functions and hooks
└── public/          # Static assets
```

### Building for Production

```bash
npm run build
npm start
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is part of Morphik and is licensed under the MIT License.
