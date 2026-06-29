# Morphik UI Component

A modern React-based UI for Morphik, built with Next.js and Tailwind CSS. This component provides a user-friendly interface for:

- Document management and uploads
- Interactive chat with your knowledge base
- Real-time document processing feedback
- Query testing and prototyping

## Development Quick Start

The UI component is a private Next.js app in this repository. It is not published as an `@morphik/ui` package.
Run these commands from `ee/ui-component`:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and connect to a running local Morphik server with `localhost:8000`.

The routed app lives under `app/`; a reusable dashboard component also exists at `components/MorphikUI.tsx`. Check `components/types.ts` for the supported internal prop contract before embedding it elsewhere in this app.

## Prerequisites

- Node.js 18 or later
- npm or yarn package manager
- A running Morphik server

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
  - Cloud connection persistence and automatic local connection clearing on restart
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

There is no `build:package` script in this package.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.
