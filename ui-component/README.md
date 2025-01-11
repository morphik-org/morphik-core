# DataBridge UI Component

A modern React-based UI for DataBridge, built with Next.js and Tailwind CSS. This component provides a user-friendly interface for:
- Document management and uploads
- Interactive chat with your knowledge base
- Real-time document processing feedback
- Query testing and prototyping

## Prerequisites

- Node.js 18 or later
- npm or yarn package manager
- A running DataBridge server

## Quick Start

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Start the development server:
```bash
npm run dev
# or
yarn dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

4. Connect to your DataBridge server using a URI from the `/local/generate_uri` endpoint

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

This project is part of DataBridge and is licensed under the MIT License.
