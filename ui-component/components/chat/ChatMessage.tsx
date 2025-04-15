"use client";

import React from 'react';

// Define our own props interface to avoid empty interface error
interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ role, content }) => {
  return (
    <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div 
        className={`max-w-3/4 p-3 rounded-lg ${
          role === 'user' 
            ? 'bg-primary text-primary-foreground' 
            : 'bg-muted'
        }`}
      >
        <div className="whitespace-pre-wrap">{content}</div>
      </div>
    </div>
  );
};

export default ChatMessageComponent;