// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../../core/resource';
import { APIPromise } from '../../core/api-promise';
import { RequestOptions } from '../../internal/request-options';
import { path } from '../../internal/utils/path';

export class Chat extends APIResource {
  /**
   * Stream a chat completion response for a document chat conversation.
   *
   * Args: chat_id: Identifier of the document chat conversation. request: The chat
   * request containing the user message. auth: Authentication context. redis: Redis
   * connection for chat history storage.
   *
   * Returns: StreamingResponse with the assistant's response.
   */
  completeChat(chatID: string, body: ChatCompleteChatParams, options?: RequestOptions): APIPromise<unknown> {
    return this._client.post(path`/document/chat/${chatID}/complete`, { body, ...options });
  }

  /**
   * Retrieve the message history for a document chat conversation.
   *
   * Args: chat_id: Identifier of the document chat conversation. auth:
   * Authentication context used to verify access to the conversation. redis: Redis
   * connection where chat messages are stored.
   *
   * Returns: A list of message dictionaries or an empty list if no history exists.
   */
  retrieveHistory(chatID: string, options?: RequestOptions): APIPromise<unknown> {
    return this._client.get(path`/document/chat/${chatID}`, options);
  }
}

export type ChatCompleteChatResponse = unknown;

export type ChatRetrieveHistoryResponse = unknown;

export interface ChatCompleteChatParams {
  message: string;

  document_id?: string | null;

  session_id?: string | null;
}

export declare namespace Chat {
  export {
    type ChatCompleteChatResponse as ChatCompleteChatResponse,
    type ChatRetrieveHistoryResponse as ChatRetrieveHistoryResponse,
    type ChatCompleteChatParams as ChatCompleteChatParams,
  };
}
