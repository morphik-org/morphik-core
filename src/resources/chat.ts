// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import { APIResource } from '../core/resource';
import { APIPromise } from '../core/api-promise';
import { RequestOptions } from '../internal/request-options';
import { path } from '../internal/utils/path';

export class Chat extends APIResource {
  /**
   * List chat conversations available to the current user.
   *
   * Args: auth: Authentication context containing user and app identifiers. limit:
   * Maximum number of conversations to return (1-500)
   *
   * Returns: A list of dictionaries describing each conversation, ordered by most
   * recent activity.
   */
  list(
    query: ChatListParams | null | undefined = {},
    options?: RequestOptions,
  ): APIPromise<ChatListResponse> {
    return this._client.get('/chats', { query, ...options });
  }

  /**
   * Retrieve the message history for a chat conversation.
   *
   * Args: chat_id: Identifier of the conversation whose history should be loaded.
   * auth: Authentication context used to verify access to the conversation. redis:
   * Redis connection where chat messages are stored.
   *
   * Returns: A list of :class:`ChatMessage` objects or an empty list if no history
   * exists.
   */
  retrieveHistory(chatID: string, options?: RequestOptions): APIPromise<ChatRetrieveHistoryResponse> {
    return this._client.get(path`/chat/${chatID}`, options);
  }

  /**
   * Update the title of a chat conversation.
   *
   * Args: chat_id: ID of the chat conversation to update title: New title for the
   * chat auth: Authentication context
   *
   * Returns: Success status
   */
  updateTitle(
    chatID: string,
    params: ChatUpdateTitleParams,
    options?: RequestOptions,
  ): APIPromise<ChatUpdateTitleResponse> {
    const { title } = params;
    return this._client.patch(path`/chats/${chatID}/title`, { query: { title }, ...options });
  }
}

export type ChatListResponse = Array<{ [key: string]: unknown }>;

export type ChatRetrieveHistoryResponse = Array<ChatRetrieveHistoryResponse.ChatRetrieveHistoryResponseItem>;

export namespace ChatRetrieveHistoryResponse {
  /**
   * Simple chat message model for chat persistence.
   */
  export interface ChatRetrieveHistoryResponseItem {
    content: string;

    role: 'user' | 'assistant';

    agent_data?: { [key: string]: unknown } | null;

    timestamp?: string;
  }
}

/**
 * Response for chat title update endpoint
 */
export interface ChatUpdateTitleResponse {
  message: string;

  status: string;

  title: string;
}

export interface ChatListParams {
  limit?: number;
}

export interface ChatUpdateTitleParams {
  /**
   * New title for the chat
   */
  title: string;
}

export declare namespace Chat {
  export {
    type ChatListResponse as ChatListResponse,
    type ChatRetrieveHistoryResponse as ChatRetrieveHistoryResponse,
    type ChatUpdateTitleResponse as ChatUpdateTitleResponse,
    type ChatListParams as ChatListParams,
    type ChatUpdateTitleParams as ChatUpdateTitleParams,
  };
}
