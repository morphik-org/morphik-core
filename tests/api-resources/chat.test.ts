// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource chat', () => {
  // skipped: tests are disabled for the time being
  test.skip('list', async () => {
    const responsePromise = client.chat.list();
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('list: request options and params are passed correctly', async () => {
    // ensure the request options are being passed correctly by passing an invalid HTTP method in order to cause an error
    await expect(client.chat.list({ limit: 1 }, { path: '/_stainless_unknown_path' })).rejects.toThrow(
      Morphik.NotFoundError,
    );
  });

  // skipped: tests are disabled for the time being
  test.skip('retrieveHistory', async () => {
    const responsePromise = client.chat.retrieveHistory('chat_id');
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('updateTitle: only required params', async () => {
    const responsePromise = client.chat.updateTitle('chat_id', { title: 'title' });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('updateTitle: required and optional params', async () => {
    const response = await client.chat.updateTitle('chat_id', { title: 'title' });
  });
});
