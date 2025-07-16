// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource agent', () => {
  // skipped: tests are disabled for the time being
  test.skip('executeQuery: only required params', async () => {
    const responsePromise = client.agent.executeQuery({ query: 'query' });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('executeQuery: required and optional params', async () => {
    const response = await client.agent.executeQuery({
      query: 'query',
      chat_id: 'chat_id',
      display_mode: 'formatted',
    });
  });
});
