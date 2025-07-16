// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource ee', () => {
  // skipped: tests are disabled for the time being
  test.skip('createApp: only required params', async () => {
    const responsePromise = client.ee.createApp({ app_id: 'app_id', name: 'name' });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('createApp: required and optional params', async () => {
    const response = await client.ee.createApp({ app_id: 'app_id', name: 'name', expiry_days: 0 });
  });
});
