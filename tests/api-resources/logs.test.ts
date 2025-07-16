// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource logs', () => {
  // skipped: tests are disabled for the time being
  test.skip('list', async () => {
    const responsePromise = client.logs.list();
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
    await expect(
      client.logs.list(
        { limit: 1, op_type: 'op_type', since: '2019-12-27T18:11:19.117Z', status: 'status' },
        { path: '/_stainless_unknown_path' },
      ),
    ).rejects.toThrow(Morphik.NotFoundError);
  });
});
