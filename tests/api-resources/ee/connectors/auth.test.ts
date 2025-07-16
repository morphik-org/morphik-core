// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource auth', () => {
  // skipped: tests are disabled for the time being
  test.skip('finalizeManualAuth: only required params', async () => {
    const responsePromise = client.ee.connectors.auth.finalizeManualAuth('connector_type', {
      credentials: { foo: 'bar' },
    });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('finalizeManualAuth: required and optional params', async () => {
    const response = await client.ee.connectors.auth.finalizeManualAuth('connector_type', {
      credentials: { foo: 'bar' },
    });
  });

  // skipped: tests are disabled for the time being
  test.skip('getInitiateAuthURL', async () => {
    const responsePromise = client.ee.connectors.auth.getInitiateAuthURL('connector_type');
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('getInitiateAuthURL: request options and params are passed correctly', async () => {
    // ensure the request options are being passed correctly by passing an invalid HTTP method in order to cause an error
    await expect(
      client.ee.connectors.auth.getInitiateAuthURL(
        'connector_type',
        { app_redirect_uri: 'app_redirect_uri' },
        { path: '/_stainless_unknown_path' },
      ),
    ).rejects.toThrow(Morphik.NotFoundError);
  });
});
