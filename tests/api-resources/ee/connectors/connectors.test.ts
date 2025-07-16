// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource connectors', () => {
  // skipped: tests are disabled for the time being
  test.skip('disconnect', async () => {
    const responsePromise = client.ee.connectors.disconnect('connector_type');
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('getAuthStatus', async () => {
    const responsePromise = client.ee.connectors.getAuthStatus('connector_type');
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('handleOAuthCallback', async () => {
    const responsePromise = client.ee.connectors.handleOAuthCallback('connector_type');
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('handleOAuthCallback: request options and params are passed correctly', async () => {
    // ensure the request options are being passed correctly by passing an invalid HTTP method in order to cause an error
    await expect(
      client.ee.connectors.handleOAuthCallback(
        'connector_type',
        { code: 'code', error: 'error', error_description: 'error_description', state: 'state' },
        { path: '/_stainless_unknown_path' },
      ),
    ).rejects.toThrow(Morphik.NotFoundError);
  });

  // skipped: tests are disabled for the time being
  test.skip('ingestFile: only required params', async () => {
    const responsePromise = client.ee.connectors.ingestFile('connector_type', { file_id: 'file_id' });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('ingestFile: required and optional params', async () => {
    const response = await client.ee.connectors.ingestFile('connector_type', {
      file_id: 'file_id',
      metadata: { foo: 'bar' },
      morphik_end_user_id: 'morphik_end_user_id',
      morphik_folder_name: 'morphik_folder_name',
      rules: [{ foo: 'bar' }],
    });
  });

  // skipped: tests are disabled for the time being
  test.skip('listFiles', async () => {
    const responsePromise = client.ee.connectors.listFiles('connector_type');
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('listFiles: request options and params are passed correctly', async () => {
    // ensure the request options are being passed correctly by passing an invalid HTTP method in order to cause an error
    await expect(
      client.ee.connectors.listFiles(
        'connector_type',
        { page_size: 0, page_token: 'page_token', path: 'path', q_filter: 'q_filter' },
        { path: '/_stainless_unknown_path' },
      ),
    ).rejects.toThrow(Morphik.NotFoundError);
  });
});
