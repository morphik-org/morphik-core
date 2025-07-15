// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource batch', () => {
  // skipped: tests are disabled for the time being
  test.skip('retrieveChunks: only required params', async () => {
    const responsePromise = client.batch.retrieveChunks({ body: { foo: 'bar' } });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('retrieveChunks: required and optional params', async () => {
    const response = await client.batch.retrieveChunks({ body: { foo: 'bar' } });
  });

  // skipped: tests are disabled for the time being
  test.skip('retrieveDocuments: only required params', async () => {
    const responsePromise = client.batch.retrieveDocuments({ body: { foo: 'bar' } });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('retrieveDocuments: required and optional params', async () => {
    const response = await client.batch.retrieveDocuments({ body: { foo: 'bar' } });
  });
});
