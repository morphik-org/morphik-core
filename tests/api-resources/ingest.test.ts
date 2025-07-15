// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik, { toFile } from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource ingest', () => {
  // skipped: tests are disabled for the time being
  test.skip('batchIngestFiles: only required params', async () => {
    const responsePromise = client.ingest.batchIngestFiles({
      files: [await toFile(Buffer.from('# my file contents'), 'README.md')],
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
  test.skip('batchIngestFiles: required and optional params', async () => {
    const response = await client.ingest.batchIngestFiles({
      files: [await toFile(Buffer.from('# my file contents'), 'README.md')],
      end_user_id: 'end_user_id',
      folder_name: 'folder_name',
      metadata: 'metadata',
      rules: 'rules',
      use_colpali: true,
    });
  });

  // skipped: tests are disabled for the time being
  test.skip('ingestFile: only required params', async () => {
    const responsePromise = client.ingest.ingestFile({
      file: await toFile(Buffer.from('# my file contents'), 'README.md'),
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
  test.skip('ingestFile: required and optional params', async () => {
    const response = await client.ingest.ingestFile({
      file: await toFile(Buffer.from('# my file contents'), 'README.md'),
      end_user_id: 'end_user_id',
      folder_name: 'folder_name',
      metadata: 'metadata',
      rules: 'rules',
      use_colpali: true,
    });
  });

  // skipped: tests are disabled for the time being
  test.skip('ingestText: only required params', async () => {
    const responsePromise = client.ingest.ingestText({ content: 'content' });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('ingestText: required and optional params', async () => {
    const response = await client.ingest.ingestText({
      content: 'content',
      end_user_id: 'end_user_id',
      filename: 'filename',
      folder_name: 'folder_name',
      metadata: { foo: 'bar' },
      rules: [{ foo: 'bar' }],
      use_colpali: true,
    });
  });
});
