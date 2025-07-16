// File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import Morphik from 'morphik';

const client = new Morphik({
  apiKey: 'My API Key',
  baseURL: process.env['TEST_API_BASE_URL'] ?? 'http://127.0.0.1:4010',
});

describe('resource query', () => {
  // skipped: tests are disabled for the time being
  test.skip('generateCompletion: only required params', async () => {
    const responsePromise = client.query.generateCompletion({ query: 'x' });
    const rawResponse = await responsePromise.asResponse();
    expect(rawResponse).toBeInstanceOf(Response);
    const response = await responsePromise;
    expect(response).not.toBeInstanceOf(Response);
    const dataAndResponse = await responsePromise.withResponse();
    expect(dataAndResponse.data).toBe(response);
    expect(dataAndResponse.response).toBe(rawResponse);
  });

  // skipped: tests are disabled for the time being
  test.skip('generateCompletion: required and optional params', async () => {
    const response = await client.query.generateCompletion({
      query: 'x',
      chat_id: 'chat_id',
      end_user_id: 'end_user_id',
      filters: { foo: 'bar' },
      folder_name: 'string',
      graph_name: 'graph_name',
      hop_depth: 1,
      include_paths: true,
      k: 1,
      llm_config: { foo: 'bar' },
      max_tokens: 0,
      min_score: 0,
      padding: 0,
      prompt_overrides: {
        entity_extraction: {
          examples: [{ label: 'label', type: 'type', properties: { foo: 'bar' } }],
          prompt_template: 'prompt_template',
        },
        entity_resolution: {
          examples: [{ canonical: 'canonical', variants: ['string'] }],
          prompt_template: 'prompt_template',
        },
        query: { prompt_template: 'prompt_template' },
      },
      schema: {},
      stream_response: true,
      temperature: 0,
      use_colpali: true,
      use_reranking: true,
    });
  });
});
