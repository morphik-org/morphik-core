import logging

import psycopg
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from pgvector.psycopg import Bit, register_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Connecting to database")

conn = psycopg.connect(dbname="pgvector_example", autocommit=True)

conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

conn.execute("DROP TABLE IF EXISTS documents_multivector")
conn.execute("CREATE TABLE documents_multivector (id bigserial PRIMARY KEY, embeddings bit(128)[])")
conn.execute(
    """
CREATE OR REPLACE FUNCTION max_sim(document bit[], query bit[]) RETURNS double precision AS $$
    WITH queries AS (
        SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query) AS foo
    ),
    documents_multivector AS (
        SELECT unnest(document) AS document
    ),
    similarities AS (
        SELECT query_number, 1 - ((document <~> query) / bit_length(query)) AS similarity FROM queries CROSS JOIN documents_multivector
    ),
    max_similarities AS (
        SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
    )
    SELECT SUM(max_similarity) FROM max_similarities
$$ LANGUAGE SQL
"""
)

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="mps",
    attn_implementation="eager",
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")


def generate_embeddings(content, is_query=False):
    logger = logging.getLogger(__name__)

    logger.info("Processing input")
    if is_query:
        processed = processor.process_queries([content]).to(model.device)
    else:
        processed = processor.process_images([content]).to(model.device)

    logger.info("Generating embeddings with model")
    with torch.no_grad():
        embeddings = model(**processed)

    logger.info("Converting embeddings to float32")
    embeddings = embeddings.to(torch.float32)

    logger.info("Converting embeddings to numpy array")
    numpy_embeddings = embeddings.numpy(force=True)

    logger.info(f"Generated embeddings with shape {numpy_embeddings.shape}")
    return numpy_embeddings


def binary_quantize(embedding):
    return Bit(embedding > 0)


# input = load_dataset('vidore/docvqa_test_subsampled', split='test[:3]')['image']
FILE_PATH = "/Users/arnav/Desktop/databridge/databridge-core/samples/Documents/62250266_origin.pdf"
input_images = convert_from_path(FILE_PATH)
for content in input_images:
    embeddings = generate_embeddings(content)
    # print(embeddings.size())
    embeddings = [binary_quantize(e) for e in embeddings[0]]

    conn.execute("INSERT INTO documents_multivector (embeddings) VALUES (%s)", (embeddings,))

query = "low-complexity IQ compensator"
query_embeddings = [binary_quantize(e) for e in generate_embeddings(query, is_query=True)[0]]
result = conn.execute(
    "SELECT id, max_sim(embeddings, %s) AS max_sim FROM documents_multivector ORDER BY max_sim DESC LIMIT 5",
    (query_embeddings,),
).fetchall()
for row in result:
    print(row)
