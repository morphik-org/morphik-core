import os

from dotenv import load_dotenv
from morphik import Morphik
from PIL import Image

load_dotenv()

## Connect to the Morphik instance
db = Morphik(os.getenv("MORPHIK_URI"), timeout=10000, is_local=True)

## Ingestion Pathway
db.ingest_file("examples/assets/colpali_example.pdf", use_colpali=True)

## Retrieving sources
chunks = db.retrieve_chunks("At what frequency do we achieve the highest Image Rejection Ratio?", use_colpali=True, k=3)

for chunk in chunks:
    if isinstance(chunk.content, Image.Image):
        # image chunks will automatically be parsed as PIL.Image.Image objects
        chunk.content.show()
    else:
        print(chunk.content)

# You can also directly query a VLM as defined in the configuration
response = db.query("At what frequency do we achieve the highest Image Rejection Ratio?", use_colpali=True, k=3)
print(response.completion)
