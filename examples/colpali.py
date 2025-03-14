import io
from databridge import DataBridge
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

## Connect to the DataBridge instance

db = DataBridge(os.getenv("DATABRIDGE_URI_2"), timeout=10000, is_local=True)


## Ingestion Pathway
db.ingest_file("assets/colpali_example.pdf", use_colpali=True)


## retrieving sources

chunks = db.retreive_chunks("At what frequency do we achieve the highest Image Rejection Ratio?", use_colpali=True, k=3)

for chunk in chunks:
    if isinstance(chunk.content, Image.Image):
        # image chunks will automatically be parsed as PIL.Image.Image objects
        chunk.content.show()
    else:
        print(chunk.content)

# You can also directly query a VLM as defined in `databridge.toml`
response = db.query("At what frequency do we achieve the highest Image Rejection Ratio?", use_colpali=True, k=3)
print(response.completion)
