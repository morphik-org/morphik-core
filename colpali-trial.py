from typing import List, cast
import torch
from PIL import Image
import logging
 
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_name = "vidore/colqwen2-v0.1"

device = get_torch_device("auto")  # This will select MPS on Mac
cpu_device = torch.device("cpu")
print(f"Using device: {device}")

# Load the model
model = cast(
    ColQwen2,
    ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Don't use automatic device mapping
        low_cpu_mem_usage=True,
    ),
).eval()

# Move model to MPS except for visual module
model.to(device)
model.visual = model.visual.to('cpu')

# Load the processor
processor = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))
logger.info("Model and processor loaded successfully")

def convert_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert a PDF to a list of images."""
    logger.info(f"Converting PDF {pdf_path} to images...")
    images = convert_from_path(pdf_path)
    logger.info(f"Successfully converted PDF to {len(images)} images")
    return images

images = convert_to_images("test.pdf")

queries = [
    "Tell me about the adaptive filter compensation model.",
    "at what frequency, in megahertz, do we obtain the highest IRR or image rejection ratio?",
]
logger.info(f"Processing {len(queries)} queries")

# Process the inputs
logger.info("Processing images and queries...")
# Process images on CPU first since visual module is on CPU
batch_images = processor.process_images(images).to('cpu')
batch_queries = processor.process_queries(queries).to(device)

# Forward pass
logger.info("Performing inference...")
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
logger.info("Inference completed successfully")

print(scores)
# Convert scores to a more interpretable format
scores_dict = {}
for i, query in enumerate(queries):
    # Get the page numbers (1-based) where this query has high relevance
    relevant_pages = [(page_idx + 1, score.item()) for page_idx, score in enumerate(scores[i]) if score > 0.5]
    
    # Sort by relevance score in descending order
    relevant_pages.sort(key=lambda x: x[1], reverse=True)
    
    scores_dict[query] = relevant_pages
logger.info("Scores processed and formatted")

# Save the results
import json
from datetime import datetime

# Create a timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"query_results_{timestamp}.json"

# Format the results for saving
results = {
    "pdf_name": "test.pdf",
    "queries": scores_dict
}

# Save to JSON file
logger.info(f"Saving results to {output_filename}")
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=4)

logger.info(f"Results successfully saved to {output_filename}")
print(f"\nResults saved to {output_filename}")



# config = PretrainedConfig.from_dict({
#     "use_bfloat16": False,
# })
# model = ColQwen2.from_pretrained(
#     model_name,
#     # config=config,
#     # torch_dtype=torch.bfloat16,
#     torch_dtype=torch.float32,
#     device_map="mps"
# ).eval()

# model_name = "vidore/colqwen2-v0.1"