from typing import List
import torch
from PIL import Image
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="mps", #"cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="eager", #"flash_attention_2" if is_flash_attn_2_available() else None,  # or "eager" if "mps"
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

def convert_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert a PDF to a list of images."""
    images = convert_from_path(pdf_path, dpi=100)
    return images

images = convert_to_images("test.pdf")

queries = [
    "Tell me about the adaptive filter compensation model.",
    "at what frequency, in megahertz, do we obtain the highest IRR or image rejection ratio?",
]

# Process the queries
batch_queries = processor.process_queries(queries).to(model.device)

# Process images and get embeddings one by one
image_embeddings_list = []
with torch.no_grad():
    query_embeddings = model(**batch_queries)
    
    for image in images:
        # Process single image
        processed_image = processor.process_images([image]).to(model.device)
        image_embedding = model(**processed_image)
        image_embeddings_list.append(image_embedding)

# Concatenate all image embeddings
image_embeddings = torch.cat(image_embeddings_list, dim=0)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
scores = scores.argmax(dim=1)

print(scores)
