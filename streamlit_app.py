import streamlit as st
import os
from pathlib import Path
import shutil
from pdf_image_extractor import convert_to_pdf, extract_images_with_context
import base64
from ollama import AsyncClient
import asyncio


st.set_page_config(
    page_title="Document Image Extractor",
    page_icon="📄",
    layout="wide"
)


async def get_image_description(image_path: str, client: AsyncClient, document_text: str) -> str:
    """Get image description using Ollama's Llama vision model"""
    try:
        # Convert image to base64
        print("Getting image description")
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode()
        
        # Get description from Ollama with document context
        response = await client.chat(
            model="llama3.2-vision",
            messages=[{
                "role": "user",
                "content": f"""Given this document context, describe this image in detail:

Document Context:
{document_text}

Please describe the image, focusing on its relevance to the document context and any visible elements or text.""",
                "images": [image_base64]
            }]
        )
        print(f"Processed image {response["message"]["content"]}")
        return response["message"]["content"]
    except Exception as e:
        return f"Error getting image description: {str(e)}"


def get_image_base64(image_path):
    """Convert image to base64 for display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def save_text_context(output_dir: Path, image_info: list, full_text: str):
    """Save contextual information to a text file"""
    context_file = output_dir / "context.txt"
    with open(context_file, "w", encoding="utf-8") as f:
        # First write the full document text
        f.write("Full Document Text:\n")
        f.write("=" * 80 + "\n\n")
        f.write(full_text)
        f.write("\n\n")
        f.write("=" * 80 + "\n\n")
        f.write("Images and Their Context:\n")
        f.write("=" * 80 + "\n\n")
        
        for img in image_info:
            f.write(f"\nImage: {Path(img['image_path']).name}\n")
            f.write(f"Page: {img['page_number']}\n")
            f.write("Context:\n")
            f.write(f"{img['context']}\n")
            f.write("-" * 80 + "\n")


async def process_file(uploaded_file, output_base_dir):
    """Process an uploaded file and return the results"""
    # Create a unique directory for this file
    file_name = Path(uploaded_file.name)
    output_dir = Path(output_base_dir) / file_name.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file temporarily
    temp_path = output_dir / file_name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Convert to PDF if needed
        if not file_name.suffix.lower() == '.pdf':
            pdf_path = convert_to_pdf(temp_path, output_dir)
        else:
            pdf_path = str(temp_path)
        
        # Extract images and context
        extraction_result = extract_images_with_context(pdf_path, str(output_dir))
        image_info = extraction_result["images"]
        full_text = extraction_result["full_text"]
        
        # Initialize Ollama client
        client = AsyncClient(host="http://localhost:11434")
        
        # Get descriptions for each image
        for img in image_info:
            img['description'] = await get_image_description(img['image_path'], client, full_text)
            break
        # Save full document text
        with open(output_dir / "full_text.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        
        return {
            "success": True,
            "output_dir": str(output_dir),
            "num_images": len(image_info),
            "images": image_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temporary file if it's not a PDF
        if not file_name.suffix.lower() == '.pdf':
            temp_path.unlink(missing_ok=True)


async def main():
    st.title("📄 IP Author Document portal")
    st.write("Upload documents to extract images and their contextual information.")
    
    # Create base output directory
    output_base_dir = Path("streamlit_output")
    output_base_dir.mkdir(exist_ok=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, etc.)",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "pptx", "ppt"]
    )
    
    if uploaded_files:
        st.write("---")
        st.subheader("Processing Results")
        
        # Process each file
        for uploaded_file in uploaded_files:
            with st.expander(f"📁 {uploaded_file.name}", expanded=True):
                result = await process_file(uploaded_file, output_base_dir)
                
                if result["success"]:
                    st.success(f"Successfully extracted {result['num_images']} images")
                    
                    # Display images and descriptions
                    output_dir = Path(result["output_dir"])
                    
                    # Create columns for the layout
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Extracted Images")
                        for img_info in result["images"]:
                            img_path = img_info["image_path"]
                            if os.path.exists(img_path):
                                st.image(
                                    img_path,
                                    caption=f"Page {img_info['page_number']}",
                                    use_container_width=True
                                )
                                st.markdown("**Image Description:**")
                                st.write(img_info.get('description', 'No description available'))
                                st.markdown("---")

                    with col2:
                        st.subheader("Document Text")
                        with open(output_dir / "full_text.txt") as f:
                            st.text_area(
                                "Full Document Content",
                                f.read(),
                                height=800
                            )

                    # Download buttons
                    st.download_button(
                        "📥 Download All Results",
                        data=shutil.make_archive(
                            str(output_dir),
                            'zip',
                            str(output_dir)
                        ),
                        file_name=f"{uploaded_file.name}_results.zip",
                        mime="application/zip"
                    )
                else:
                    st.error(f"Error processing file: {result['error']}")

        # Cleanup option
        if st.button("🗑️ Clear All Results"):
            shutil.rmtree(output_base_dir)
            output_base_dir.mkdir(exist_ok=True)
            st.success("All results cleared!")
            st.experimental_rerun()


if __name__ == "__main__":
    asyncio.run(main())
