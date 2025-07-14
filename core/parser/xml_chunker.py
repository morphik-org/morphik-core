import logging
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from lxml import etree

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False


logger = logging.getLogger(__name__)


class XMLChunker:
    """Schema-agnostic XML chunking that preserves hierarchical structure."""

    def __init__(self, content: bytes, config: Dict[str, Any]):
        """
        Initialize XMLChunker with XML content and configuration.

        Args:
            content: Raw XML file bytes
            config: Configuration dictionary with keys:
                - max_tokens: Maximum tokens per chunk (default: 350)
                - preferred_unit_tags: List of preferred unit tag names
                - ignore_tags: List of tag names to ignore during processing
        """
        self.content = content
        self.max_tokens = config.get("max_tokens", 350)
        self.preferred_unit_tags = config.get("preferred_unit_tags", ["SECTION", "Section", "Article", "clause"])
        self.ignore_tags = config.get("ignore_tags", ["TOC", "INDEX"])

        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self._count_tokens = self._tiktoken_count_tokens
                logger.debug("Using tiktoken for token counting")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}. Falling back to whitespace tokenizer.")
                self._count_tokens = self._whitespace_count_tokens
        else:
            logger.info("tiktoken not available, using whitespace tokenizer")
            self._count_tokens = self._whitespace_count_tokens

    def _tiktoken_count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _whitespace_count_tokens(self, text: str) -> int:
        """Fallback token counter using whitespace splitting."""
        return len(text.split())

    def _profile_tree(self, root: ET.Element) -> Dict[str, int]:
        """Profile the XML tree to count occurrences of each tag."""
        tag_counts = Counter()
        for elem in root.iter():
            tag_counts[elem.tag] += 1
        return dict(tag_counts)

    def _choose_unit_tag(self, tag_profile: Dict[str, int]) -> str:
        """Choose the best unit tag for chunking based on profile and preferences."""
        # First, try preferred unit tags
        for preferred in self.preferred_unit_tags:
            if preferred in tag_profile:
                logger.info(f"Using preferred unit tag: {preferred}")
                return preferred

        # Filter out ignored tags
        filtered_profile = {tag: count for tag, count in tag_profile.items() if tag not in self.ignore_tags}

        if not filtered_profile:
            logger.warning("No suitable tags found after filtering, using root element")
            return list(tag_profile.keys())[0] if tag_profile else "root"

        # Choose tag with reasonable frequency (not too many, not too few)
        sorted_tags = sorted(filtered_profile.items(), key=lambda x: x[1])

        # Prefer tags that appear multiple times but not too frequently
        for tag, count in sorted_tags:
            if 2 <= count <= 50:  # Reasonable range
                logger.info(f"Auto-selected unit tag: {tag} (count: {count})")
                return tag

        # Fallback to most common tag
        most_common_tag = max(filtered_profile.items(), key=lambda x: x[1])[0]
        logger.info(f"Fallback to most common tag: {most_common_tag}")
        return most_common_tag

    def _breadcrumbs(self, elem: ET.Element, root: ET.Element) -> List[str]:
        """Generate breadcrumb path from root to element."""
        path = []
        current = elem

        # Build path from element to root
        while current is not None and current != root:
            # Try to get a meaningful identifier
            elem_id = self._best_xml_id(current)
            if elem_id:
                path.append(f"{current.tag}#{elem_id}")
            else:
                path.append(current.tag)

            # Find parent (ElementTree doesn't have parent references)
            parent = None
            for candidate in root.iter():
                if current in candidate:
                    parent = candidate
                    break
            current = parent

        path.reverse()
        return path

    def _best_xml_id(self, elem: ET.Element) -> Optional[str]:
        """Find the best identifier for an XML element."""
        # Try common ID attributes
        for attr in ["id", "xml:id", "ID", "name", "title"]:
            if attr in elem.attrib:
                return elem.attrib[attr]

        # Try to extract from first text content
        text = self._elem_text(elem)
        if text:
            words = text.split()[:3]  # First 3 words
            if words:
                return "_".join(words).replace(" ", "_")

        return None

    def _elem_text(self, elem: ET.Element, max_length: int = 100) -> str:
        """Extract text content from element, limited to max_length."""
        text_parts = []

        # Get direct text
        if elem.text:
            text_parts.append(elem.text.strip())

        # Get text from immediate children
        for child in elem:
            if child.text:
                text_parts.append(child.text.strip())
            if child.tail:
                text_parts.append(child.tail.strip())

        full_text = " ".join(text_parts).strip()
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "..."

        return full_text

    def _chunkify(self, root: ET.Element, unit_tag: str) -> List[Dict[str, Any]]:
        """Break XML into chunks based on unit tag."""
        chunks = []
        unit_elements = root.findall(f".//{unit_tag}")

        if not unit_elements:
            logger.warning(f"No elements found with tag '{unit_tag}', creating single chunk from root")
            unit_elements = [root]

        logger.info(f"Found {len(unit_elements)} elements with tag '{unit_tag}'")

        for i, elem in enumerate(unit_elements):
            # Extract text content
            text_content = self._elem_text(elem, max_length=10000)  # Larger limit for chunking

            if not text_content.strip():
                continue

            # Check if content fits in token limit
            if self._count_tokens(text_content) <= self.max_tokens:
                # Content fits, create single chunk
                chunk_data = {
                    "text": text_content,
                    "unit": unit_tag,
                    "xml_id": self._best_xml_id(elem),
                    "breadcrumbs": self._breadcrumbs(elem, root),
                    "source_path": f"{unit_tag}[{i}]",
                    "prev": unit_elements[i - 1].attrib.get("id") if i > 0 else None,
                    "next": unit_elements[i + 1].attrib.get("id") if i < len(unit_elements) - 1 else None,
                }
                chunks.append(chunk_data)
            else:
                # Content too large, split recursively
                sub_chunks = self._recursive_split(elem, root, unit_tag, i)
                chunks.extend(sub_chunks)

        logger.info(f"Created {len(chunks)} chunks total")
        return chunks

    def _recursive_split(
        self, elem: ET.Element, root: ET.Element, unit_tag: str, unit_index: int
    ) -> List[Dict[str, Any]]:
        """Recursively split large elements into smaller chunks."""
        chunks = []

        # Try to split by child elements
        if len(elem) > 0:
            current_chunk_text = ""
            chunk_parts = []

            # Add element's direct text if any
            if elem.text and elem.text.strip():
                current_chunk_text = elem.text.strip()

            for child in elem:
                child_text = self._elem_text(child, max_length=10000)

                # Check if adding this child would exceed token limit
                test_text = current_chunk_text + "\n" + child_text if current_chunk_text else child_text

                if self._count_tokens(test_text) <= self.max_tokens:
                    # Add to current chunk
                    if current_chunk_text:
                        current_chunk_text += "\n" + child_text
                    else:
                        current_chunk_text = child_text
                    chunk_parts.append(child)
                else:
                    # Current chunk is full, save it and start new one
                    if current_chunk_text.strip():
                        chunk_data = {
                            "text": current_chunk_text.strip(),
                            "unit": f"{unit_tag}_part",
                            "xml_id": self._best_xml_id(elem),
                            "breadcrumbs": self._breadcrumbs(elem, root),
                            "source_path": f"{unit_tag}[{unit_index}]_part{len(chunks)}",
                            "prev": None,
                            "next": None,
                        }
                        chunks.append(chunk_data)

                    # Start new chunk with current child
                    current_chunk_text = child_text
                    chunk_parts = [child]

            # Add final chunk if any content remains
            if current_chunk_text.strip():
                chunk_data = {
                    "text": current_chunk_text.strip(),
                    "unit": f"{unit_tag}_part",
                    "xml_id": self._best_xml_id(elem),
                    "breadcrumbs": self._breadcrumbs(elem, root),
                    "source_path": f"{unit_tag}[{unit_index}]_part{len(chunks)}",
                    "prev": None,
                    "next": None,
                }
                chunks.append(chunk_data)
        else:
            # No child elements, split text by sentences/paragraphs
            full_text = self._elem_text(elem, max_length=50000)
            text_chunks = self._split_text_by_sentences(full_text)

            for j, text_chunk in enumerate(text_chunks):
                if text_chunk.strip():
                    chunk_data = {
                        "text": text_chunk.strip(),
                        "unit": f"{unit_tag}_text",
                        "xml_id": self._best_xml_id(elem),
                        "breadcrumbs": self._breadcrumbs(elem, root),
                        "source_path": f"{unit_tag}[{unit_index}]_text{j}",
                        "prev": None,
                        "next": None,
                    }
                    chunks.append(chunk_data)

        return chunks

    def _split_text_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks that respect sentence boundaries."""
        # Simple sentence splitting
        sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk(self) -> List[Dict[str, Any]]:
        """
        Main chunking method that processes XML content and returns chunks.

        Returns:
            List of dictionaries, each representing a chunk with keys:
            - text: The chunk content
            - unit: The XML tag used as chunking unit
            - xml_id: Identifier for the XML element
            - breadcrumbs: Hierarchical path to the element
            - source_path: Path indicating location in document
            - prev: ID of previous chunk (if applicable)
            - next: ID of next chunk (if applicable)
        """
        try:
            # Parse XML with recovery mode if lxml is available
            if LXML_AVAILABLE:
                logger.debug("Using lxml parser with recovery mode")
                parser = etree.XMLParser(recover=True)
                tree = etree.fromstring(self.content, parser)
                # Convert lxml element to ElementTree element for compatibility
                xml_string = etree.tostring(tree, encoding="unicode")
                root = ET.fromstring(xml_string)
            else:
                logger.debug("Using standard ElementTree parser")
                root = ET.fromstring(self.content)

            # Profile the XML structure
            tag_profile = self._profile_tree(root)
            logger.info(f"XML tag profile: {tag_profile}")

            # Choose unit tag for chunking
            unit_tag = self._choose_unit_tag(tag_profile)

            # Create chunks
            chunks = self._chunkify(root, unit_tag)

            logger.info(f"Successfully created {len(chunks)} chunks using unit tag '{unit_tag}'")
            return chunks

        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            # Fallback: treat as plain text
            text_content = self.content.decode("utf-8", errors="ignore")
            return [
                {
                    "text": text_content,
                    "unit": "fallback_text",
                    "xml_id": None,
                    "breadcrumbs": [],
                    "source_path": "fallback",
                    "prev": None,
                    "next": None,
                }
            ]
        except Exception as e:
            logger.error(f"Unexpected error during XML chunking: {e}")
            raise
