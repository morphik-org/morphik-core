"""
XML Chunker Module

This module provides schema-agnostic XML chunking functionality that:
- Intelligently identifies the most logical "unit" for chunking within any XML file
- Preserves document hierarchy by including "breadcrumbs" (ancestor headings)
- Respects configurable token limits with recursive splitting
- Handles malformed XML gracefully using lxml's recovery mode
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from lxml import etree, html

logger = logging.getLogger(__name__)


class XMLChunker:
    """Schema-agnostic XML chunker that preserves hierarchical structure"""

    def __init__(self, content: bytes, config: Dict[str, Any]):
        """
        Initialize the XMLChunker.
        
        Args:
            content: XML content as bytes
            config: Configuration dictionary with keys:
                - max_tokens: Maximum tokens per chunk (default: 350)
                - preferred_unit_tags: List of preferred tag names for chunking
                - ignore_tags: List of tag names to ignore during processing
        """
        self.content = content
        self.max_tokens = config.get("max_tokens", 350)
        self.preferred_unit_tags = config.get("preferred_unit_tags", ["SECTION", "Section", "Article", "clause"])
        self.ignore_tags = config.get("ignore_tags", ["TOC", "INDEX"])
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        # Parse XML document
        self.root = self._parse_xml(content)
        
        if self.root is None:
            raise ValueError("Failed to parse XML content")

    def _init_tokenizer(self):
        """Initialize tokenizer with tiktoken if available, otherwise use whitespace tokenizer"""
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("o200k_base")
            self._count_tokens = lambda text: len(self.tokenizer.encode(text))
            logger.info("Using tiktoken tokenizer for XML chunking")
        except ImportError:
            logger.warning("tiktoken not available, using whitespace tokenizer for XML chunking")
            self._count_tokens = lambda text: len(text.split())

    def _parse_xml(self, content: bytes) -> Optional[etree._Element]:
        """Parse XML content with error recovery"""
        try:
            # Try parsing as XML first
            parser = etree.XMLParser(recover=True, strip_cdata=False)
            root = etree.fromstring(content, parser)
            logger.info("Successfully parsed content as XML")
            return root
        except Exception as e:
            logger.warning(f"XML parsing failed: {e}, trying HTML parser")
            try:
                # Fallback to HTML parser
                root = html.fromstring(content)
                logger.info("Successfully parsed content as HTML")
                return root
            except Exception as e2:
                logger.error(f"Both XML and HTML parsing failed: XML: {e}, HTML: {e2}")
                return None

    def _profile_tree(self, element: etree._Element) -> Dict[str, int]:
        """Profile the XML tree to understand its structure"""
        tag_counts = Counter()
        
        def count_tags(elem):
            if elem.tag and elem.tag not in self.ignore_tags:
                tag_counts[elem.tag] += 1
            for child in elem:
                count_tags(child)
        
        count_tags(element)
        return dict(tag_counts)

    def _choose_unit_tag(self, element: etree._Element) -> str:
        """Choose the best tag to use as chunking unit"""
        profile = self._profile_tree(element)
        
        # First, try preferred unit tags
        for preferred in self.preferred_unit_tags:
            if preferred in profile and profile[preferred] > 0:
                logger.info(f"Using preferred unit tag: {preferred}")
                return preferred
        
        # If no preferred tags found, use the most common tag with reasonable frequency
        if profile:
            # Filter out very common tags (likely structural)
            filtered_profile = {tag: count for tag, count in profile.items() 
                              if count > 1 and count < len(list(element.iter())) // 2}
            
            if filtered_profile:
                chosen_tag = max(filtered_profile, key=filtered_profile.get)
                logger.info(f"Using most common suitable tag: {chosen_tag}")
                return chosen_tag
        
        # Fallback to any child tag
        for child in element:
            if child.tag and child.tag not in self.ignore_tags:
                logger.info(f"Using fallback tag: {child.tag}")
                return child.tag
        
        # Ultimate fallback
        logger.warning("No suitable unit tag found, using element tag itself")
        return element.tag

    def _get_breadcrumbs(self, element: etree._Element) -> List[str]:
        """Get breadcrumb trail for an element"""
        breadcrumbs = []
        current = element.getparent()
        
        while current is not None:
            # Look for text content that could serve as a heading
            text = self._elem_text(current, include_children=False).strip()
            if text and len(text) < 200:  # Reasonable heading length
                breadcrumbs.append(text)
            elif current.tag:
                # Use tag name if no suitable text
                breadcrumbs.append(f"<{current.tag}>")
            
            current = current.getparent()
        
        breadcrumbs.reverse()
        return breadcrumbs

    def _best_xml_id(self, element: etree._Element) -> str:
        """Get the best identifier for an XML element"""
        # Try various attributes for ID
        for attr in ['id', 'xml:id', 'name', 'ref']:
            if attr in element.attrib:
                return element.attrib[attr]
        
        # Generate ID based on position
        parent = element.getparent()
        if parent is not None:
            siblings = [child for child in parent if child.tag == element.tag]
            try:
                index = siblings.index(element)
                return f"{element.tag}_{index}"
            except ValueError:
                pass
        
        return f"{element.tag}_unknown"

    def _elem_text(self, element: etree._Element, include_children: bool = True) -> str:
        """Extract text content from an element"""
        if include_children:
            # Get all text including from child elements
            return etree.tostring(element, method="text", encoding="unicode") or ""
        else:
            # Get only direct text content
            return (element.text or "") + "".join(element.itertext())

    def _chunkify(self, elements: List[etree._Element], unit_tag: str) -> List[Dict[str, Any]]:
        """Convert elements into chunks with metadata"""
        chunks = []
        
        for i, elem in enumerate(elements):
            text = self._elem_text(elem).strip()
            if not text:
                continue
            
            token_count = self._count_tokens(text)
            
            # If element is too large, try to split it recursively
            if token_count > self.max_tokens:
                text = self._recursive_split(elem, text)
            
            # Create chunk with metadata
            chunk = {
                "text": text,
                "unit": unit_tag,
                "xml_id": self._best_xml_id(elem),
                "breadcrumbs": self._get_breadcrumbs(elem),
                "source_path": self._get_element_path(elem),
                "token_count": self._count_tokens(text)
            }
            
            # Add navigation links
            if i > 0:
                prev_elem = elements[i - 1]
                chunk["prev"] = self._best_xml_id(prev_elem)
            
            if i < len(elements) - 1:
                next_elem = elements[i + 1]
                chunk["next"] = self._best_xml_id(next_elem)
            
            chunks.append(chunk)
        
        return chunks

    def _recursive_split(self, element: etree._Element, text: str) -> str:
        """Recursively split large elements while respecting structure"""
        # If element has children, try splitting by children
        children = list(element)
        if children:
            child_texts = []
            current_chunk = ""
            
            for child in children:
                child_text = self._elem_text(child).strip()
                if not child_text:
                    continue
                
                # Check if adding this child would exceed token limit
                test_chunk = current_chunk + "\n\n" + child_text if current_chunk else child_text
                if self._count_tokens(test_chunk) > self.max_tokens and current_chunk:
                    # Save current chunk and start new one
                    child_texts.append(current_chunk)
                    current_chunk = child_text
                else:
                    current_chunk = test_chunk
            
            # Add final chunk
            if current_chunk:
                child_texts.append(current_chunk)
            
            # Return the first chunk that fits, or truncated text
            if child_texts:
                return child_texts[0]
        
        # Fallback: truncate text to fit token limit
        words = text.split()
        truncated = ""
        for word in words:
            test_text = truncated + " " + word if truncated else word
            if self._count_tokens(test_text) > self.max_tokens:
                break
            truncated = test_text
        
        return truncated or text[:self.max_tokens]  # Ensure we return something

    def _get_element_path(self, element: etree._Element) -> str:
        """Get XPath-like path for an element"""
        path_parts = []
        current = element
        
        while current is not None:
            if current.tag:
                # Count position among siblings with same tag
                parent = current.getparent()
                if parent is not None:
                    siblings = [child for child in parent if child.tag == current.tag]
                    try:
                        index = siblings.index(current)
                        path_parts.append(f"{current.tag}[{index}]")
                    except ValueError:
                        path_parts.append(current.tag)
                else:
                    path_parts.append(current.tag)
            
            current = current.getparent()
        
        path_parts.reverse()
        return "/" + "/".join(path_parts)

    def chunk(self) -> List[Dict[str, Any]]:
        """
        Main chunking method that returns a list of chunk dictionaries.
        
        Returns:
            List of dictionaries where each dictionary represents a chunk with keys:
            - text: The chunk content
            - unit: The XML tag used for chunking
            - xml_id: Identifier for the chunk
            - breadcrumbs: List of ancestor headings
            - source_path: XPath-like path to the element
            - token_count: Number of tokens in the chunk
            - prev: ID of previous chunk (if exists)
            - next: ID of next chunk (if exists)
        """
        if self.root is None:
            logger.error("No valid XML root element to chunk")
            return []
        
        try:
            # Choose the unit tag for chunking
            unit_tag = self._choose_unit_tag(self.root)
            
            # Find all elements with the chosen unit tag
            unit_elements = self.root.xpath(f".//{unit_tag}")
            
            if not unit_elements:
                # Fallback: use direct children of root
                unit_elements = list(self.root)
                unit_tag = "child_element"
            
            logger.info(f"Found {len(unit_elements)} elements with tag '{unit_tag}' for chunking")
            
            # Convert elements to chunks
            chunks = self._chunkify(unit_elements, unit_tag)
            
            logger.info(f"Created {len(chunks)} chunks from XML content")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during XML chunking: {e}")
            # Fallback: create a single chunk with all text
            full_text = self._elem_text(self.root)
            if self._count_tokens(full_text) > self.max_tokens:
                full_text = self._recursive_split(self.root, full_text)
            
            return [{
                "text": full_text,
                "unit": "fallback",
                "xml_id": "root",
                "breadcrumbs": [],
                "source_path": "/root",
                "token_count": self._count_tokens(full_text)
            }]