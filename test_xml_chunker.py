#!/usr/bin/env python3
"""
Test script for XML chunking functionality.

This script tests the XMLChunker implementation and integration with MorphikParser.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example XML content for testing
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <SECTION id="intro">
        <title>Introduction</title>
        <content>This is the introduction section with some important information about XML processing.</content>
    </SECTION>
    <SECTION id="methodology">
        <title>Methodology</title>
        <content>This section describes the methodology used for schema-agnostic XML chunking. It includes details about element selection and hierarchical preservation.</content>
        <subsection>
            <title>Technical Approach</title>
            <content>We use lxml for robust XML parsing with recovery mode to handle malformed documents.</content>
        </subsection>
    </SECTION>
    <SECTION id="results">
        <title>Results</title>
        <content>The results show that our approach successfully chunks XML documents while preserving their hierarchical structure and maintaining semantic context through breadcrumbs.</content>
    </SECTION>
    <TOC>
        <entry>Introduction</entry>
        <entry>Methodology</entry>
        <entry>Results</entry>
    </TOC>
</document>"""


async def test_xml_chunker_standalone():
    """Test XMLChunker directly."""
    logger.info("Testing XMLChunker standalone...")
    
    try:
        from core.parser.xml_chunker import XMLChunker
        
        # Test configuration
        config = {
            "max_tokens": 350,
            "preferred_unit_tags": ["SECTION", "Section", "Article", "clause"],
            "ignore_tags": ["TOC", "INDEX"]
        }
        
        # Create chunker and process
        chunker = XMLChunker(content=SAMPLE_XML.encode('utf-8'), config=config)
        chunks = chunker.chunk()
        
        logger.info(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"\n--- Chunk {i+1} ---")
            logger.info(f"Unit: {chunk.get('unit')}")
            logger.info(f"XML ID: {chunk.get('xml_id')}")
            logger.info(f"Breadcrumbs: {chunk.get('breadcrumbs')}")
            logger.info(f"Source Path: {chunk.get('source_path')}")
            logger.info(f"Token Count: {chunk.get('token_count')}")
            logger.info(f"Content: {chunk.get('text')[:100]}...")
            if chunk.get('prev'):
                logger.info(f"Previous: {chunk.get('prev')}")
            if chunk.get('next'):
                logger.info(f"Next: {chunk.get('next')}")
        
        logger.info("✅ XMLChunker standalone test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ XMLChunker standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_morphik_parser_integration():
    """Test XMLChunker integration with MorphikParser."""
    logger.info("Testing MorphikParser integration...")
    
    try:
        from core.config import get_settings
        from core.parser.morphik_parser import MorphikParser
        
        # Get settings
        settings = get_settings()
        
        # Create parser with settings
        parser = MorphikParser(
            chunk_size=1000,
            chunk_overlap=200,
            use_unstructured_api=False,
            settings=settings
        )
        
        # Test XML file detection
        xml_content = SAMPLE_XML.encode('utf-8')
        filename = "test_document.xml"
        
        # Check XML detection
        is_xml = parser._is_xml_file(xml_content, filename)
        logger.info(f"XML detection result: {is_xml}")
        
        if not is_xml:
            logger.error("❌ XML file detection failed")
            return False
        
        # Test parsing
        metadata, text = await parser.parse_file_to_text(xml_content, filename)
        logger.info(f"Parsed text length: {len(text)}")
        logger.info(f"Metadata keys: {list(metadata.keys())}")
        
        if not metadata.get('is_xml'):
            logger.error("❌ XML metadata not set correctly")
            return False
        
        # Test chunking
        chunks = await parser.split_text(text, metadata=metadata)
        logger.info(f"Created {len(chunks)} Chunk objects")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"\n--- Morphik Chunk {i+1} ---")
            logger.info(f"Content: {chunk.content[:100]}...")
            logger.info(f"Chunk Number: {chunk.chunk_number}")
            logger.info(f"Metadata keys: {list(chunk.metadata.keys())}")
            if chunk.metadata.get('unit'):
                logger.info(f"Unit: {chunk.metadata.get('unit')}")
            if chunk.metadata.get('xml_id'):
                logger.info(f"XML ID: {chunk.metadata.get('xml_id')}")
        
        logger.info("✅ MorphikParser integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ MorphikParser integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_non_xml_files_unaffected():
    """Test that non-XML files continue to work normally."""
    logger.info("Testing non-XML files remain unaffected...")
    
    try:
        from core.config import get_settings
        from core.parser.morphik_parser import MorphikParser
        
        # Get settings
        settings = get_settings()
        
        # Create parser
        parser = MorphikParser(
            chunk_size=1000,
            chunk_overlap=200,
            use_unstructured_api=False,
            settings=settings
        )
        
        # Test with plain text
        text_content = b"This is a simple text document. It should be processed normally without XML chunking."
        filename = "test_document.txt"
        
        # Check XML detection (should be False)
        is_xml = parser._is_xml_file(text_content, filename)
        logger.info(f"XML detection for text file: {is_xml}")
        
        if is_xml:
            logger.error("❌ Text file incorrectly detected as XML")
            return False
        
        # Test parsing
        metadata, text = await parser.parse_file_to_text(text_content, filename)
        logger.info(f"Text file parsed successfully: {len(text)} characters")
        
        if metadata.get('is_xml'):
            logger.error("❌ Text file incorrectly marked as XML")
            return False
        
        # Test chunking
        chunks = await parser.split_text(text, metadata=metadata)
        logger.info(f"Text file chunked normally: {len(chunks)} chunks")
        
        logger.info("✅ Non-XML files test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Non-XML files test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_malformed_xml():
    """Test handling of malformed XML."""
    logger.info("Testing malformed XML handling...")
    
    try:
        from core.parser.xml_chunker import XMLChunker
        
        # Malformed XML (missing closing tag)
        malformed_xml = """<?xml version="1.0"?>
        <document>
            <section>This section is missing its closing tag
            <section>This is another section</section>
        </document>"""
        
        config = {
            "max_tokens": 350,
            "preferred_unit_tags": ["section"],
            "ignore_tags": []
        }
        
        # Should handle gracefully with lxml recovery mode
        chunker = XMLChunker(content=malformed_xml.encode('utf-8'), config=config)
        chunks = chunker.chunk()
        
        logger.info(f"Malformed XML produced {len(chunks)} chunks")
        
        if len(chunks) == 0:
            logger.error("❌ Malformed XML handling failed - no chunks produced")
            return False
        
        logger.info("✅ Malformed XML test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Malformed XML test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration():
    """Test XML configuration loading."""
    logger.info("Testing XML configuration...")
    
    try:
        from core.config import get_settings
        
        settings = get_settings()
        
        # Check if PARSER_XML exists and has correct defaults
        if not hasattr(settings, 'PARSER_XML'):
            logger.error("❌ PARSER_XML not found in settings")
            return False
        
        xml_config = settings.PARSER_XML
        logger.info(f"XML config max_tokens: {xml_config.max_tokens}")
        logger.info(f"XML config preferred_unit_tags: {xml_config.preferred_unit_tags}")
        logger.info(f"XML config ignore_tags: {xml_config.ignore_tags}")
        
        # Verify expected values
        if xml_config.max_tokens != 350:
            logger.error(f"❌ Expected max_tokens=350, got {xml_config.max_tokens}")
            return False
        
        if "SECTION" not in xml_config.preferred_unit_tags:
            logger.error(f"❌ Expected 'SECTION' in preferred_unit_tags")
            return False
        
        if "TOC" not in xml_config.ignore_tags:
            logger.error(f"❌ Expected 'TOC' in ignore_tags")
            return False
        
        logger.info("✅ Configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("🧪 Starting XML chunking tests...\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("XMLChunker Standalone", test_xml_chunker_standalone),
        ("MorphikParser Integration", test_morphik_parser_integration),
        ("Non-XML Files Unaffected", test_non_xml_files_unaffected),
        ("Malformed XML Handling", test_malformed_xml),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! XML chunking implementation is working correctly.")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    asyncio.run(main())