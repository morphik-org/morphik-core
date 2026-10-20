from core.parser.xml_chunker import XMLChunker


def test_attribute_only_xml_falls_back_to_raw_text():
    content = '\ufeff<Settings>\n  <Option name="retries" value="3" />\n</Settings>'.encode("utf-8")

    chunks = XMLChunker(content=content, config={}).chunk()

    assert len(chunks) == 1
    assert chunks[0]["unit"] == "fallback_text"
    assert 'name="retries"' in chunks[0]["text"]
    assert not chunks[0]["text"].startswith("\ufeff")


def test_xml_with_text_is_chunked_normally():
    content = b"<doc><section><p>Hello world</p></section><section><p>Second part</p></section></doc>"

    chunks = XMLChunker(content=content, config={}).chunk()

    assert chunks
    assert all(chunk["unit"] != "fallback_text" for chunk in chunks)
