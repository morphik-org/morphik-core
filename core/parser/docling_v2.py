import logging
import os
import tempfile
from html import escape as html_escape
from typing import Dict, List, Optional, Tuple

from httpx import AsyncClient, Timeout

logger = logging.getLogger(__name__)

# Lazy imports for docling (heavy dependencies, only needed for local mode)
DocumentConverter = None
InputFormat = None
PdfFormatOption = None
PdfPipelineOptions = None
EasyOcrOptions = None
TableStructureOptions = None
ContentLayer = None
DocItemLabel = None


def _ensure_docling_imports():
    """Lazily import docling dependencies only when needed for local parsing."""
    global DocumentConverter, InputFormat, PdfFormatOption, PdfPipelineOptions
    global EasyOcrOptions, TableStructureOptions, ContentLayer, DocItemLabel

    if DocumentConverter is None:
        from docling.datamodel.base_models import InputFormat as _InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions as _EasyOcrOptions
        from docling.datamodel.pipeline_options import PdfPipelineOptions as _PdfPipelineOptions
        from docling.datamodel.pipeline_options import TableStructureOptions as _TableStructureOptions
        from docling.document_converter import DocumentConverter as _DocumentConverter
        from docling.document_converter import PdfFormatOption as _PdfFormatOption
        from docling_core.types.doc.document import ContentLayer as _ContentLayer
        from docling_core.types.doc.labels import DocItemLabel as _DocItemLabel

        DocumentConverter = _DocumentConverter
        InputFormat = _InputFormat
        PdfFormatOption = _PdfFormatOption
        PdfPipelineOptions = _PdfPipelineOptions
        EasyOcrOptions = _EasyOcrOptions
        TableStructureOptions = _TableStructureOptions
        ContentLayer = _ContentLayer
        DocItemLabel = _DocItemLabel


class DoclingV2Parser:
    """Docling parser that returns page-wise XML chunks with bbox metadata.

    Supports both local parsing (with GPU) and remote API parsing.
    When PARSER_MODE is 'api', documents are sent to GPU servers for processing.
    """

    _docling_converter = None

    def __init__(self, settings=None):
        """Initialize the parser with optional settings for API mode.

        Args:
            settings: Settings object with PARSER_MODE, MORPHIK_EMBEDDING_API_DOMAIN,
                     and MORPHIK_EMBEDDING_API_KEY for API mode configuration.
        """
        self.settings = settings
        self._parse_api_endpoints: Optional[List[str]] = None
        self._parse_api_key: Optional[str] = None

        if settings and getattr(settings, "PARSER_MODE", "local") == "api":
            api_domain = getattr(settings, "MORPHIK_EMBEDDING_API_DOMAIN", None)
            if api_domain:
                if isinstance(api_domain, str):
                    api_domain = [api_domain]
                self._parse_api_endpoints = [f"{d.rstrip('/')}/parse/v2" for d in api_domain]
                self._parse_api_key = getattr(settings, "MORPHIK_EMBEDDING_API_KEY", None)
                logger.info(f"DoclingV2Parser API mode enabled with {len(self._parse_api_endpoints)} endpoint(s)")

    @classmethod
    def _get_converter(cls):
        """Get or create the Docling converter for local parsing."""
        if cls._docling_converter is None:
            _ensure_docling_imports()
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            try:
                import easyocr  # noqa: F401

                pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])
            except ImportError:
                logger.info("EasyOCR not installed; disabling OCR for Docling v2 parser.")
                pipeline_options.do_ocr = False

            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options = TableStructureOptions(mode="accurate")
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_picture_images = True

            cls._docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
        return cls._docling_converter

    @classmethod
    def convert_bytes(cls, file_bytes: bytes, filename: str):
        """Convert a file (bytes) to a Docling document."""
        _ensure_docling_imports()
        suffix = os.path.splitext(filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            converter = cls._get_converter()
            result = converter.convert(temp_path)
            return result.document
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    @staticmethod
    def _label_key(label: object) -> str:
        if hasattr(label, "name"):
            return str(label.name).upper()
        if hasattr(label, "value"):
            return str(label.value).upper()
        return str(label).upper()

    @staticmethod
    def _bbox_to_loc(bbox, page_width: float, page_height: float) -> Optional[str]:
        if bbox is None or page_width <= 0 or page_height <= 0:
            return None

        def _norm(value: float, max_value: float) -> int:
            scaled = (value / max_value) * 500
            return max(0, min(500, int(round(scaled))))

        x1 = _norm(bbox.l, page_width)
        y1 = _norm(bbox.t, page_height)
        x2 = _norm(bbox.r, page_width)
        y2 = _norm(bbox.b, page_height)
        return f"{x1},{y1},{x2},{y2}"

    @classmethod
    def build_page_xml_chunks(
        cls,
        doc,
        document_id: str,
        filename: str,
    ) -> List[Tuple[str, int]]:
        """Build one XML chunk per page with bbox metadata."""
        _ensure_docling_imports()
        label_to_tag: Dict = {}

        def _add(label_name: str, tag: str) -> None:
            label = getattr(DocItemLabel, label_name, None)
            if label is not None:
                label_to_tag[label] = tag

        _add("TEXT", "t")
        _add("PARAGRAPH", "t")
        _add("SECTION_HEADER", "h")
        _add("TITLE", "title")
        _add("PAGE_HEADER", "r")
        _add("PAGE_FOOTER", "f")
        _add("TABLE", "tbl")
        _add("PICTURE", "img")
        _add("CHART", "chart")
        _add("LIST_ITEM", "li")
        _add("CAPTION", "cap")
        _add("FOOTNOTE", "fn")
        _add("FORMULA", "math")
        _add("CODE", "code")
        _add("CHECKBOX_SELECTED", "cb")
        _add("CHECKBOX_UNSELECTED", "cb")
        _add("FORM", "form")
        _add("KEY_VALUE_REGION", "kv")
        _add("REFERENCE", "ref")
        _add("DOCUMENT_INDEX", "idx")
        _add("HANDWRITTEN_TEXT", "hw")

        label_by_name = {cls._label_key(k): v for k, v in label_to_tag.items()}
        table_label = getattr(DocItemLabel, "TABLE", None)
        checkbox_selected_label = getattr(DocItemLabel, "CHECKBOX_SELECTED", None)

        pages: Dict[int, List[str]] = {}

        for item, _level in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
            if not hasattr(item, "prov") or not item.prov:
                continue

            prov = item.prov[0]
            page_no = getattr(prov, "page_no", None)
            if page_no is None:
                continue

            page = None
            pages_obj = getattr(doc, "pages", None)
            if isinstance(pages_obj, dict):
                page = pages_obj.get(page_no)
            elif isinstance(pages_obj, list):
                idx = page_no - 1
                if 0 <= idx < len(pages_obj):
                    page = pages_obj[idx]
            else:
                try:
                    page = pages_obj[page_no]
                except Exception:  # noqa: BLE001
                    page = None
            if not page or not getattr(page, "size", None):
                continue

            bbox = getattr(prov, "bbox", None)
            loc = cls._bbox_to_loc(bbox, page.size.width, page.size.height)

            label = getattr(item, "label", None)
            tag = None
            if label in label_to_tag:
                tag = label_to_tag[label]
            elif label is not None:
                tag = label_by_name.get(cls._label_key(label))
            if not tag:
                tag = "t"

            text = ""
            if table_label is not None and label == table_label and hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc=doc)
                except TypeError:
                    text = item.export_to_markdown()
                except Exception:  # noqa: BLE001
                    text = ""
            elif hasattr(item, "text") and item.text:
                text = item.text
            elif hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc=doc)
                except TypeError:
                    text = item.export_to_markdown()
                except Exception:  # noqa: BLE001
                    text = ""

            text = text or ""
            text = text.strip()
            if not text and tag not in {"img", "chart"}:
                continue

            attr_parts = []
            if loc:
                attr_parts.append(f'loc="{loc}"')

            if tag == "cb":
                checked = "true" if label == checkbox_selected_label else "false"
                attr_parts.append(f'checked="{checked}"')

            attr_str = (" " + " ".join(attr_parts)) if attr_parts else ""

            if tag in {"img", "chart"}:
                # Always self-closing for images/charts - no base64 data
                element = f"<{tag}{attr_str}/>"
            else:
                escaped_text = html_escape(text, quote=False)
                element = f"<{tag}{attr_str}>{escaped_text}</{tag}>"

            pages.setdefault(page_no, []).append(element)

        file_attr = html_escape(filename, quote=True)
        doc_attr = html_escape(document_id, quote=True)
        xml_chunks: List[Tuple[str, int]] = []

        for page_no in sorted(pages.keys()):
            elements = "\n".join(pages[page_no])
            xml = f'<doc id="{doc_attr}" file="{file_attr}">' f'<p n="{page_no}">{elements}</p>' "</doc>"
            xml_chunks.append((xml, page_no))

        return xml_chunks

    async def _parse_via_api(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        display_filename: Optional[str],
    ) -> List[Tuple[str, int]]:
        """Parse document via remote API (GPU server)."""
        if not self._parse_api_endpoints or not self._parse_api_key:
            raise RuntimeError("Parser V2 API not configured")

        headers = {"Authorization": f"Bearer {self._parse_api_key}"}
        timeout = Timeout(read=300.0, connect=30.0, write=60.0, pool=30.0)

        last_error: Optional[Exception] = None
        for endpoint in self._parse_api_endpoints:
            try:
                async with AsyncClient(timeout=timeout) as client:
                    files = {"file": (filename, file_bytes)}
                    data = {"filename": filename, "document_id": document_id}
                    if display_filename and display_filename != filename:
                        data["display_filename"] = display_filename
                    resp = await client.post(endpoint, files=files, data=data, headers=headers)
                    resp.raise_for_status()
                    result = resp.json()

                    # Convert API response to list of (xml, page_number) tuples
                    chunks = result.get("chunks", [])
                    return [(c["xml"], c["page_number"]) for c in chunks]
            except Exception as e:
                logger.warning(f"Parse V2 API call to {endpoint} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All parse V2 API endpoints failed. Last error: {last_error}")

    def _parse_local(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        display_filename: Optional[str],
    ) -> List[Tuple[str, int]]:
        """Parse document using local Docling."""
        doc = self.convert_bytes(file_bytes, filename)
        return self.build_page_xml_chunks(doc, document_id, display_filename or filename)

    async def parse(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        display_filename: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """Parse document and return page-wise XML chunks with bbox metadata.

        Uses API if configured, otherwise falls back to local parsing.

        Args:
            file_bytes: Raw file content
            filename: Original filename
            document_id: Document identifier for the XML

        Returns:
            List of (xml_string, page_number) tuples
        """
        if self._parse_api_endpoints:
            try:
                return await self._parse_via_api(file_bytes, filename, document_id, display_filename)
            except Exception as e:
                logger.warning(f"API parsing failed, falling back to local: {e}")
                return self._parse_local(file_bytes, filename, document_id, display_filename)
        else:
            return self._parse_local(file_bytes, filename, document_id, display_filename)
