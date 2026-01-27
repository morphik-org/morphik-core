import logging
import os
import tempfile
from html import escape as html_escape
from typing import Dict, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel

logger = logging.getLogger(__name__)


class DoclingV2Parser:
    """Docling parser that returns page-wise XML chunks with bbox metadata."""

    _docling_converter: Optional[DocumentConverter] = None

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        if cls._docling_converter is None:
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
        label_to_tag: Dict[DocItemLabel, str] = {}

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
