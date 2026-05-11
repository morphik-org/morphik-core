import subprocess
from io import BytesIO
from pathlib import Path

from PIL import Image

from core.services import ingestion_service as ingestion_module
from core.services.ingestion_service import IngestionService


class FakePixmap:
    def __init__(self, image_bytes: bytes):
        self._image_bytes = image_bytes

    def tobytes(self, format: str) -> bytes:
        assert format == "png"
        return self._image_bytes


class FakePage:
    def __init__(self, image_bytes: bytes | None = None, error: Exception | None = None):
        self._image_bytes = image_bytes
        self._error = error

    def get_pixmap(self, matrix):
        if self._error:
            raise self._error
        return FakePixmap(self._image_bytes or b"")


class FakeDocument:
    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (12, 12), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _non_blank_png_bytes() -> bytes:
    image = Image.new("RGB", (12, 12), "white")
    image.putpixel((5, 5), (0, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_render_pdf_with_pymupdf_skips_blank_and_failed_pages(monkeypatch):
    service = IngestionService(None, None, None, None, None)
    fake_document = FakeDocument(
        [
            FakePage(_non_blank_png_bytes()),
            FakePage(error=RuntimeError("bad embedded image")),
            FakePage(_png_bytes((255, 255, 255))),
            FakePage(_non_blank_png_bytes()),
        ]
    )

    monkeypatch.setattr(ingestion_module.fitz, "open", lambda *args, **kwargs: fake_document)

    rendered_pages = service._render_pdf_with_pymupdf(b"%PDF", dpi=72, include_bytes=True)

    assert len(rendered_pages) == 2
    assert all(image_b64.startswith("data:image/png;base64,") for image_b64, _ in rendered_pages)
    assert fake_document.closed is True


def test_office_conversion_skips_blank_and_failed_pages(monkeypatch):
    service = IngestionService(None, None, None, None, None)
    fake_document = FakeDocument(
        [
            FakePage(_non_blank_png_bytes()),
            FakePage(error=RuntimeError("bad embedded image")),
            FakePage(_png_bytes((255, 255, 255))),
            FakePage(_non_blank_png_bytes()),
        ]
    )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/soffice" if name == "soffice" else None)
    monkeypatch.setattr(ingestion_module.fitz, "open", lambda *args, **kwargs: fake_document)

    def fake_run(cmd, capture_output, text, timeout):
        output_dir = Path(cmd[cmd.index("--outdir") + 1])
        input_path = Path(cmd[-1])
        expected_pdf_path = output_dir / f"{input_path.stem}.pdf"
        expected_pdf_path.write_bytes(b"%PDF")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    chunks = service._convert_office_to_images(b"pptx-bytes", ".pptx", "PowerPoint presentation", [])

    assert len(chunks) == 2
    assert all(chunk.metadata["is_image"] is True for chunk in chunks)
    assert all(chunk.content.startswith("data:image/png;base64,") for chunk in chunks)
    assert fake_document.closed is True
