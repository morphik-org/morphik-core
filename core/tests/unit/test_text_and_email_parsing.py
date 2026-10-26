"""Source code, config files, and emails (.eml, Outlook .msg) are indexed as text instead of going to Docling."""

import codecs
from datetime import datetime, timezone
from email.message import EmailMessage
from types import SimpleNamespace

import pytest

from core.parser import morphik_parser
from core.parser.email_parser import msg_fields, msg_fields_from_bytes, render_email
from core.parser.morphik_parser import MorphikParser
from core.parser.text_files import decode_text

pytestmark = pytest.mark.unit

SQL = "SELECT TOP 10 * FROM dbo.Interactions WHERE Agent = N'Łukasz';\n"


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(
        morphik_parser, "get_settings", lambda: SimpleNamespace(REGISTERED_MODELS={}, PARSER_MODE="local")
    )
    parser = MorphikParser()

    async def docling(file, filename):
        pytest.fail(f"{filename} should not reach Docling")

    parser._parse_document_local = docling
    parser._parse_document_local_deep = docling
    return parser


@pytest.mark.parametrize(
    "data",
    [
        codecs.BOM_UTF16_LE + SQL.encode("utf-16-le"),  # SQL Server Management Studio
        codecs.BOM_UTF16_BE + SQL.encode("utf-16-be"),
        codecs.BOM_UTF8 + SQL.encode("utf-8"),
        SQL.encode("utf-8"),
    ],
)
def test_text_is_decoded_by_its_bom_then_as_utf8(data):
    assert decode_text(data) == SQL


def test_text_that_is_not_utf8_is_read_as_windows_1252():
    assert decode_text("Café – résumé".encode("cp1252")) == "Café – résumé"


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["report.sql", "etl.py", "deploy.ps1", "app.config.ts", "site.css", "db.ini"])
async def test_code_and_config_files_are_read_as_text(parser, filename):
    data = codecs.BOM_UTF16_LE + SQL.encode("utf-16-le")

    assert await parser.parse_file_to_text(data, filename) == ({}, SQL)
    assert (await parser.parse_file_to_text_deep(data, filename))[1] == SQL


@pytest.mark.asyncio
async def test_typescript_is_not_mistaken_for_mpeg_transport_stream_video(parser, monkeypatch):
    monkeypatch.setattr(parser, "_parse_video", lambda *a: pytest.fail("parsed as video"))

    source = b"export const answer = 42;\n"

    assert await parser.parse_file_to_text(source, "answer.ts") == ({}, source.decode())
    assert await parser.parse_file_to_text_deep(source, "answer.ts") == ({}, source.decode())


def _eml(html_only=False) -> bytes:
    message = EmailMessage()
    message["From"] = "Jane Example <jane@example.com>"
    message["To"] = "QA Team <qa@example.com>"
    message["Cc"] = "dev@example.com"
    message["Date"] = "Fri, 23 Oct 2026 10:53:32 +0000"
    message["Subject"] = "Repro steps for 47490"
    html = "<html><head><style>p {color: red}</style></head><body><p>Open the queue.</p><p>It hangs.</p></body></html>"
    if html_only:
        message.set_content(html, subtype="html")
    else:
        message.set_content("Open the queue.\nIt hangs.\n")
        message.add_alternative(html, subtype="html")
    message.add_attachment(b"%PDF-1.4", maintype="application", subtype="pdf", filename="trace.pdf")
    return message.as_bytes()


EXPECTED_EMAIL = (
    "From: Jane Example <jane@example.com>\n"
    "To: QA Team <qa@example.com>\n"
    "Cc: dev@example.com\n"
    "Date: Fri, 23 Oct 2026 10:53:32 +0000\n"
    "Subject: Repro steps for 47490\n"
    "\n"
    "Open the queue.\n"
    "It hangs.\n"
    "\n"
    "Attachments:\n"
    "- trace.pdf\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("html_only", [False, True])
async def test_eml_is_rendered_as_headers_body_and_attachment_names(parser, html_only):
    metadata, text = await parser.parse_file_to_text(_eml(html_only), "repro.eml")

    assert metadata == {}
    assert text.replace("Open the queue.\n\nIt hangs.", "Open the queue.\nIt hangs.") == EXPECTED_EMAIL


def _utf16(value: str) -> bytes:
    return value.encode("utf-16-le")


def _properties_with_sent_time(sent: datetime) -> bytes:
    filetime = int((sent - datetime(1601, 1, 1, tzinfo=timezone.utc)).total_seconds() * 10_000_000)
    entry = (0x00390040).to_bytes(4, "little") + (6).to_bytes(4, "little") + filetime.to_bytes(8, "little")
    return bytes(32) + entry


def test_msg_properties_render_like_an_eml():
    streams = {
        "__substg1.0_0037001F": _utf16("Repro steps for 47490"),
        "__substg1.0_0C1A001F": _utf16("Jane Example"),
        # An Exchange address is replaced by the SMTP address next to it
        "__substg1.0_0C1F001F": _utf16("/O=EXCHANGELABS/OU=EXCHANGE ADMINISTRATIVE GROUP/CN=RECIPIENTS/CN=JANE"),
        "__substg1.0_5D01001F": _utf16("jane@example.com"),
        "__substg1.0_0E04001F": _utf16("QA Team <qa@example.com>"),
        "__substg1.0_0E03001F": _utf16("dev@example.com"),
        "__substg1.0_10130102": b"<p>Open the queue.</p><p>It hangs.</p>",
        "__attach_version1.0_#00000000/__substg1.0_3704001F": _utf16("TRACE~1.PDF"),
        "__attach_version1.0_#00000000/__substg1.0_3707001F": _utf16("trace.pdf"),
        "__recip_version1.0_#00000000/__substg1.0_3001001F": _utf16("QA Team"),
    }
    properties = _properties_with_sent_time(datetime(2026, 10, 23, 10, 53, 32, tzinfo=timezone.utc))

    text = render_email(msg_fields(streams, properties))

    assert text.replace("Open the queue.\n\nIt hangs.", "Open the queue.\nIt hangs.") == EXPECTED_EMAIL


def test_msg_prefers_the_plain_body_and_reads_ansi_strings():
    fields = msg_fields(
        {
            "__substg1.0_0037001E": "Café order".encode("cp1252"),
            "__substg1.0_1000001F": _utf16("Plain body\x00"),
            "__substg1.0_10130102": b"<p>HTML body</p>",
        }
    )

    assert (fields.subject, fields.body, fields.date) == ("Café order", "Plain body", "")


@pytest.mark.asyncio
async def test_a_file_that_is_not_a_msg_fails_with_a_reason(parser):
    with pytest.raises(ValueError, match="not an OLE compound file"):
        await parser.parse_file_to_text(b"From: someone\n\nplain text", "note.msg")


def test_msg_bytes_are_read_through_olefile(monkeypatch):
    class FakeOle:
        def __init__(self, data):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def listdir(self, streams=True, storages=False):
            return [
                ["__substg1.0_0037001F"],
                ["__attach_version1.0_#00000000", "__substg1.0_37010102"],  # attachment data is never read
                ["__attach_version1.0_#00000000", "__substg1.0_3707001F"],
            ]

        def openstream(self, path):
            if path == ["__attach_version1.0_#00000000", "__substg1.0_37010102"]:
                pytest.fail("attachment bytes were read")
            values = {"__substg1.0_0037001F": _utf16("Subject only"), "__substg1.0_3707001F": _utf16("log.zip")}
            return SimpleNamespace(read=lambda: values[path[-1]])

        def exists(self, path):
            return False

    from core.parser import email_parser

    monkeypatch.setattr(email_parser.olefile, "isOleFile", lambda data: True)
    monkeypatch.setattr(email_parser.olefile, "OleFileIO", FakeOle)

    fields = msg_fields_from_bytes(b"ole bytes")

    assert (fields.subject, fields.attachments) == ("Subject only", ["log.zip"])
