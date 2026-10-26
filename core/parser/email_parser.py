"""Email files (.eml and Outlook .msg) rendered as text: headers, body, and the names of attached files."""

import email
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import policy
from email.utils import format_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

import olefile

from core.parser.text_files import decode_text

EMAIL_EXTENSIONS = {".eml", ".msg"}

# MAPI property IDs read from an Outlook .msg file
PR_SUBJECT = "0037"
PR_SENDER_NAME = "0C1A"
PR_SENDER_EMAIL_ADDRESS = "0C1F"
PR_SENDER_SMTP_ADDRESS = "5D01"
PR_DISPLAY_TO = "0E04"
PR_DISPLAY_CC = "0E03"
PR_BODY = "1000"
PR_HTML = "1013"
PR_ATTACH_LONG_FILENAME = "3707"
PR_ATTACH_FILENAME = "3704"
_MSG_PROPERTIES = {
    PR_SUBJECT,
    PR_SENDER_NAME,
    PR_SENDER_EMAIL_ADDRESS,
    PR_SENDER_SMTP_ADDRESS,
    PR_DISPLAY_TO,
    PR_DISPLAY_CC,
    PR_BODY,
    PR_HTML,
    PR_ATTACH_LONG_FILENAME,
    PR_ATTACH_FILENAME,
}
# PR_CLIENT_SUBMIT_TIME as a PT_SYSTIME tag, then PR_MESSAGE_DELIVERY_TIME for drafts that were never sent
_MSG_TIME_TAGS = (0x00390040, 0x0E060040)
_STREAM_NAME = re.compile(r"__substg1\.0_([0-9A-F]{4})([0-9A-F]{4})$", re.I)
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


@dataclass
class EmailFields:
    sender: str = ""
    to: str = ""
    cc: str = ""
    date: str = ""
    subject: str = ""
    body: str = ""
    attachments: List[str] = field(default_factory=list)


def is_email_filename(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in EMAIL_EXTENSIONS


def parse_email(data: bytes, filename: str) -> str:
    if Path(filename).suffix.lower() == ".msg":
        return render_email(msg_fields_from_bytes(data))
    return render_email(eml_fields(data))


def render_email(fields: EmailFields) -> str:
    lines = [
        f"{label}: {value}"
        for label, value in (
            ("From", fields.sender),
            ("To", fields.to),
            ("Cc", fields.cc),
            ("Date", fields.date),
            ("Subject", fields.subject),
        )
        if value
    ]
    parts = ["\n".join(lines), fields.body.strip()]
    if fields.attachments:
        parts.append("Attachments:\n" + "\n".join(f"- {name}" for name in fields.attachments))
    return "\n\n".join(part for part in parts if part) + "\n"


class _HtmlText(HTMLParser):
    _BLOCKS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "blockquote", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _HtmlText()
    parser.feed(html)
    parser.close()
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _part_text(part) -> str:
    try:
        return part.get_content()
    except (LookupError, UnicodeError):
        # Unknown or wrong charset label
        return decode_text(part.get_payload(decode=True) or b"")


def eml_fields(data: bytes) -> EmailFields:
    message = email.message_from_bytes(data, policy=policy.default)
    fields = EmailFields(
        sender=str(message.get("From", "")),
        to=str(message.get("To", "")),
        cc=str(message.get("Cc", "")),
        date=str(message.get("Date", "")),
        subject=str(message.get("Subject", "")),
    )
    body = message.get_body(preferencelist=("plain", "html"))
    if body is not None:
        text = _part_text(body)
        fields.body = html_to_text(text) if body.get_content_subtype() == "html" else text
    fields.attachments = [part.get_filename() or "(unnamed)" for part in message.iter_attachments()]
    return fields


def _msg_string(value: bytes, property_type: str) -> str:
    if property_type == "001F":
        return value.decode("utf-16-le", errors="replace").rstrip("\x00")
    return decode_text(value).rstrip("\x00")


def msg_fields(streams: Dict[str, bytes], properties: Optional[bytes] = None) -> EmailFields:
    """Build the fields from a .msg file's property streams, keyed by their path inside the compound file
    ("__substg1.0_0037001F", "__attach_version1.0_#00000000/__substg1.0_3707001F"). `properties` is the
    top-level __properties_version1.0 stream, which holds the fixed-size values such as the sent time."""
    message: Dict[str, str] = {}
    attachments: Dict[str, Dict[str, str]] = {}
    for path, value in streams.items():
        parts = path.split("/")
        match = _STREAM_NAME.match(parts[-1])
        if not match:
            continue
        property_id, property_type = match[1].upper(), match[2].upper()
        if property_type not in ("001F", "001E", "0102"):
            continue
        # PR_HTML is often stored as binary. Its charset is in the HTML itself; decode_text covers UTF-8 and cp1252.
        text = decode_text(value) if property_type == "0102" else _msg_string(value, property_type)
        if len(parts) == 1:
            message[property_id] = text
        elif len(parts) == 2 and parts[0].lower().startswith("__attach_version1.0_"):
            attachments.setdefault(parts[0], {})[property_id] = text

    sender_email = message.get(PR_SENDER_SMTP_ADDRESS) or message.get(PR_SENDER_EMAIL_ADDRESS, "")
    if "@" not in sender_email:
        sender_email = ""  # An Exchange X.500 address, not useful to a reader
    sender_name = message.get(PR_SENDER_NAME, "")
    if sender_name and sender_email and sender_email.casefold() != sender_name.casefold():
        sender = f"{sender_name} <{sender_email}>"
    else:
        sender = sender_name or sender_email

    body = message.get(PR_BODY, "")
    if not body.strip() and message.get(PR_HTML):
        body = html_to_text(message[PR_HTML])

    return EmailFields(
        sender=sender,
        to=message.get(PR_DISPLAY_TO, ""),
        cc=message.get(PR_DISPLAY_CC, ""),
        date=_msg_sent_time(properties),
        subject=message.get(PR_SUBJECT, ""),
        body=body,
        attachments=[
            values.get(PR_ATTACH_LONG_FILENAME) or values.get(PR_ATTACH_FILENAME) or "(unnamed)"
            for _, values in sorted(attachments.items())
        ],
    )


def _msg_sent_time(properties: Optional[bytes]) -> str:
    """The top-level property stream is a 32-byte header followed by 16-byte entries: tag, flags, value."""
    if not properties:
        return ""
    values = {}
    for offset in range(32, len(properties) - 15, 16):
        tag = int.from_bytes(properties[offset : offset + 4], "little")
        if tag in _MSG_TIME_TAGS:
            values[tag] = int.from_bytes(properties[offset + 8 : offset + 16], "little")
    for tag in _MSG_TIME_TAGS:
        if values.get(tag):
            try:
                return format_datetime(_FILETIME_EPOCH + timedelta(microseconds=values[tag] // 10))
            except OverflowError:
                return ""
    return ""


def msg_fields_from_bytes(data: bytes) -> EmailFields:
    if not olefile.isOleFile(io.BytesIO(data)):
        raise ValueError("Not an Outlook .msg file: it is not an OLE compound file")
    try:
        with olefile.OleFileIO(io.BytesIO(data)) as ole:
            streams = {}
            for entry in ole.listdir(streams=True, storages=False):
                match = _STREAM_NAME.match(entry[-1])
                # Skips embedded messages and attachment data, which can be large
                if match and match[1].upper() in _MSG_PROPERTIES and len(entry) <= 2:
                    streams["/".join(entry)] = ole.openstream(entry).read()
            properties = (
                ole.openstream("__properties_version1.0").read() if ole.exists("__properties_version1.0") else None
            )
    except OSError as e:
        raise ValueError(f"Could not read Outlook .msg file: {e}") from e
    fields = msg_fields(streams, properties)
    if not (fields.subject or fields.body or fields.sender):
        raise ValueError("Outlook .msg file has no subject, sender, or body")
    return fields
