"""Poll Gmail for verification codes sent during job application submission.

Uses IMAP with a Google App Password (not the account password).
Requires 2-Step Verification enabled on the account, then an App
Password created at myaccount.google.com -> Security -> App passwords.

Environment variables:
    GMAIL_USER             e.g. candidate@example.com
    GMAIL_APP_PASSWORD     16-character app password (spaces ok)
    GMAIL_IMAP_HOST        optional, defaults to imap.gmail.com
"""

from __future__ import annotations

import email as email_lib
import imaplib
import ipaddress
import logging
import os
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

IMAP_HOST = os.getenv("GMAIL_IMAP_HOST", "imap.gmail.com").strip()
IMAP_USER = os.getenv("GMAIL_USER", "").strip()
IMAP_PASS = os.getenv("GMAIL_APP_PASSWORD", "").strip()
ALLOW_EMAIL_VERIFICATION = os.getenv("ALLOW_EMAIL_VERIFICATION", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Sender/subject substrings that indicate a verification email.
_VERIFICATION_HINTS = (
    "greenhouse",
    "lever",
    "ashby",
    "workable",
    "workday",
    "verification",
    "security code",
    "verify your email",
    "confirm your email",
    "one-time",
    "otp",
)

# Standalone 6-digit number. Won't match phone numbers because those
# contain spaces, dashes, or more/fewer digits.
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass
class VerificationCode:
    code: str
    subject: str
    sender: str
    received_at: str


def _plain_text_body(msg) -> str:
    """Extract a plain-text version of the email body."""
    if msg.is_multipart():
        # Prefer the text/plain part
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
        # Fall back to stripping HTML
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode("utf-8", errors="replace")
                    return re.sub(r"<[^>]+>", " ", html)
        return ""

    payload = msg.get_payload(decode=True)
    if payload:
        return payload.decode("utf-8", errors="replace")
    return str(msg.get_payload())


def _is_verification_email(sender: str, subject: str) -> bool:
    haystack = f"{sender} {subject}".lower()
    return any(hint in haystack for hint in _VERIFICATION_HINTS)


def _received_after(msg, cutoff: datetime) -> bool:
    try:
        date_str = msg.get("Date", "")
        if not date_str:
            return False
        received = parsedate_to_datetime(date_str)
        if received.tzinfo is None:
            received = received.replace(tzinfo=timezone.utc)
        return received >= cutoff
    except Exception:
        return True


def _imap_host_is_public(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, 993, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for entry in addresses:
        address = entry[4][0]
        try:
            if not ipaddress.ip_address(address).is_global:
                return False
        except ValueError:
            return False
    return bool(addresses)


def _poll_once(
    cutoff: datetime,
    mark_as_read: bool,
) -> VerificationCode | None:
    with imaplib.IMAP4_SSL(IMAP_HOST) as M:
        M.login(IMAP_USER, IMAP_PASS)
        M.select("INBOX")

        search_criteria = f'(UNSEEN SINCE "{cutoff.strftime("%d-%b-%Y")}")'
        status, data = M.search(None, search_criteria)
        if status != "OK" or not data or not data[0]:
            return None

        msg_ids = data[0].split()

        # Newest first. Only scan the last 30 to keep this fast.
        for msg_id in reversed(msg_ids[-30:]):
            status, msg_data = M.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue

            raw = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw)

            if not _received_after(msg, cutoff):
                continue

            sender = str(msg.get("From", ""))
            subject = str(msg.get("Subject", ""))

            if not _is_verification_email(sender, subject):
                continue

            body = _plain_text_body(msg)

            match = _CODE_RE.search(body) or _CODE_RE.search(subject)
            if not match:
                continue

            code = match.group(1)

            if mark_as_read:
                try:
                    M.store(msg_id, "+FLAGS", "\\Seen")
                except Exception:
                    pass

            return VerificationCode(
                code=code,
                subject=subject,
                sender=sender,
                received_at=str(msg.get("Date", "")),
            )

    return None


def fetch_verification_code(
    since_seconds: int = 300,
    timeout_seconds: int = 120,
    poll_interval: int = 5,
    mark_as_read: bool = False,
) -> VerificationCode | None:
    """
    Poll Gmail for a recent verification code.

    Args:
        since_seconds: Only consider messages newer than this.
        timeout_seconds: Total time to keep polling.
        poll_interval: Seconds between IMAP polls.
        mark_as_read: Mark the email read once the code is extracted.

    Returns:
        VerificationCode if found, else None.
    """
    if not ALLOW_EMAIL_VERIFICATION:
        logger.info("email_reader: email verification is disabled by policy")
        return None
    if not IMAP_USER or not IMAP_PASS:
        logger.info(
            "email_reader: GMAIL_USER or GMAIL_APP_PASSWORD not set — "
            "email verification is disabled."
        )
        return None
    if not _imap_host_is_public(IMAP_HOST):
        logger.warning("email_reader: refusing non-public IMAP host")
        return None

    since_seconds = max(30, min(int(since_seconds), 86_400))
    timeout_seconds = max(5, min(int(timeout_seconds), 600))
    poll_interval = max(1, min(int(poll_interval), 30))

    deadline = time.time() + timeout_seconds
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_seconds)

    while time.time() < deadline:
        try:
            code = _poll_once(cutoff, mark_as_read=mark_as_read)
            if code:
                logger.info("email_reader: found a verification email")
                return code
        except Exception as exc:
            logger.warning("email_reader: IMAP poll failed: %s", exc)

        time.sleep(poll_interval)

    logger.info("email_reader: no verification code found within timeout")
    return None
