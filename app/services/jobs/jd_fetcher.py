"""Fetch the job description text from a job posting URL.

Tries a series of known selectors for the major ATS platforms, then
falls back to "largest visible text block on the page". Returns plain
text suitable for passing to /analyze or the CV generator.

Handles: Greenhouse, Lever, Ashby, Workday, SmartRecruiters, Workable,
and generic career pages.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.core.network import public_async_client
from app.core.security import UnsafeInputError, validate_public_http_url

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Ordered list of CSS selectors to try, per domain pattern.
# The first selector that yields > 200 chars wins.
_KNOWN_SELECTORS = [
    # Greenhouse
    "div.job__description",
    "div#content",
    "div[class*='job-description']",
    # Lever
    "div.section-wrapper",
    "div[data-qa='job-description']",
    "div.posting-page",
    # Ashby
    "div[class*='_description_']",
    "div[class*='JobPosting']",
    # Workday
    "div[data-automation-id='jobPostingDescription']",
    # SmartRecruiters
    "div[class*='job-description']",
    "section[class*='description']",
    # Workable
    "div[data-ui='job-description']",
    "section[class*='job-description']",
    # Generic
    "article",
    "main",
    "div[role='main']",
]

# Noise to strip from any block before returning it.
_STRIP_SELECTORS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "form",
    "iframe",
    "svg",
    "button",
]


def _clean_block(text: str) -> str:
    """Normalize whitespace and drop junk lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        # Drop obvious boilerplate / cookie banner fragments.
        low = s.lower()
        if any(
            junk in low
            for junk in (
                "cookie",
                "accept all",
                "privacy policy",
                "terms of service",
                "sign in",
                "log in",
                "powered by",
                "© ",
                "all rights reserved",
            )
        ):
            continue
        lines.append(s)

    return "\n".join(lines).strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    for sel in _STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()


def _largest_text_block(soup: BeautifulSoup) -> Optional[str]:
    """
    Fallback: find the container with the most paragraph text.

    Most job pages put the description inside a div or section that
    contains several <p> tags. Score every candidate by total char
    count of its <p> descendants and return the top one.
    """
    best_el = None
    best_len = 0

    for tag in soup.find_all(["div", "section", "article", "main"]):
        paragraphs = tag.find_all("p", recursive=True)
        if len(paragraphs) < 3:
            continue
        total = sum(len(p.get_text(strip=True)) for p in paragraphs)
        if total > best_len:
            best_len = total
            best_el = tag

    if best_el is None or best_len < 200:
        return None

    return best_el.get_text("\n", strip=True)


async def fetch_job_description(url: str, timeout: float = 20.0) -> str:
    """
    Fetch and extract the JD text from a job URL.

    Raises RuntimeError if the page could not be fetched or no
    meaningful text was extracted.
    """
    if not url:
        raise RuntimeError("URL is empty")
    try:
        url = validate_public_http_url(url, resolve_dns=True)
    except UnsafeInputError as exc:
        raise RuntimeError(f"Not a valid URL: {exc}") from exc

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    }

    try:
        async with public_async_client(
            timeout=timeout, follow_redirects=False, headers=headers
        ) as client:
            current_url = url
            for _ in range(5):
                try:
                    validate_public_http_url(current_url, resolve_dns=True)
                except UnsafeInputError as exc:
                    raise RuntimeError(f"Redirect target is not allowed: {exc}") from exc
                resp = await client.get(current_url)
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise RuntimeError("Redirect response did not include a location")
                    current_url = urljoin(current_url, location)
                    continue
                break
            else:
                raise RuntimeError("Too many redirects while fetching job description")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Fetch failed for {url}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Fetch returned HTTP {resp.status_code} for {url}")

    try:
        content_length = int(resp.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > 8 * 1024 * 1024:
        raise RuntimeError("Fetched job description response is too large")

    html = resp.text
    if len(html) < 500:
        raise RuntimeError(
            f"Page content is suspiciously short ({len(html)} chars). "
            "The URL may require JavaScript or a login."
        )

    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)

    # 1. Try known selectors
    for sel in _KNOWN_SELECTORS:
        for el in soup.select(sel):
            text = el.get_text("\n", strip=True)
            cleaned = _clean_block(text)
            if len(cleaned) >= 300:
                logger.info(
                    "JD fetch: matched selector %r on %s (%d chars)",
                    sel,
                    url,
                    len(cleaned),
                )
                return cleaned

    # 2. Fallback: largest text block
    fallback = _largest_text_block(soup)
    if fallback:
        cleaned = _clean_block(fallback)
        if len(cleaned) >= 300:
            logger.info(
                "JD fetch: used largest-text-block fallback (%d chars)",
                len(cleaned),
            )
            return cleaned

    # 3. Last resort: whole-body text
    body = soup.body.get_text("\n", strip=True) if soup.body else ""
    cleaned = _clean_block(body)
    if len(cleaned) >= 300:
        logger.warning(
            "JD fetch: fell back to whole-body text (%d chars). "
            "The extracted text may contain noise.",
            len(cleaned),
        )
        return cleaned

    raise RuntimeError(
        f"Could not extract a meaningful JD from {url}. "
        f"Page length was {len(html)} chars, extracted {len(cleaned)}. "
        "The page may require JavaScript rendering."
    )
