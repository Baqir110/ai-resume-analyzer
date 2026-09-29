"""Helpers for the ATS scoring tests.

The PDF Parsing category is only measurable against real PDF bytes, and a
hand-built file is the only way to exercise it without invoking pdflatex -- which
is not installed everywhere and would make the scoring tests depend on a LaTeX
distribution.
"""

from __future__ import annotations

__all__ = ["minimal_pdf", "pdf_with_text"]


def _build_pdf(content_stream: bytes) -> bytes:
    """
    Assemble a one-page PDF around an already-encoded content stream.

    Object offsets are computed as the document is written, because a PDF xref
    table is only valid if every offset is exact; a hand-typed one produces a
    file that opens in a viewer but that a strict parser will reject, which
    would make these tests measure the wrong thing.
    """
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        # Uncompressed on purpose. Flate would be more realistic, but the
        # scoring tests must not depend on this helper getting compression
        # right -- a fixture that is subtly malformed tests nothing useful.
        b"<< /Length "
        + str(len(content_stream)).encode()
        + b" >>\nstream\n"
        + content_stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
    out += b"startxref\n" + str(xref_at).encode() + b"\n%%EOF\n"
    return bytes(out)


def pdf_with_text(lines: list[str]) -> bytes:
    """A one-page PDF carrying the given lines as real, extractable text."""
    # Tokens are space-separated. PDF operators are delimited by whitespace, so
    # concatenating them as b"BT" + b"/F1" yields "Tf12" — a single unknown
    # operator — and the page renders nothing at all.
    parts: list[bytes] = [b"BT", b"/F1", b"11", b"Tf", b"12", b"TL", b"72", b"760", b"Td"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        parts.append(b"(")
        parts.append(escaped.encode("latin-1", "replace"))
        parts.append(b")")
        parts.append(b"Tj")
        parts.append(b"T*")
    parts.append(b"ET")
    return _build_pdf(b" ".join(parts))


def minimal_pdf() -> bytes:
    """
    A structurally valid PDF with enough text to pass the readability checks.

    Modelled on what the CV generator actually emits: a header, the standard
    section headings, and a skills line. Enough for the category under test to
    produce a meaningful number rather than a floor value.
    """
    return pdf_with_text(
        [
            "Jane Doe",
            "Senior Platform Engineer",
            "jane.doe@example.com +44 20 7946 0000",
            "PROFILE",
            "Senior Platform Engineer with eight years in container infrastructure.",
            "EXPERIENCE",
            "Senior Platform Engineer, Northwind Systems, 2020 - Present",
            "Ran Kubernetes clusters on AWS across three production environments.",
            "Replaced manual provisioning with Terraform and Ansible automation.",
            "Built CI/CD pipelines using GitHub Actions and Jenkins.",
            "Instrumented services with Prometheus and Grafana.",
            "EDUCATION",
            "BSc Computer Science, University of Manchester, 2013 - 2017",
            "SKILLS",
            "Python Go Docker Kubernetes Terraform Ansible AWS Jenkins GitHub Actions",
            "PostgreSQL Redis Prometheus Grafana GitLab CI",
            "CERTIFICATIONS",
            "Certified Kubernetes Administrator",
            "AWS Solutions Architect Associate",
        ]
    )
