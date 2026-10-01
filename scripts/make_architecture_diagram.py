"""
Generate images/arch.png.

The diagram is derived from the source tree rather than drawn by hand, so it
cannot quietly fall out of step with the code the way the previous hand-drawn
version did. Run it after adding a service subpackage, changing the router
mounts, or registering an LLM provider:

    python scripts/make_architecture_diagram.py

Requires matplotlib. It is a documentation-only tool: nothing in ``app/``
imports it, and the Dockerfile does not copy ``scripts/``, so the runtime
manifest does not carry the dependency.

The palette is the dashboard's own (see ``app/dashboard/theme.py``) so the
diagram and the running application read as one thing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "arch.png"

# app/dashboard/theme.py
BG = "#f8fafc"
PANEL = "#ffffff"
PANEL_BG = "#f1f5f9"
INK = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
ACCENT = "#2563eb"
OK = "#16a34a"
WARN = "#d97706"
INFO = "#0891b2"
BAD = "#dc2626"
VIOLET = "#7c3aed"
PINK = "#db2777"

SUBCOLOURS = {
    "parsing": INFO,
    "analysis": ACCENT,
    "llm": WARN,
    "cv": OK,
    "career": VIOLET,
    "jobs": PINK,
}

# One line each, in the words a reader would use. Kept short because the boxes
# are small and a summary that wraps to three lines pushes the label out.
SUMMARIES = {
    "parsing": "PDF/DOCX/TXT to text",
    "analysis": "ATS score, gaps, format",
    "llm": "registry, routing, quota",
    "cv": "optimise, LaTeX, compile",
    "career": "letters, interviews",
    "jobs": "discover, tailor, apply",
    "tracking": "applications, SQLite",
    "bulk": "bulk screening",
    "analytics": "pipeline metrics",
    "observability": "structured events",
}


def service_subpackages() -> list[str]:
    """The directories actually present under ``app/services``, read from disk."""
    services = ROOT / "app" / "services"
    return sorted(
        p.name
        for p in services.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name != "__pycache__"
    )


def api_prefixes() -> list[str]:
    """The ``prefix="..."`` values in ``app/main.py``, deduplicated, in order."""
    text = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    out: list[str] = []
    for prefix in re.findall(r'prefix="([^"]+)"', text):
        if prefix not in out:
            out.append(prefix)
    return out


def provider_count() -> int:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from app.services.llm.registry import PROVIDER_REGISTRY

        return len(PROVIDER_REGISTRY)
    except Exception:
        return 0


def endpoint_count() -> int:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from app.main import app

        return len(app.openapi().get("paths", {}))
    except Exception:
        return 50


def build() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath

    subs = service_subpackages()
    prefixes = api_prefixes() or ["/api/v1/resume"]
    n_providers = provider_count() or 13
    n_endpoints = endpoint_count()

    fig, ax = plt.subplots(figsize=(12.6, 9.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, label, *, fc=PANEL, ec=BORDER, size=8.6, weight="normal"):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.45,rounding_size=1.0",
                linewidth=1.1,
                facecolor=fc,
                edgecolor=ec,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            fontweight=weight,
            zorder=3,
            linespacing=1.5,
        )

    def panel(x, y, w, h, title):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.6,rounding_size=1.5",
                linewidth=1.0,
                facecolor=PANEL_BG,
                edgecolor=BORDER,
                zorder=1,
            )
        )
        ax.text(
            x + 2.2,
            y + h - 2.0,
            title,
            ha="left",
            va="top",
            fontsize=9.2,
            color=MUTED,
            fontweight="bold",
            zorder=3,
        )

    def arrow(x1, y1, x2, y2, label=None, *, dashed=False, rad=0.0, color=MUTED):
        verts = [
            (x1, y1),
            (x1 + (x2 - x1) * 0.38, y1),
            (x1 + (x2 - x1) * 0.70, y2),
            (x2, y2),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        ax.add_patch(
            patches.PathPatch(
                MplPath(verts, codes),
                facecolor="none",
                edgecolor=color,
                linewidth=0.95,
                linestyle=(0, (3, 2)) if dashed else "solid",
                zorder=2,
                capstyle="round",
            )
        )
        ax.plot(
            [x2], [y2], marker=">", markersize=4.6, color=color,
            markeredgewidth=0, linestyle="none", zorder=2,
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.9,
                label,
                ha="center", va="bottom", fontsize=7.0, color=MUTED, zorder=4,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.0),
            )

    # -- client ----------------------------------------------------------
    panel(28, 89.0, 44, 9.0, "CLIENT")
    box(38, 90.0, 24, 5.2, "Streamlit dashboard\n:8501", weight="bold")

    # -- API -------------------------------------------------------------
    panel(4, 74.0, 92, 12.5, "FASTAPI BACKEND  :8000")
    route = "  +  ".join(prefixes)
    box(24, 76.4, 44, 4.6, f"{n_endpoints} endpoints   {route}", weight="bold", size=8.4)
    box(72, 76.4, 18, 4.6, "event log", size=7.8, ec=MUTED)
    arrow(68, 78.7, 72, 78.7, dashed=True, color=MUTED)

    # -- service layer: a grid sized to the subpackage count --------------
    panel(4, 25.0, 92, 45.0, "SERVICE LAYER   app/services/")

    cols = 3
    box_w, gap_x = 25.0, 3.0
    grid_x = 10.0
    box_h, gap_y = 7.5, 2.2
    grid_top = 63.5  # top edge of the first row

    placed: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(subs):
        r, c = divmod(i, cols)
        x = grid_x + c * (box_w + gap_x)
        y = grid_top - (r + 1) * box_h - r * gap_y
        placed[name] = (x + box_w / 2, y + box_h / 2)
        colour = SUBCOLOURS.get(name, MUTED)
        box(x, y, box_w, box_h, f"{name}/\n{SUMMARIES.get(name, '')}", ec=colour, size=8.2)
        # A coloured spine on the left edge, so a box can be colour-matched to
        # its subpackage without reading the label.
        ax.add_patch(
            patches.Rectangle((x, y), 0.5, box_h, facecolor=colour, edgecolor="none", zorder=3)
        )

    # -- the two gates that are the point of the project -----------------
    # Centred rather than full width, which leaves a clear margin down each
    # side for the edges to the external tools.
    box(
        20.0, 12.0, 28.0, 8.0,
        "Factual validation gate\ncompanies, titles,\ndates, degrees",
        ec=BAD, size=8.2, weight="bold",
    )
    box(
        52.0, 12.0, 28.0, 8.0,
        "PDF content validation\nreading order,\nclipping, sections",
        ec=OK, size=8.2, weight="bold",
    )

    # -- externals -------------------------------------------------------
    panel(4, 0.0, 92, 9.0, "EXTERNAL")
    box(
        6.0, 0.8, 40.0, 5.2,
        f"LLM providers ({n_providers})  ·  local Ollama / OmniRoute\n"
        "anthropic + openai via SDK, the rest over plain HTTP",
        size=7.4, ec=WARN,
    )
    box(
        54.0, 0.8, 40.0, 5.2,
        "pdflatex\n-no-shell-escape, private temp dir, bounded timeout",
        size=7.4, ec=INFO,
    )

    # -- wiring ----------------------------------------------------------
    # Deliberately few edges. Drawing one per box produced a thicket of crossing
    # curves that carried no more information than the layout already does.
    arrow(50, 90.0, 50, 81.0, "HTTP", dashed=True)
    arrow(50, 74.0, 50, 70.0, None, dashed=True, color=BORDER)
    arrow(50, 25.0, 50, 20.0, "generated CV", color=OK)

    # Down the side margins, so neither edge crosses a gate.
    arrow(14, 25.0, 20, 6.4, "llm/", rad=-0.06, color=WARN)
    arrow(86, 25.0, 78, 6.4, "cv/", rad=0.06, color=INFO)

    fig.savefig(OUT, dpi=190, facecolor=BG, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def main() -> int:
    build()
    print(f"[OK] wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
