"""Render every page/slide of the deliverables and audit the result visually.

    python3 scripts/audit_renders.py            # paper + report
    python3 scripts/audit_renders.py --deck     # also rasterise the .pptx

For the PDFs it reports, per page: figure count, whether any text crosses the page
margin, and writes a PNG contact sheet to ``results/renders/``. For the deck it
converts with LibreOffice when available and runs the same checks, because a
text box that fits the shape can still collide with a neighbour.

This is the automated half of "read all images of report and fix any UI/UX
errors"; the PNGs it writes are the artefacts a human then looks at.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "renders"


def audit_pdf(path: Path, label: str, dpi: int = 90) -> dict:
    import pymupdf

    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(path)
    problems: list[str] = []
    pages = []
    for number, page in enumerate(doc, start=1):
        rect = page.rect
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")
        overflow = []
        for block in text_blocks:
            x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            if x0 < -1 or y0 < -1 or x1 > rect.width + 1 or y1 > rect.height + 1:
                overflow.append(text.strip()[:40])
        # a block wider than 96% of the page and shorter than 6 pt is usually a rule
        empty = not text_blocks and not images
        if overflow:
            problems.append(f"{label} p{number}: {len(overflow)} block(s) outside the page "
                            f"({overflow[:2]})")
        if empty:
            problems.append(f"{label} p{number}: page has no text and no images")
        pix = page.get_pixmap(dpi=dpi)
        png = OUT / f"{label}_p{number:02d}.png"
        pix.save(png)
        pages.append({"page": number, "images": len(images), "blocks": len(text_blocks),
                      "render": str(png.relative_to(ROOT))})
    return {"file": str(path.relative_to(ROOT)), "pages": pages, "problems": problems}


def audit_deck() -> dict:
    deck = ROOT / "presentations" / "3DFUN-AI-Project.pptx"
    if not deck.exists():
        return {"file": "presentations/3DFUN-AI-Project.pptx", "problems": ["deck missing"]}
    problems: list[str] = []
    rendered: list[str] = []
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        target = OUT / "deck"
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(target),
                        str(deck)], capture_output=True, timeout=600)
        pdf = target / (deck.stem + ".pdf")
        if pdf.exists():
            result = audit_pdf(pdf, "deck")
            problems += result["problems"]
            rendered = [p["render"] for p in result["pages"]]
    else:
        from pptx import Presentation

        prs = Presentation(str(deck))
        for i, slide in enumerate(prs.slides, start=1):
            boxes = [(s.left, s.top, s.left + s.width, s.top + s.height)
                     for s in slide.shapes if s.width and s.height]
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    ax0, ay0, ax1, ay1 = boxes[a]
                    bx0, by0, bx1, by1 = boxes[b]
                    ox = min(ax1, bx1) - max(ax0, bx0)
                    oy = min(ay1, by1) - max(ay0, by0)
                    if ox > 9144 and oy > 9144:      # >0.01 in in both axes
                        problems.append(f"deck s{i}: shapes {a} and {b} overlap by "
                                        f"{ox/914400:.2f} x {oy/914400:.2f} in")
                        break
    return {"file": "presentations/3DFUN-AI-Project.pptx", "rendered": rendered,
            "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", action="store_true", help="also audit the .pptx")
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args()

    reports = []
    for relative, label in [("paper/main.pdf", "paper"), ("docs/report.pdf", "report")]:
        path = ROOT / relative
        if not path.exists():
            reports.append({"file": relative, "problems": [f"{relative} missing — build it first"]})
            continue
        reports.append(audit_pdf(path, label, args.dpi))
    if args.deck:
        reports.append(audit_deck())

    (OUT / "audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    total_problems = 0
    for report in reports:
        problems = report["problems"]
        total_problems += len(problems)
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        pages = len(report.get("pages", [])) or len(report.get("rendered", []))
        print(f"[{status}] {report['file']}" + (f" — {pages} pages rendered" if pages else ""))
        for problem in problems[:8]:
            print(f"        {problem}")
    print(f"\ncontact sheets in {OUT.relative_to(ROOT)}/ (audit.json has the detail)")
    return 0 if total_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
