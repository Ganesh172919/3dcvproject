"""Render the IEEE report as browser-readable pages using Poppler."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def render_preview(pdf, destination):
    with TemporaryDirectory(prefix="course-report-preview-") as scratch:
        prefix = Path(scratch) / "page"
        subprocess.run(["pdftoppm", "-r", "140", "-png", str(pdf), str(prefix)], check=True)
        pages = sorted(prefix.parent.glob("page-*.png"))
        if not 4 <= len(pages) <= 6:
            raise ValueError("Expected a four- to six-page IEEE report.")
        destination.mkdir(parents=True, exist_ok=True)
        previews = []
        for number, page in enumerate(pages, start=1):
            filename = f"page-{number}.webp"
            with Image.open(page) as rendered:
                rendered.save(destination / filename, lossless=True)
                width, height = rendered.size
            lazy = ' loading="lazy"' if number > 1 else ""
            previews.append(
                f'    <figure><img src="{filename}" width="{width}" height="{height}"{lazy} '
                f'alt="Course report, page {number} of {len(pages)}"></figure>'
            )
    images = "\n".join(previews)
    (destination / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="report-sha256" content="{sha256(pdf.read_bytes()).hexdigest()}">
<title>IEEE report preview · 3DFUN AI</title>
<link rel="stylesheet" href="/static/css/app.css">
</head>
<body>
<a class="skip-link" href="#report">Skip to report</a>
<main class="course-report-preview" id="report">
  <h1>IEEE report preview</h1>
  <p>3DFUN AI · Group 17 · {len(pages)} pages</p>
  <nav class="row wrap gap8" aria-label="Report links">
    <a class="btn primary" href="/report.pdf?download=1" download="report.pdf">Download PDF</a>
    <a class="btn" href="/app/showcase">Back to course results</a>
  </nav>
{images}
</main>
</body>
</html>
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "out/course17/report.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "web/public/report-preview")
    args = parser.parse_args()
    try:
        render_preview(args.pdf, args.output)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
