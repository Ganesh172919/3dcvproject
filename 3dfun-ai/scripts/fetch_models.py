"""Download the selected ONNX weights into engine/models and verify SHA-256.

    python3 scripts/fetch_models.py --all          # download every Studio model
    python3 scripts/fetch_models.py --all --check  # verify only
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "engine" / "models"
DOWNLOAD_CHUNK_BYTES = 32 * 1024 * 1024

# name -> (url, expected_sha256 or None, note)
SOURCES = {
    "u2netp.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        "U^2-Net portable saliency model (Apache-2.0), used for the alpha matte",
    ),
    "depth_anything_v2_small_quantized.onnx": (
        # Depth Anything V2 Small, exported to ONNX and dynamically quantised to
        # int8 for CPU inference. The mirror below is the copy this project was
        # built and measured with; if it is unreachable, export it yourself:
        #   pip install depth-anything-v2-onnx  (or torch + the official checkpoint)
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx",
        "fcf51f1b230362b28690bb9d1809bf0431f29cad20534e3f589bd7285547f20d",
        "Depth Anything V2 Small, int8 ONNX (Apache-2.0), monocular relative depth",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, output: BinaryIO, size: int = 0) -> None:
    if size <= DOWNLOAD_CHUNK_BYTES:
        with urllib.request.urlopen(url, timeout=90) as response:
            shutil.copyfileobj(response, output, length=1 << 20)
        return

    # The large TripoSR encoder can stall when requested as one response.
    for start in range(0, size, DOWNLOAD_CHUNK_BYTES):
        end = min(start + DOWNLOAD_CHUNK_BYTES, size) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if start == 0 and response.status == 200:
                shutil.copyfileobj(response, output, length=1 << 20)
                if output.tell() != size:
                    raise ValueError("Download did not match the expected file size.")
                return
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                raise ValueError(f"Model server returned an unexpected byte range at offset {start}.")
            shutil.copyfileobj(response, output, length=1 << 20)
        if output.tell() != end + 1:
            raise ValueError(f"Model download stopped early at offset {output.tell()}.")
        print(f"       {output.tell() / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch the ONNX models used by 3DFUN AI")
    ap.add_argument("--check", action="store_true", help="verify without downloading")
    ap.add_argument("--triposr", action="store_true", help="also install the 436 MB TripoSR INT8 model")
    ap.add_argument("--midas", action="store_true", help="also install the 67 MB MiDaS depth model")
    ap.add_argument("--all", action="store_true", help="install every model in the Studio")
    args = ap.parse_args(argv)
    MODELS.mkdir(parents=True, exist_ok=True)
    sources = SOURCES.copy()
    sizes = {}
    if args.midas or args.all:
        sources["midas_v21_small.onnx"] = (
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
            "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58",
            "MiDaS v2.1 Small (MIT)",
        )
    if args.triposr or args.all:
        sys.path.insert(0, str(ROOT / "engine"))
        from recon.triposr import MODEL_FILES, MODEL_URL

        for name, (size, digest) in MODEL_FILES.items():
            sizes[name] = size
            sources[name] = (f"{MODEL_URL}/{name}?download=true", digest, "TripoSR (MIT)")

    missing = []
    for name, (url, want, note) in sources.items():
        target = MODELS / name
        if target.is_file() and sha256(target) == want:
            print(f"[ok]   {name:44s} {target.stat().st_size / 1e6:6.1f} MB")
            continue
        if args.check:
            missing.append(name)
            print(f"[miss] {name:44s} {note}")
            continue
        print(f"[get]  {name}  <- {url}", flush=True)
        partial = None
        try:
            with tempfile.NamedTemporaryFile(dir=MODELS, prefix=name + ".", suffix=".part", delete=False) as out:
                partial = Path(out.name)
                download(url, out, sizes.get(name, 0))
            if sha256(partial) != want:
                raise ValueError("Downloaded file failed its SHA-256 check.")
            partial.replace(target)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            print(f"       download failed: {exc}")
            missing.append(name)
            continue
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
        print(f"       {target.stat().st_size / 1e6:.1f} MB, verified")

    if missing:
        print("\nMissing or invalid model files:", ", ".join(missing))
        return 1
    print("\nAll models present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
