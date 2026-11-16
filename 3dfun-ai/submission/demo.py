"""Produce an 84-second, captioned demo from real project renders."""
import argparse
from pathlib import Path
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "web/public/examples"
OUT = ROOT / "out/course17"
W, H, FPS, SECONDS = 1280, 720, 24, 84
INK, DIM, ACCENT = "#17283c", "#425c6c", "#047d74"
ITEMS = (
    ("From a photo to a 3D mesh", "01  /  SINGLE IMAGE", "One image supplies a silhouette and surface cues. The hidden side is inferred.",
     ("Source photo", PUB / "source-sneaker.png"), ("Textured mesh turntable", PUB / "turntable.gif")),
    ("Look at the mesh without texture", "02  /  GEOMETRY", "Turn the model to inspect triangulation, thickness and the estimated back.",
     ("Triangular wireframe", PUB / "wireframe.png"), ("Clay render", PUB / "clay.png")),
    ("Try a different object", "03  /  SECOND EXAMPLE", "Rounded shapes suit the thickness prior. Small features can still be softened.",
     ("Rubber duck input", PUB / "rubber-duck-input.png"), ("Reconstructed mesh", PUB / "rubber-duck-preview.png")),
    ("Compare depth models", "04  /  FAILURE CASE", "Both meshes close the ring's opening. A closed mesh can still have the wrong shape.",
     ("Depth Anything V2", PUB / "ring-preview.png"), ("MiDaS Small", PUB / "ring-midas-preview.png")),
    ("Train on known geometry", "05  /  CUSTOM TRAINING", "On eight held-out views, cloud Chamfer was 0.098 vs. 0.136 for constant depth.",
     ("Held-out depth prediction and validation curves", ROOT / "results/course17/tinydepth_figure.png"), None),
    ("Run it locally", "06  /  REPRODUCE", "Open README.md for the image demo, calibration, and training commands.",
     ("Toy car input", PUB / "toy-car-input.png"), ("Reconstructed mesh", PUB / "toy-car-preview.png")),
)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def read_asset(path):
    with Image.open(path) as img:
        return [(frame.convert("RGBA"), max(1, round(frame.info.get("duration", 120) * FPS / 1000)))
                for frame in ImageSequence.Iterator(img)]


def asset_frames(item, bounds):
    title, path = item
    x, y, w, h = bounds
    rendered = []
    for frame, length in read_asset(path):
        base = Image.new("RGB", (w, h), "#ffffff")
        d = ImageDraw.Draw(base)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=16, fill="#ffffff", outline="#d9e3e7", width=2)
        thumb = ImageOps.contain(frame, (w - 38, h - 62), Image.Resampling.LANCZOS)
        base.paste(thumb, ((w - thumb.width) // 2, 12 + (h - 62 - thumb.height) // 2), thumb)
        d.text((20, h - 34), title, fill=INK, font=font(19, True))
        rendered += [base] * length
    return rendered, (x, y)


def scene(index):
    heading, overline, note, left, right = ITEMS[index]
    img = Image.new("RGB", (W, H), "#f3f7f8")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 8), fill=ACCENT)
    draw.text((64, 38), "3DFUN AI   /   GROUP 17", font=font(21, True), fill=INK)
    draw.text((64, 101), overline, font=font(17, True), fill=ACCENT)
    draw.text((64, 129), heading, font=font(51, True), fill=INK)
    draw.text((64, 195), note, font=font(23), fill=DIM)
    draw.text((64, 677), "IIIT DHARWAD  /  3D COMPUTER VISION", font=font(14, True), fill=DIM)
    if right:
        frames = [asset_frames(left, (64, 247, 560, 402)), asset_frames(right, (656, 247, 560, 402))]
    else:
        frames = [asset_frames(left, (64, 247, 1152, 402))]
    return img, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "demo.mp4")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen([get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)],
                          stdin=subprocess.PIPE) as encode:
        per_scene = FPS * SECONDS // len(ITEMS)
        for i in range(len(ITEMS)):
            canvas, assets = scene(i)
            distinct = max(len(loop) for loop, _ in assets)
            frames = []
            for k in range(distinct):
                frame = canvas.copy()
                for loop, position in assets:
                    frame.paste(loop[k % len(loop)], position)
                frames.append(frame.tobytes())
                if len(frames) * W * H * 3 >= 85_000_000:
                    frames.clear()
                    break
            for k in range(per_scene):
                if frames:
                    encode.stdin.write(frames[k % len(frames)])
                else:
                    frame = canvas.copy()
                    for loop, position in assets:
                        frame.paste(loop[k % len(loop)], position)
                    encode.stdin.write(frame.tobytes())
            print(f"Encoded segment {i + 1}/{len(ITEMS)}", flush=True)
        encode.stdin.close()
        if encode.wait():
            raise RuntimeError("FFmpeg failed.")


if __name__ == "__main__":
    main()
