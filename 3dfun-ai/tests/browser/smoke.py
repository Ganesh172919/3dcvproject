"""Headless browser smoke test for the 3DFUN AI SPA (Playwright, chromium)."""
import os
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = os.environ.get("THREEFUN_BROWSER_URL", "http://127.0.0.1:8010" if os.name == "nt" else "http://127.0.0.1:8000")
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "engine/testdata/sneaker.png"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="3dfun-browser-"))
OUT.mkdir(parents=True, exist_ok=True)
email = f"browser{int(time.time())}@example.com"

errors, console = [], []


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("THREEFUN_BROWSER_CHANNEL"),
                                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 950})
        page.on("console", lambda m: (console.append(f"{m.type}: {m.text}"),
                                      errors.append(f"{m.type}: {m.text}") if m.type == "error" else None))
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url} {r.failure}")
                if r.failure != "net::ERR_ABORTED" else None)

        # landing page
        page.goto(f"{BASE}/", wait_until="networkidle")
        expect(page.locator("#example-stage .viewer-hud")).to_contain_text("4,998 tris", timeout=30_000)
        page.screenshot(path=str(OUT / "01-landing.png"), full_page=False)
        assert "3DFUN" in page.title(), page.title()
        page.get_by_role("button", name="Teapot", exact=True).click()
        expect(page.locator("#example-stage .viewer-hud")).to_contain_text("3 tex", timeout=30_000)
        page.get_by_role("button", name="Photo", exact=True).click()
        expect(page.get_by_role("img", name="Teapot reference photo")).to_be_visible()

        # A guest can pick a real local file and keep it through registration.
        page.set_input_files("#home-file", IMAGE)
        page.wait_for_url("**/app/studio?draft=*")
        page.get_by_role("main").get_by_role("link", name="Create account", exact=True).click()
        expect(page.locator("#name")).to_be_visible()
        page.fill("#email", email)
        page.fill("#name", "Browser Test")
        page.fill("#password", "Hunter2Hunter2")
        page.screenshot(path=str(OUT / "02-register.png"))
        page.click("form button[type=submit]")
        page.wait_for_url("**/app/studio?upload=*")
        page.reload()
        expect(page.get_by_text(Path(IMAGE).name, exact=True)).to_be_visible()
        page.screenshot(path=str(OUT / "03-after-register.png"))
        print("after register url:", page.url)

        # studio
        page.goto(f"{BASE}/app/studio", wait_until="networkidle")
        expect(page.get_by_role("heading", name="Image to 3D Studio")).to_be_visible()
        page.screenshot(path=str(OUT / "04-studio.png"), full_page=True)
        page.set_input_files("input[type=file]", IMAGE)
        expect(page.get_by_role("button", name="Generate model")).to_be_enabled(timeout=30_000)
        page.screenshot(path=str(OUT / "05-upload.png"), full_page=True)
        if os.environ.get("THREEFUN_BROWSER_GENERATE") == "1":
            # Use the seeded account so the completed demo can be opened after the test.
            page.get_by_role("button", name="Sign out").click()
            page.get_by_role("button", name="Use demo account").click()
            page.wait_for_url("**/app/studio")
            engine = os.environ.get("THREEFUN_BROWSER_ENGINE", "triposr")
            sample = {"triposr": "Chair", "midas": "Teapot", "local": "Sneaker"}[engine]
            page.get_by_label("Engine", exact=True).select_option(engine)
            page.get_by_role("button", name=sample, exact=True).click()
            page.get_by_text("Export and advanced options").click()
            page.get_by_label("GLTF", exact=True).check()
            page.get_by_role("button", name="Generate model").click(timeout=30_000)
            expect(page.get_by_role("heading", name="Your model")).to_be_visible(timeout=240_000)
            expect(page.locator(".viewer-hud")).to_contain_text("tri", timeout=30_000)
            if engine != "triposr":
                expect(page.locator(".viewer-hud")).to_contain_text("3 tex")
            expect(page.get_by_text("Watertight", exact=True)).to_be_visible()
            page.screenshot(path=str(OUT / "06-result.png"))
            stem = sample.lower()
            with page.expect_download() as glb:
                page.get_by_role("link", name="Download GLB", exact=True).click()
            glb.value.save_as(OUT / f"{stem}.glb")
            assert (OUT / f"{stem}.glb").read_bytes()[:4] == b"glTF"
            from zipfile import ZipFile
            with page.expect_download() as gltf:
                page.get_by_role("link", name="Download glTF bundle").click()
            gltf.value.save_as(OUT / f"{stem}-gltf.zip")
            with ZipFile(OUT / f"{stem}-gltf.zip") as archive:
                assert {"model.gltf", "model.bin"} <= set(archive.namelist())
            print("Demo model:", page.get_by_role("link", name="Open model details").get_attribute("href"))
            page.get_by_role("link", name="Open model details").click()
            expect(page.locator(".viewer-hud")).to_contain_text("tri", timeout=30_000)
            expect(page.get_by_role("img", name="Reference image used to generate the model")).to_be_visible()

            page.get_by_role("link", name="Dashboard", exact=True).click()
            expect(page.get_by_role("heading", name="Experiment notebook")).to_be_visible()
            expect(page.get_by_role("region", name="Runs on the selected image")).to_be_visible()
            with page.expect_download() as csv:
                page.get_by_role("button", name="Download CSV").click()
            assert "image_sha256" in csv.value.path().read_text(encoding="utf-8")

        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="Menu", exact=True).click()
        page.get_by_role("link", name="Chat", exact=True).click()
        expect(page.get_by_role("heading", name="Gemini chat")).to_be_visible()
        expect(page.get_by_text("Checking Gemini setup…")).to_have_count(0)
        assert page.evaluate("document.documentElement.scrollWidth === innerWidth"), "Mobile chat overflows horizontally"
        page.screenshot(path=str(OUT / "07-mobile-chat.png"), full_page=True)
        browser.close()

    print("\n--- console errors ---")
    print("\n".join(errors[:30]) or "none")
    print("\n--- console tail ---")
    print("\n".join(console[-15:]) or "none")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
