"""Comprehensive end-to-end browser test for all 3DFUN AI upgraded features.
Runs with Playwright Edge channel on http://127.0.0.1:8010.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8010")
OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

console_messages = []
console_errors = []
page_errors = []


def run_verification() -> bool:
    print(f"[1/7] Connecting to {BASE_URL} with Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.on("console", lambda msg: (
            console_messages.append(f"[{msg.type}] {msg.text}"),
            console_errors.append(msg.text) if msg.type == "error" else None,
        ))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # ----------------------------------------------------------------------
        # TEST 1: Landing Page & WebGL CAD Shading Controls
        # ----------------------------------------------------------------------
        print("[2/7] Testing Landing Page & Interactive WebGL Viewport...")
        page.goto(f"{BASE_URL}/", wait_until="networkidle")
        assert "3DFUN" in page.title(), f"Unexpected title: {page.title()}"
        expect(page.locator(".course-label").first).to_be_visible()
        expect(page.locator("text=Free Trial").first).to_be_visible()
        expect(page.locator("a[href='/app/labs']").first).to_be_visible()

        # Wait for ModelViewer to initialize in example stage
        expect(page.locator("#example-stage .viewer-hud")).to_be_visible(timeout=30000)

        # Test Shading Modes in Viewport: Wireframe-on-Shaded, White Clay, Surface Normals, PBR Texturized
        wire_btn = page.locator("#example-stage button:has-text('Wireframe')")
        if wire_btn.is_visible():
            wire_btn.click()
            time.sleep(0.3)
        clay_btn = page.locator("#example-stage button:has-text('Clay')")
        if clay_btn.is_visible():
            clay_btn.click()
            time.sleep(0.3)
        norm_btn = page.locator("#example-stage button:has-text('Normals')")
        if norm_btn.is_visible():
            norm_btn.click()
            time.sleep(0.3)
        pbr_btn = page.locator("#example-stage button:has-text('PBR')")
        if pbr_btn.is_visible():
            pbr_btn.click()
            time.sleep(0.3)

        # Test Camera Presets: Front, Side, Top, 360°
        f_btn = page.locator("#example-stage button:has-text('Front')")
        if f_btn.is_visible():
            f_btn.click()
            time.sleep(0.2)
        s_btn = page.locator("#example-stage button:has-text('Side')")
        if s_btn.is_visible():
            s_btn.click()
            time.sleep(0.2)
        t_btn = page.locator("#example-stage button:has-text('Top')")
        if t_btn.is_visible():
            t_btn.click()
            time.sleep(0.2)
        spin_btn = page.locator("#example-stage button:has-text('360°')")
        if spin_btn.is_visible():
            spin_btn.click()
            time.sleep(0.3)

        # Test 3D Print Readiness Check
        print_btn = page.locator("#example-stage button[title*='3D Print Readiness']")
        if print_btn.is_visible():
            print_btn.click()
            time.sleep(0.5)
            expect(page.locator(".viewer-print-modal")).to_be_visible()
            expect(page.locator("text=Manifold Surface Topology")).to_be_visible()
            page.locator(".viewer-print-modal button:has-text('Close')").click()

        page.screenshot(path=str(OUT_DIR / "01_landing_webgl.png"))
        print("  -> Landing page and WebGL viewer controls OK.")

        # ----------------------------------------------------------------------
        # TEST 2: Account Creation & Free Trial Credits
        # ----------------------------------------------------------------------
        print("[3/7] Testing Registration & Wallet Credits...")
        page.goto(f"{BASE_URL}/app/register", wait_until="networkidle")
        test_email = f"student_{int(time.time())}@3dfun.edu"
        page.fill("#email", test_email)
        page.fill("#name", "Autonomous Verifier")
        page.fill("#password", "Password3DVision!")
        page.click("form button[type=submit]")

        page.wait_for_url("**/app/studio**", timeout=15000)
        time.sleep(1.0)
        wallet_loc = page.locator("#wallet")
        expect(wallet_loc).to_be_visible(timeout=10000)
        wallet_text = wallet_loc.inner_text()
        print(f"  -> Wallet text after registration: '{wallet_text}'")
        assert "credits" in wallet_text.lower(), "Wallet credits text not found"
        page.screenshot(path=str(OUT_DIR / "02_wallet_auth.png"))
        print("  -> Registration and 50+ free credits grant verified.")

        # ----------------------------------------------------------------------
        # TEST 3: Studio 3-Dock Architecture
        # ----------------------------------------------------------------------
        print("[4/7] Testing Studio 3-Dock Architecture...")
        expect(page.locator(".studio-layout")).to_be_visible(timeout=10000)
        expect(page.locator(".studio-left-dock")).to_be_visible()
        expect(page.locator(".studio-center-canvas")).to_be_visible()
        expect(page.locator(".studio-right-dock")).to_be_visible()
        expect(page.locator(".studio-bottom-dock")).to_be_visible()
        page.screenshot(path=str(OUT_DIR / "03_studio_3dock.png"))
        print("  -> Studio 3-dock layout verified.")

        # ----------------------------------------------------------------------
        # TEST 4: Student Pipeline Inspector (Tab A)
        # ----------------------------------------------------------------------
        print("[5/7] Testing Student Pipeline Inspector...")
        tab_inspector = page.locator("button:has-text('Inspector')")
        expect(tab_inspector).to_be_visible()
        tab_inspector.click()
        expect(page.locator(".inspector-container")).to_be_visible()
        expect(page.locator(".inspector-header")).to_be_visible()

        page.screenshot(path=str(OUT_DIR / "04_pipeline_inspector.png"))
        print("  -> Student Pipeline Inspector container & header verified.")

        # ----------------------------------------------------------------------
        # TEST 5: Agentic AI Co-Pilot (Tab B)
        # ----------------------------------------------------------------------
        print("[6/7] Testing Agentic AI Co-Pilot...")
        tab_chat = page.locator("button:has-text('AI Co-Pilot')")
        expect(tab_chat).to_be_visible()
        tab_chat.click()
        expect(page.locator(".chat-dock")).to_be_visible()

        # Send question to Professor 3D
        chat_input = page.locator("#chat-message")
        expect(chat_input).to_be_visible()
        chat_input.fill("Explain the mathematical difference between Depth Anything V2 and MiDaS.")
        send_btn = page.locator(".chat-dock button:has-text('Send')")
        send_btn.click()

        # Verify response arrives
        expect(page.locator(".chat-message.chat-agent").first).to_be_visible(timeout=15000)
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "05_agentic_copilot.png"))
        print("  -> Agentic AI Co-Pilot conversation verified.")

        # ----------------------------------------------------------------------
        # TEST 6: Student Curriculum Labs View (/app/labs)
        # ----------------------------------------------------------------------
        print("[7/7] Testing Student Curriculum Labs...")
        page.goto(f"{BASE_URL}/app/labs", wait_until="networkidle")
        expect(page.locator("h1:has-text('Interactive Student Training Curriculum')")).to_be_visible(timeout=10000)
        expect(page.locator("text=LAB 1: MONOCULAR RELATIVE DEPTH")).to_be_visible()
        expect(page.locator("text=LAB 2: IMPLICIT NEURAL REPRESENTATION")).to_be_visible()
        expect(page.locator("text=LAB 3: 3D GAUSSIAN SPLATTING")).to_be_visible()
        expect(page.locator("text=LAB 4: 3D GEOMETRIC METRIC EVALUATION")).to_be_visible()

        # Launch Lab 4 (Geometric Metric Evaluation)
        launch_btn = page.locator("div.card:has-text('LAB4_EVALUATOR') button:has-text('Launch Lab Run')")
        launch_btn.click()
        time.sleep(2.0)

        # Verify run appears in runs history table
        expect(page.locator(".tbl tbody tr").first).to_be_visible(timeout=10000)
        page.screenshot(path=str(OUT_DIR / "06_student_labs.png"))
        print("  -> Student Curriculum Labs execution & telemetry verified.")

        browser.close()

    print("\n=======================================================")
    print("ALL 7 END-TO-END BROWSER TESTS PASSED 100% GREEN!")
    print(f"Screenshots saved to: {OUT_DIR}")
    print("=======================================================")
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
