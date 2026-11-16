"""Automated Playwright browser test for Iteration 3:
Student 3D Construction Visualizer (8 Stages) & Training Lab Modules.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8010"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports", "iteration3")
os.makedirs(REPORTS_DIR, exist_ok=True)


def test_iteration3():
    print("[1/5] Launching browser to verify Iteration 3 Inspector & Labs...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Step 1: Instant Demo Login
        print("[2/5] Entering Instant Demo Sandbox...")
        page.goto(f"{BASE_URL}/app/login")
        page.wait_for_selector("button:has-text('One-Click Instant Demo Sandbox')", timeout=10000)
        page.click("button:has-text('One-Click Instant Demo Sandbox')")
        page.wait_for_url("**/app/studio*", timeout=15000)
        page.wait_for_selector(".studio-layout", timeout=15000)

        # Load chair sample
        print("  -> Loading Studio with Chair Sample Asset...")
        page.goto(f"{BASE_URL}/app/studio?sample=chair")
        page.wait_for_selector(".studio-layout", timeout=15000)
        page.wait_for_timeout(2000)
        page.screenshot(path=os.path.join(REPORTS_DIR, "01_studio_sample_chair.png"))

        # Step 2: Switch to Student Pipeline Inspector Tab
        print("[3/5] Inspecting 8-Stage Student Pipeline Visualizer...")
        tab_inspector = page.locator("button:has-text('Student Pipeline Inspector')")
        if tab_inspector.is_visible():
            tab_inspector.click()
            page.wait_for_timeout(500)

        page.wait_for_selector(".inspector-container", timeout=10000)
        page.screenshot(path=os.path.join(REPORTS_DIR, "02_inspector_open.png"))

        # Verify all 8 stage buttons are available
        stage_buttons = page.locator(".inspector-stage-btn")
        count = stage_buttons.count()
        print(f"  -> Found {count} stage navigation buttons.")
        assert count == 8, f"Expected 8 pipeline stages, got {count}"

        # Test Stage 1: Preprocessing & Matting
        print("  -> Testing S1: Preprocessing & Alpha Matting...")
        stage_buttons.nth(0).click()
        page.wait_for_timeout(400)
        assert page.locator(".inspector-stage-head").is_visible()
        page.screenshot(path=os.path.join(REPORTS_DIR, "03_stage1_preprocessing.png"))

        # Test Stage 2: Monocular Depth
        print("  -> Testing S2: Monocular Depth Estimation...")
        stage_buttons.nth(1).click()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "04_stage2_depth.png"))

        # Test Stage 3: Camera Ray Unprojection & Point Cloud
        print("  -> Testing S3: Camera Ray Unprojection & Point Cloud...")
        stage_buttons.nth(2).click()
        page.wait_for_timeout(400)
        assert "Point Cloud" in page.locator(".inspector-body").inner_text()
        page.screenshot(path=os.path.join(REPORTS_DIR, "05_stage3_pointcloud.png"))

        # Test Stage 4: Volumetric Inflation & Slicing Plane
        print("  -> Testing S4: Volumetric Inflation & Slicing...")
        stage_buttons.nth(3).click()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "06_stage4_volume_sdf.png"))

        # Test Stage 5: Marching Cubes Geometry
        print("  -> Testing S5: Marching Cubes Isosurface Geometry...")
        stage_buttons.nth(4).click()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "07_stage5_geometry.png"))

        # Test Stage 6: Smoothing
        print("  -> Testing S6: Taubin Mesh Smoothing...")
        stage_buttons.nth(5).click()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "08_stage6_smoothing.png"))

        # Test Stage 7: UV Atlas
        print("  -> Testing S7: Planar UV Unwrapping...")
        stage_buttons.nth(6).click()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "09_stage7_uv.png"))

        # Test Stage 8: PBR Texturing
        print("  -> Testing S8: PBR Texture Back-Projection...")
        stage_buttons.nth(7).click()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "10_stage8_texture.png"))

        # Step 3: Test Student Curriculum Labs View
        print("[4/5] Testing Student Curriculum Labs View (/app/labs)...")
        page.goto(f"{BASE_URL}/app/labs")
        page.wait_for_selector(".lab-card", timeout=10000)
        page.wait_for_timeout(1000)
        page.screenshot(path=os.path.join(REPORTS_DIR, "11_labs_catalog.png"))

        lab_cards = page.locator(".lab-card")
        lab_count = lab_cards.count()
        print(f"  -> Found {lab_count} active curriculum labs.")
        assert lab_count >= 4, f"Expected at least 4 curriculum labs, got {lab_count}"

        # Trigger Lab evaluation run
        print("[5/5] Launching Lab Geometric Benchmark Run...")
        run_btn = page.locator("button:has-text('Launch Lab Run')").first
        if run_btn.is_visible():
            run_btn.click()
            page.wait_for_timeout(2500)
            page.screenshot(path=os.path.join(REPORTS_DIR, "12_lab_run_launched.png"))

        browser.close()
        print("\n=======================================================")
        print("ALL ITERATION 3 INSPECTOR & LABS TESTS PASSED 100%!")
        print(f"Screenshots saved to: {REPORTS_DIR}")
        print("=======================================================")


if __name__ == "__main__":
    test_iteration3()
