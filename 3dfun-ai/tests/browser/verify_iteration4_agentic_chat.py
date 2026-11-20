"""Automated Playwright browser test for Iteration 4:
Agentic AI Chat Copilot with Autonomous Tool Calling & Dual Personas.
"""
import os
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8010"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports", "iteration4")
os.makedirs(REPORTS_DIR, exist_ok=True)


def test_iteration4_chat():
    print("[1/5] Launching browser to verify Iteration 4 Agentic AI Chat...")
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
        page.wait_for_timeout(1000)

        # Step 2: Open Embedded Chat Dock in Studio
        print("[3/5] Testing Studio Dock 3D AI Chat Tab...")
        chat_tab = page.locator("button:has-text('3D AI Chat')")
        if not chat_tab.is_visible():
            chat_tab = page.locator("button:has-text('AI Co-Pilot')")
        assert chat_tab.is_visible(), "3D AI Chat tab button not found in studio dock"
        chat_tab.click()
        page.wait_for_timeout(600)

        page.wait_for_selector(".chat-dock", timeout=10000)
        page.screenshot(path=os.path.join(REPORTS_DIR, "01_studio_chat_dock_open.png"))

        # Verify 3D AI Chat Persona default
        chat_btn = page.locator("button:has-text('3D AI Chat')").first
        studio_btn = page.locator("button:has-text('3D Studio Tools')")
        assert chat_btn.is_visible(), "3D AI Chat button missing"
        assert studio_btn.is_visible(), "3D Studio Tools button missing"

        # Ask academic question via prompt chip
        print("  -> Asking Academic Question (Marching Cubes & Euler Characteristic)...")
        chip_marching = page.locator(".chat-chips button:has-text('Marching Cubes')").first
        if chip_marching.is_visible():
            chip_marching.click()
        else:
            page.fill("#chat-message", "What is Marching Cubes and the Euler characteristic?")

        page.wait_for_timeout(300)
        send_btn = page.locator(".chat-form button[type='submit']")
        send_btn.click()

        # Wait for reply
        page.wait_for_selector(".chat-agent", timeout=15000)
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(REPORTS_DIR, "02_studio_chat_academic_response.png"))

        # Switch to Studio Director persona
        print("  -> Switching to Studio Director Persona...")
        studio_btn.click()
        page.wait_for_timeout(500)
        page.screenshot(path=os.path.join(REPORTS_DIR, "03_studio_director_switched.png"))

        # Command an autonomous tool execution: Inspect depth tensor statistics
        print("  -> Commanding Autonomous Tool: inspect_stage_tensor...")
        page.fill("#chat-message", "Inspect depth tensor statistics and dynamic range")
        send_btn.click()

        # Wait for agent tool execution badge
        page.wait_for_selector(".tool-execution-card", timeout=15000)
        page.wait_for_timeout(1500)
        tool_card = page.locator(".tool-execution-card").last
        assert tool_card.is_visible(), "Tool execution badge card not rendered"
        print("  -> Tool execution badge verified successfully.")
        page.screenshot(path=os.path.join(REPORTS_DIR, "04_tool_executed_badge.png"))

        # Step 3: Test Fullscreen Dedicated Chat View (/app/chat)
        print("[4/5] Testing Dedicated Fullscreen Chat View (/app/chat)...")
        page.goto(f"{BASE_URL}/app/chat")
        page.wait_for_selector(".studio-title", timeout=10000)
        page.wait_for_selector(".chat-dock", timeout=10000)

        # Submit NeRF vs 3DGS question
        print("  -> Asking CV Benchmark Comparison: NeRF vs 3DGS...")
        page.fill("#chat-message", "Compare NeRF vs 3D Gaussian Splatting (3DGS)")
        page.click(".chat-form button[type='submit']")
        page.wait_for_selector(".chat-agent", timeout=15000)
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(REPORTS_DIR, "05_fullscreen_chat_nerf_vs_3dgs.png"))

        browser.close()
        print("\n=======================================================")
        print("ALL ITERATION 4 AGENTIC AI CHAT TESTS PASSED 100%!")
        print(f"Screenshots saved to: {REPORTS_DIR}")
        print("=======================================================")


if __name__ == "__main__":
    test_iteration4_chat()
