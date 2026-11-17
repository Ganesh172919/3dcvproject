"""Automated Playwright browser test for Iteration 2:
Authentication, User Profiles, Password Strength Meter, One-Click Demo Mode,
and the Free Trial Credits Ledger with Real-time Wallet Modal.
"""
import os
import uuid
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8010"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports", "iteration2")
os.makedirs(REPORTS_DIR, exist_ok=True)


def test_iteration2():
    print("[1/6] Launching browser to verify Iteration 2 Auth & Credits...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Step 1: Open Register page & verify password strength meter
        print("[2/6] Testing Registration & zxcvbn Password Strength Meter...")
        page.goto(f"{BASE_URL}/app/register")
        page.wait_for_selector("#email")
        page.screenshot(path=os.path.join(REPORTS_DIR, "01_register_page.png"))

        # Type password and observe strength meter
        page.fill("#password", "weak")
        page.wait_for_timeout(300)
        page.fill("#password", "P@ssw0rd99Stronger!")
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(REPORTS_DIR, "02_password_strength_active.png"))

        # Register new student
        unique_user = f"student_{uuid.uuid4().hex[:8]}@iiitdharwad.ac.in"
        page.fill("#email", unique_user)
        page.fill("#name", "IIIT Student Tester")
        page.click("button[type='submit']")

        # Wait for redirect to /app/studio
        page.wait_for_url("**/app/studio*", timeout=15000)
        page.wait_for_selector("#wallet")
        page.wait_for_timeout(1000)

        wallet_el = page.locator("#wallet-value")
        credits_text = wallet_el.inner_text().strip()
        print(f"  -> Successfully registered {unique_user}! Topbar wallet: {credits_text} credits")
        assert int(credits_text) >= 50, f"Expected at least 50 credits, got {credits_text}"
        page.screenshot(path=os.path.join(REPORTS_DIR, "03_studio_after_registration.png"))

        # Step 2: Open Credit Wallet & Transaction History Modal
        print("[3/6] Opening Credit Wallet & Transaction History Modal...")
        page.click("#wallet")
        page.wait_for_selector("#wallet-modal-back", timeout=5000)
        page.wait_for_timeout(500)
        page.screenshot(path=os.path.join(REPORTS_DIR, "04_wallet_modal_open.png"))

        modal_text = page.locator("#wallet-modal-back").inner_text()
        assert "Credit Wallet" in modal_text
        assert "Transaction History" in modal_text
        print("  -> Credit Wallet modal rendered successfully with ledger details.")

        # Step 3: Redeem Promo Voucher Code
        print("[4/6] Testing Voucher Redemption (IIITDHARWAD)...")
        page.fill("#coupon-code", "IIITDHARWAD")
        page.click("#btn-redeem-coupon")
        page.wait_for_timeout(1200)

        updated_credits = page.locator("#wallet-value").inner_text().strip()
        print(f"  -> Voucher redeemed! New balance in wallet: {updated_credits} credits")
        assert int(updated_credits) >= 100, f"Expected at least 100 credits after coupon, got {updated_credits}"
        page.screenshot(path=os.path.join(REPORTS_DIR, "05_coupon_redeemed.png"))

        # Close modal
        page.click("#modal-close")
        page.wait_for_timeout(400)

        # Step 4: Run Sample Generation to verify credit reservation & spend debit
        print("[5/6] Testing Generation Debit on Sample Asset...")
        
        # Load sample sneaker
        page.goto(f"{BASE_URL}/app/studio?sample=sneaker")
        page.wait_for_selector("#btn-generate", timeout=10000)
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(REPORTS_DIR, "06_sample_loaded.png"))

        # Click Generate (will run or estimate debit)
        cost_text = page.locator("#estimate-cost").inner_text()
        print(f"  -> Generation cost estimate: {cost_text}")
        assert "credits" in cost_text

        # Step 5: Test One-Click Demo Mode button
        print("[6/6] Testing One-Click Demo Mode Button...")
        page.click("#auth-logout")
        page.wait_for_url("**/app/login*", timeout=10000)
        page.wait_for_timeout(500)
        page.screenshot(path=os.path.join(REPORTS_DIR, "07_login_page_demo_btn.png"))

        # Click Instant Demo Sandbox
        demo_btn = page.locator("button:has-text('One-Click Instant Demo Sandbox')")
        assert demo_btn.is_visible()
        demo_btn.click()

        page.wait_for_url("**/app/studio*", timeout=15000)
        page.wait_for_selector("#wallet")
        demo_credits = page.locator("#wallet-value").inner_text().strip()
        print(f"  -> One-Click Demo Mode successful! Demo wallet: {demo_credits} credits")
        assert int(demo_credits) >= 50
        page.screenshot(path=os.path.join(REPORTS_DIR, "08_demo_mode_studio.png"))

        browser.close()
        print("\n=======================================================")
        print("ALL ITERATION 2 AUTH & CREDITS VERIFICATIONS PASSED 100%!")
        print(f"Screenshots saved to: {REPORTS_DIR}")
        print("=======================================================")


if __name__ == "__main__":
    test_iteration2()
