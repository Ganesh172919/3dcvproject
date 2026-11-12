"""Capture screenshots of every page and run a UI/UX audit.

Usage
-----
    python tools/ui_audit.py            # writes shots to docs/screenshots/

The script drives the real application with a headless Chromium: it logs in,
walks the studio, dashboard, assets, gallery and admin pages, records console
errors, network failures, horizontal overflow, clipped text and tap-target
sizes, and saves a full-page screenshot per route.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:8000"
ADMIN = ("admin@3dfun.dev", "DemoPassword123!")
PAGES = [
    ("landing", "/site/index.html", False),
    ("app-home", "/app/", False),
    ("studio", "/app/studio", True),
    ("dashboard", "/app/dashboard", True),
    ("assets", "/app/assets", True),
    ("gallery", "/app/gallery", False),
    ("admin", "/app/admin", True),
    ("benchmark", "/app/benchmark", True),
    ("settings", "/app/settings", True),
    ("model-detail", "/app/assets", True),
    ("login", "/app/login", False),
]

AUDIT_JS = """
() => {
  const out = {overflow: document.documentElement.scrollWidth - window.innerWidth,
               clipped: [], tiny: [], unlabeled: []};
  const vw = window.innerWidth, vh = window.innerHeight;
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    if (r.width === 0 || r.height === 0 || cs.visibility === 'hidden' || cs.display === 'none') continue;
    if (el.scrollHeight > el.clientHeight + 4 && cs.overflowY !== 'auto' && cs.overflowY !== 'scroll'
        && el.tagName !== 'BODY' && el.tagName !== 'HTML' && cs.overflow !== 'visible') {
      out.clipped.push({tag: el.tagName, cls: el.className && el.className.toString().slice(0,60),
                        sh: el.scrollHeight, ch: el.clientHeight});
    }
    if ((el.tagName === 'BUTTON' || el.tagName === 'A' || el.tagName === 'INPUT') && r.height < 28) {
      out.tiny.push({tag: el.tagName, txt: (el.innerText || el.value || '').slice(0, 24),
                     h: Math.round(r.height), w: Math.round(r.width)});
    }
    if (r.right > vw + 2 || r.left < -2) {
      out.unlabeled.push({tag: el.tagName, cls: (el.className||'').toString().slice(0,40),
                          left: Math.round(r.left), right: Math.round(r.right)});
    }
  }
  return out;
}
"""


def login(page) -> bool:
    """Sign in through the real form and confirm the SPA stored the token."""
    page.goto(f"{BASE}/app/login", wait_until="networkidle")
    page.wait_for_selector("#email", timeout=15000)
    page.fill("#email", ADMIN[0])
    page.fill("#password", ADMIN[1])
    page.click("button[type=submit]")
    page.wait_for_timeout(3000)
    token = page.evaluate("() => localStorage.getItem('3dfun_token')")
    return bool(token)


def main() -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    report = {"pages": [], "console": [], "network": [], "login": False}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--enable-webgl",
                                          "--use-gl=swiftshader", "--disable-dev-shm-usage"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                                  device_scale_factor=1)
        page = ctx.new_page()
        page.on("console", lambda m: report["console"].append(
            {"type": m.type, "text": m.text[:300]} if m.type in ("error", "warning") else None))
        page.on("requestfailed", lambda r: report["network"].append(
            {"url": r.url[:160], "err": (r.failure or "")[:120]}))
        report["login"] = login(page)

        for name, path, needs_auth in PAGES:
            if needs_auth and not report["login"]:
                continue
            if name == "model-detail":
                # open the first asset card if the account owns one
                page.goto(BASE + "/app/assets", wait_until="networkidle")
                page.wait_for_timeout(1500)
                card = page.query_selector(".asset-card, a[href^='/app/assets/ast_']")
                href = card.get_attribute("href") if card else None
                if not href:
                    report["pages"].append({"name": name, "path": "/app/assets", "audit": {"skipped": True}})
                    continue
                path = href.replace(BASE, "")
            page.goto(BASE + path, wait_until="networkidle")
            page.wait_for_timeout(1800)
            try:
                audit = page.evaluate(AUDIT_JS)
            except Exception as exc:                                    # noqa: BLE001
                audit = {"error": str(exc)}
            page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
            report["pages"].append({"name": name, "path": path, "audit": audit})

        # mobile pass
        ctx2 = browser.new_context(viewport={"width": 390, "height": 844})
        m = ctx2.new_page()
        for name, path in (("mobile-landing", "/site/index.html"), ("mobile-studio", "/app/studio")):
            m.goto(BASE + path, wait_until="networkidle")
            m.wait_for_timeout(1200)
            m.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
        browser.close()

    report["console"] = [c for c in report["console"] if c]
    (ROOT / "docs" / "screenshots" / "audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"login: {report['login']}")
    for entry in report["pages"]:
        a = entry["audit"]
        print(f"{entry['name']:12s} overflow={a.get('overflow')} "
              f"clipped={len(a.get('clipped', []))} tiny={len(a.get('tiny', []))} "
              f"offscreen={len(a.get('unlabeled', []))}")
    errs = [c for c in report["console"] if c["type"] == "error"]
    print(f"console errors: {len(errs)}")
    for c in errs[:10]:
        print("  -", c["text"][:160])
    print(f"failed requests: {len(report['network'])}")
    for n in report["network"][:10]:
        print("  -", n["url"], n["err"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
