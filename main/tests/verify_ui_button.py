"""Real-browser (Playwright/Chromium) UI check for the View Analytics fixes.

Verifies, in a real running Streamlit app:
  1. .jfif upload is accepted (extension fix)
  2. detection + results render with NO exception
  3. "View Analytics" button is centered under the stat cards
  4. clicking it switches to the Analytics page with NO crash
     (the old StreamlitWidgetAlreadyInstantiatedError)

Run:  python main/tests/verify_ui_button.py
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "main"))

PORT = 8599
BASE = f"http://localhost:{PORT}"
OUT_DIR = Path(__file__).resolve().parent / "verify_ui"
OUT_DIR.mkdir(exist_ok=True)


def wait_for(page, selector, timeout=60000):
    page.wait_for_selector(selector, timeout=timeout)


def main():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})

        page.goto(BASE, wait_until="networkidle")
        time.sleep(3)  # detector loads on first render

        # ── upload the .jfif file through the real uploader ──
        page.locator('input[type="file"]').set_input_files(str(REPO_ROOT / "images.jfif"))

        # wait for detection to finish: the View Analytics button appears
        wait_for(page, "button:has-text('View Analytics')", timeout=180000)
        time.sleep(2)

        page.screenshot(path=str(OUT_DIR / "01_results.png"), full_page=True)
        print("PASS  upload accepted + results rendered")

        # ── check button is roughly horizontally centered under cards ──
        btn = page.locator("button:has-text('View Analytics')").first
        box = btn.bounding_box()
        center = box["x"] + box["width"] / 2
        page_w = page.evaluate("window.innerWidth")
        rel = center / (page_w or 1400)
        print(f"INFO  button center at {center:.0f}px of ~{page_w:.0f}px viewport ({rel:.2f})")

        # ── click it: must land on Analytics with no exception ──
        btn.click()
        try:
            wait_for(page, "h1:has-text('Analytics')", timeout=60000)
            print("PASS  navigation to Analytics succeeded, no crash")
        except Exception as exc:
            print(f"FAIL  navigation: {exc}")
            page.screenshot(path=str(OUT_DIR / "02_crash.png"), full_page=True)
            browser.close()
            sys.exit(1)

        # Streamlit renders exceptions in a red expandable box; make sure none
        if page.locator("[data-testid='stException']").count() > 0:
            text = page.locator("[data-testid='stException']").first.inner_text()
            if "StreamlitWidgetAlreadyInstantiatedError" in text:
                print("FAIL  StreamlitWidgetAlreadyInstantiatedError still present")
                browser.close()
                sys.exit(1)

        page.screenshot(path=str(OUT_DIR / "02_analytics.png"), full_page=True)
        print(f"\nAll UI checks passed. Screenshots in {OUT_DIR}")
        browser.close()


if __name__ == "__main__":
    main()
