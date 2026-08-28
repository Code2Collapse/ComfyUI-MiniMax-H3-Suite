#!/usr/bin/env python3
"""Headless Playwright capture for MiniMax H3 DOM widgets (GPU-box deliverable).

Requires: playwright installed in the active Python env.
ComfyUI must already be running — this script does not start it.

Example:
  python tools/playwright_widget_screenshot.py --url http://127.0.0.1:8188 --out %TEMP%/mmx_widgets.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

NODES = (
    "MiniMaxH3_TrackCrop",
    "MiniMaxH3_MaskPrep",
    "MiniMaxH3_SigmaInspector",
    "MiniMaxH3_DriftQC",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Screenshot MiniMax H3 widget nodes in ComfyUI")
    parser.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    parser.add_argument("--out", default="mmx_widgets.png", help="Output PNG path")
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed — pip install playwright && playwright install chromium", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(args.url, wait_until="networkidle", timeout=args.timeout_ms)
        page.wait_for_timeout(1500)

        for node_type in NODES:
            page.keyboard.press("Control+l")
            page.wait_for_timeout(200)
            page.keyboard.type(node_type)
            page.wait_for_timeout(400)
            page.keyboard.press("Enter")
            page.wait_for_timeout(600)

        page.screenshot(path=str(out), full_page=True)
        browser.close()

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
