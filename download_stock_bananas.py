#!/usr/bin/env python3
"""
download_stock_bananas.py
-------------------------
Download banana images from free stock photo sites for training data.
Saves to Banana-Ripeness-Bunch/inputs/stock_photos/

Usage:
    python download_stock_bananas.py
"""

import os
import requests
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "inputs" / "stock_photos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated list of direct image URLs from free stock sites
# All are CC0 / free to use for commercial/research purposes
IMAGE_URLS = [
    # Unsplash - retail / market / crate settings
    ("unsplash_fruit_stand_1.jpg", "https://images.unsplash.com/photo-1542283237-5d3c7c9b9e5b?w=800&q=80"),
    ("unsplash_fruit_stand_2.jpg", "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=800&q=80"),
    ("unsplash_banana_display.jpg", "https://images.unsplash.com/photo-1603833665858-e61d17a86271?w=800&q=80"),
    ("unsplash_green_bananas.jpg", "https://images.unsplash.com/photo-1581769337359-53f7ceff381f?w=800&q=80"),
    ("unsplash_banana_pile.jpg", "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=800&q=80"),
    ("unsplash_banana_bunch_1.jpg", "https://images.unsplash.com/photo-1596802106686-2c7d3e6e4e8e?w=800&q=80"),
    ("unsplash_market_bananas.jpg", "https://images.unsplash.com/photo-1603833665858-e61d17a86271?w=800&q=80"),
    # Pexels / Pickpik - crate / box settings
    ("pexels_banana_crate.jpg", "https://images.pexels.com/photos/1398879/pexels-photo-1398879.jpeg?w=800&q=80"),
    ("pexels_bananas_1.jpg", "https://images.pexels.com/photos/1092628/pexels-photo-1092628.jpeg?w=800&q=80"),
    ("pexels_bananas_2.jpg", "https://images.pexels.com/photos/2414038/pexels-photo-2414038.jpeg?w=800&q=80"),
    ("pexels_banana_bunch.jpg", "https://images.pexels.com/photos/2872751/pexels-photo-2872751.jpeg?w=800&q=80"),
    ("pickpik_banana_crate.jpg", "https://get.pickpik.com/full/2582/banana-fruit-market-crate.jpg"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def download_image(filename, url):
    """Download a single image, skip if already exists."""
    dest = OUTPUT_DIR / filename
    if dest.exists():
        print(f"  [SKIP] {filename} (already exists)")
        return True

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type:
            print(f"  [SKIP] {filename} (not an image: {content_type})")
            return False

        dest.write_bytes(resp.content)
        size_kb = dest.stat().st_size / 1024
        print(f"  [OK]   {filename} ({size_kb:.0f} KB)")
        return True

    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        return False


def main():
    print(f"\nDownloading banana stock photos to: {OUTPUT_DIR}")
    print(f"Total images to try: {len(IMAGE_URLS)}\n")

    success = 0
    fail = 0

    for filename, url in IMAGE_URLS:
        if download_image(filename, url):
            success += 1
        else:
            fail += 1
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"  Downloaded: {success}")
    print(f"  Skipped/Failed: {fail}")
    print(f"  Total in folder: {len(list(OUTPUT_DIR.glob('*')))}")
    print(f"{'='*50}")

    if success == 0:
        print("\n  All downloads failed. Try manual download from:")
        print("  - https://unsplash.com/s/photos/banana-bunch")
        print("  - https://www.pexels.com/search/banana/")
        print("  - https://www.pickpik.com/search?q=banana")


if __name__ == "__main__":
    main()
