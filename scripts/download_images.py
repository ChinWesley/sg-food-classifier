"""Download training images for each Singapore dish via DuckDuckGo image search.

Usage:
    python scripts/download_images.py              # all classes, 150 images each
    python scripts/download_images.py --per-class 60 --only laksa satay

Images land in data/<class_name>/. Re-running skips classes that already have
enough images. After downloading, skim each folder and delete wrong/irrelevant
pictures. That manual clean-up matters more than anything else for accuracy.
"""

import argparse
import hashlib
import io
import time
from pathlib import Path

import requests
from ddgs import DDGS
from PIL import Image

# class folder name -> search queries (several queries = more variety)
CLASSES = {
    "chicken_rice": ["hainanese chicken rice", "singapore chicken rice hawker"],
    "laksa": ["singapore laksa bowl", "katong laksa"],
    "char_kway_teow": ["char kway teow singapore", "fried kway teow hawker"],
    "nasi_lemak": ["nasi lemak singapore", "nasi lemak plate"],
    "roti_prata": ["roti prata curry", "roti prata singapore"],
    "satay": ["satay skewers peanut sauce", "singapore satay"],
    "chilli_crab": ["singapore chilli crab", "chilli crab dish"],
    "kaya_toast": ["kaya toast soft boiled eggs", "kaya butter toast singapore"],
    "bak_chor_mee": ["bak chor mee", "minced meat noodles singapore"],
    "hokkien_mee": ["singapore hokkien mee", "fried hokkien prawn mee"],
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MAX_SIDE = 512  # downscale big images to save disk space; training uses 224px


def fetch_urls(query: str, n: int) -> list[str]:
    for attempt in range(3):
        try:
            results = DDGS().images(query, max_results=n)
            return [r["image"] for r in results if r.get("image")]
        except Exception as e:  # rate limits happen; back off and retry
            wait = 10 * (attempt + 1)
            print(f"  search failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return []


def save_image(url: str, out_dir: Path, seen: set[str]) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        digest = hashlib.md5(resp.content).hexdigest()
        if digest in seen:
            return False
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if min(img.size) < 150:  # skip thumbnails/icons
            return False
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(out_dir / f"{digest[:12]}.jpg", "JPEG", quality=90)
        seen.add(digest)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--only", nargs="*", help="subset of class names")
    args = parser.parse_args()

    for name, queries in CLASSES.items():
        if args.only and name not in args.only:
            continue
        out_dir = DATA_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("*.jpg")))
        if existing >= args.per_class:
            print(f"{name}: already has {existing} images, skipping")
            continue

        seen: set[str] = set()
        saved = existing
        per_query = args.per_class // len(queries) + 40  # over-fetch; some URLs fail
        for q in queries:
            for url in fetch_urls(q, per_query):
                if saved >= args.per_class:
                    break
                if save_image(url, out_dir, seen):
                    saved += 1
            time.sleep(2)
        print(f"{name}: {saved} images")


if __name__ == "__main__":
    main()
