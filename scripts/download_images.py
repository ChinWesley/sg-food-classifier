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
    "chicken_rice": ["hainanese chicken rice", "singapore chicken rice hawker", "roasted chicken rice plate", "steamed chicken rice singapore"],
    "laksa": ["singapore laksa bowl", "katong laksa", "curry laksa noodles", "laksa hawker centre"],
    "char_kway_teow": ["char kway teow singapore", "fried kway teow hawker", "char kway teow plate", "char kway teow cockles"],
    "nasi_lemak": ["nasi lemak singapore", "nasi lemak plate", "nasi lemak fried chicken", "nasi lemak sambal ikan bilis"],
    "roti_prata": ["roti prata curry", "roti prata singapore", "roti canai curry", "plain prata egg prata"],
    "satay": ["satay skewers peanut sauce", "singapore satay", "chicken satay plate", "lau pa sat satay"],
    "chilli_crab": ["singapore chilli crab", "chilli crab dish", "chilli crab mantou", "chilli crab restaurant singapore"],
    "kaya_toast": ["kaya toast soft boiled eggs", "kaya butter toast singapore", "ya kun kaya toast", "kaya toast kopi"],
    "bak_chor_mee": ["bak chor mee", "minced meat noodles singapore", "bak chor mee dry", "tai hwa bak chor mee"],
    "hokkien_mee": ["singapore hokkien mee", "fried hokkien prawn mee", "hokkien mee sambal lime", "hokkien prawn noodles hawker"],
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
        out_path = out_dir / f"{digest[:12]}.jpg"
        if digest in seen or out_path.exists():
            return False
        img = Image.open(io.BytesIO(resp.content))
        img = img.convert("RGBA").convert("RGB") if img.mode == "P" else img.convert("RGB")
        if min(img.size) < 150:  # skip thumbnails/icons
            return False
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(out_path, "JPEG", quality=90)
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
        per_query = 100  # over-fetch; some URLs fail
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
