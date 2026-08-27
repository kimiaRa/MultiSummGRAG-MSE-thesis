import hashlib, requests
from pathlib import Path
from PIL import Image
import imagehash
from io import BytesIO
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import IMAGE
HEADERS = {"User-Agent": "fsi-thesis-bot/1.0"}


def download_and_filter(image_urls: list[str], out_dir: Path,
                        seen_hashes: set) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for url in image_urls[:IMAGE["max_per_page"] * 3]:
        if not url.startswith("http"):
            continue
        try:
            resp = requests.get(url, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")

            if img.width < IMAGE["min_width"] or img.height < IMAGE["min_height"]:
                continue

            ph = imagehash.phash(img)
            if any(abs(ph - h) < IMAGE["hash_threshold"] for h in seen_hashes):
                continue
            seen_hashes.add(ph)

            fname = hashlib.md5(url.encode()).hexdigest()[:12] + ".jpg"
            fpath = out_dir / fname
            img.save(fpath, "JPEG", quality=85)

            saved.append({
                "source_url": url,
                "local_path": str(fpath),
                "width":      img.width,
                "height":     img.height,
                "caption":    "",        # filled by BLIP-2 in Phase 2
            })

            if len(saved) >= IMAGE["max_per_page"]:
                break
        except Exception:
            continue

    return saved