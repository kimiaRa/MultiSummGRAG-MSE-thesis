import asyncio, csv, time, requests
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from playwright_stealth import Stealth
_stealth = Stealth()
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import SCRAPER

_WP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── WordPress REST API helper ─────────────────────────────────────────────────

def _is_wordpress_com(url: str) -> bool:
    return urlparse(url).netloc.lower().endswith(".wordpress.com")


def _fetch_wordpress_api(url: str) -> dict | None:
    """
    Fetch content via WordPress API.

    For *.wordpress.com blogs: uses the WordPress.com public REST API
    (public-api.wordpress.com/rest/v1.1), which is always available.

    For self-hosted WordPress sites: tries the WP REST v2 endpoint
    (/wp-json/wp/v2/posts), useful as a bot-block fallback.

    Returns a scraper-compatible dict, or None if the site is not WordPress.
    """
    parsed  = urlparse(url)
    domain  = parsed.netloc.lower()
    headers = {"User-Agent": _WP_UA}
    texts, images, main_title = [], [], ""

    if domain.endswith(".wordpress.com"):
        # WordPress.com public API — always available, no auth needed
        for content_type in ("posts", "pages"):
            try:
                r = requests.get(
                    f"https://public-api.wordpress.com/rest/v1.1/sites/{domain}/{content_type}",
                    params={"number": 20, "fields": "title,content,featured_image"},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                items = data.get(content_type, [])
                for item in items:
                    title   = BeautifulSoup(item.get("title", ""), "html.parser").get_text(strip=True)
                    content = BeautifulSoup(item.get("content", ""), "html.parser").get_text(separator="\n", strip=True)
                    if title and not main_title:
                        main_title = title
                    if title:
                        texts.append(title)
                    if content:
                        texts.append(content)
                    img = item.get("featured_image")
                    if img:
                        images.append(img)
            except Exception:
                continue

    else:
        # Self-hosted WordPress — try WP REST v2
        api_base = f"{parsed.scheme}://{parsed.netloc}/wp-json/wp/v2"
        for endpoint in ("posts", "pages"):
            try:
                r = requests.get(
                    f"{api_base}/{endpoint}",
                    params={"per_page": 20,
                            "_fields": "title,content,featured_media_src_url"},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code != 200:
                    continue
                items = r.json()
                if not isinstance(items, list):
                    continue
                for item in items:
                    title = BeautifulSoup(
                        item.get("title", {}).get("rendered", ""), "html.parser"
                    ).get_text(strip=True)
                    content = BeautifulSoup(
                        item.get("content", {}).get("rendered", ""), "html.parser"
                    ).get_text(separator="\n", strip=True)
                    if title and not main_title:
                        main_title = title
                    if title:
                        texts.append(title)
                    if content:
                        texts.append(content)
                    img = item.get("featured_media_src_url")
                    if img:
                        images.append(img)
            except Exception:
                continue

    if not texts:
        return None

    return {
        "title":      main_title,
        "html":       "\n\n".join(texts),
        "image_urls": images,
        "scraped_at": time.time(),
        "error":      None,
    }


# ── Playwright helpers ────────────────────────────────────────────────────────

def _same_domain(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc == urlparse(url_b).netloc


async def _internal_links(page, base_url: str, limit: int) -> list[str]:
    hrefs = await page.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.href)"
    )
    base_norm = base_url.rstrip("/")
    seen, links = set(), []
    for href in hrefs:
        href = href.split("#")[0].rstrip("/")
        if not href.startswith("http"):
            continue
        if href in seen or not _same_domain(href, base_url) or href == base_norm:
            continue
        seen.add(href)
        links.append(href)
        if len(links) >= limit:
            break
    return links


async def scrape_url(page, url: str, retries: int = SCRAPER["max_retries"]) -> dict:
    for attempt in range(retries):
        try:
            await page.goto(url, timeout=SCRAPER["timeout_ms"],
                            wait_until=SCRAPER["wait_until"])
            html  = await page.content()
            title = await page.title()


            # detect bot challenge / block pages
            _block_signals = [
                "checking your browser", "just a moment", "please wait",
                "403 error", "request blocked", "access denied",
                "enable javascript", "voight-kampff",
                "validation request", "user validation required",
                "captcha", "prove you are human",
            ]
            title_lower = title.lower()
            html_lower  = html.lower()[:500]
            if any(s in title_lower or s in html_lower for s in _block_signals):
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"url": url, "html": None, "image_urls": [],
                        "scraped_at": time.time(), "error": "bot_block"}

            img_srcs = await page.eval_on_selector_all(
                "img", "els => els.map(e => e.src).filter(Boolean)"
            )
            return {
                "url":        url,
                "title":      title,
                "html":       html,
                "image_urls": list(set(img_srcs)),
                "scraped_at": time.time(),
                "error":      None,
            }
        except PWTimeout:
            if attempt == retries - 1:
                return {"url": url, "html": None, "image_urls": [],
                        "scraped_at": time.time(), "error": "timeout"}
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            return {"url": url, "html": None, "image_urls": [],
                    "scraped_at": time.time(), "error": str(e)}


# ── Main scrape logic ─────────────────────────────────────────────────────────

async def scrape_city(city_cfg: dict, rows: list[dict]) -> list[dict]:
    raw_dir = Path(city_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    sem     = asyncio.Semaphore(SCRAPER["concurrency"])
    visited = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )

        async def scrape_one(url: str, lat, lng, coords_approx: bool,
                              depth: int) -> list[dict]:
            if url in visited:
                return []
            visited.add(url)

            async with sem:
                # ── WordPress.com: REST API first, Playwright fallback ─────────
                if _is_wordpress_com(url):
                    api = await asyncio.to_thread(_fetch_wordpress_api, url)
                    if api:
                        api.update({"url": url, "Lat": lat, "Lng": lng,
                                    "coords_approx": coords_approx})
                        _save_raw(raw_dir, url, api["html"])
                        return [api]
                    # API failed → fall through to Playwright

                # ── Playwright ────────────────────────────────────────────────
                page = await browser.new_page(user_agent=_WP_UA)
                await _stealth.apply_stealth_async(page)
                try:
                    result = await scrape_url(page, url)
                    result["Lat"]           = lat
                    result["Lng"]           = lng
                    result["coords_approx"] = coords_approx
                    if result["html"]:
                        _save_raw(raw_dir, url, result["html"])

                    # ── Bot-blocked: try WordPress API as fallback ─────────────
                    if result.get("error") == "bot_block":
                        api = await asyncio.to_thread(_fetch_wordpress_api, url)
                        if api:
                            api.update({"url": url, "Lat": lat, "Lng": lng,
                                        "coords_approx": coords_approx})
                            _save_raw(raw_dir, url, api["html"])
                            return [api]

                    links = []
                    if depth < SCRAPER["max_depth"] and result["html"]:
                        links = await _internal_links(
                            page, url, SCRAPER["max_subpages"]
                        )
                finally:
                    await page.close()

            sub = await asyncio.gather(*[
                scrape_one(link, lat, lng, coords_approx, depth + 1)
                for link in links
            ])
            return [result] + [r for group in sub for r in group]

        def _parse_approx(val) -> bool:
            return str(val).strip().lower() == "true"

        groups = await asyncio.gather(*[
            scrape_one(
                row["URL"], row.get("Lat"), row.get("Lng"),
                _parse_approx(row.get("coords_approx", False)),
                depth=0,
            )
            for row in rows
        ])
        await browser.close()

    return [_merge_group(group) for group in groups]


def _save_raw(raw_dir: Path, url: str, content: str):
    slug = url.replace("https://", "").replace("/", "_")[:80]
    (raw_dir / f"{slug}.html").write_text(content, encoding="utf-8", errors="replace")


def _merge_group(group: list[dict]) -> dict:
    """Merges seed + subpage results into one record per FSI."""
    if len(group) == 1:
        return group[0]
    seed       = group[0]
    html_parts = [r["html"] for r in group if r.get("html")]
    all_images = list({img for r in group for img in r.get("image_urls", [])})
    return {
        **seed,
        "html":       "\n".join(html_parts) if html_parts else None,
        "image_urls": all_images,
    }


def load_urls(path: str) -> list[dict]:
    """Reads urls_cleaned.csv with columns: URL, Lat, Lng[, coords_approx]"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("URL")]
