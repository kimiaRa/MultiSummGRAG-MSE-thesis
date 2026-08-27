from pathlib import Path
from extractor import extract_text
from image_pipeline import download_and_filter
from geocoder import geocode_fsi


def build_fsi_record(scraped: dict, city: str, img_dir: Path,
                     seen_hashes: set,
                     country: str = "") -> dict:
    url        = scraped["url"]
    html       = scraped.get("html") or ""
    inner_text = scraped.get("inner_text")

    extracted = extract_text(html, url, inner_text)

    # attach extracted text to scraped dict so geocoder can use it
    scraped["html_text"] = extracted["text"]

    images = download_and_filter(
        scraped.get("image_urls", []), img_dir, seen_hashes
    )
    geo = geocode_fsi(scraped, city=city, country=country)

    return {
        "url":         url,
        "title":       scraped.get("title", ""),
        "text":        extracted["text"],
        "description": extracted["description"],
        "author":      extracted["author"],
        "images":      images,
        "geo":         geo,
        "city":        city,
        "district":    None,
        "fsi_type":    None,
        "scraped_at":  scraped.get("scraped_at"),
        "error":       scraped.get("error"),
    }