import trafilatura
from trafilatura.settings import use_config

_cfg = use_config()
_cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")


def extract_text(html: str, url: str = "",
                 inner_text: str = None) -> dict:
    if not html:
        return {
            "text":        inner_text or "",
            "description": "",
            "author":      "",
        }

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        config=_cfg,
    )
    meta = trafilatura.extract_metadata(html, default_url=url)

    if (not text or len(text) < 100) and inner_text and \
            len(inner_text) > len(text or ""):
        text = inner_text

    return {
        "text":        text or "",
        "description": meta.description if meta else "",
        "author":      meta.author      if meta else "",
    }