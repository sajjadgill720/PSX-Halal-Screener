"""PSX company news lookup and risk flagging."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search

from .error_utils import log_exception, log_message


NEWS_KEYWORDS = ["interest", "loan", "alcohol", "gambling", "fine", "violation"]


def get_company_news(ticker: str, company_name: str) -> list:
    """Search the web for recent company news items."""

    query = f"{company_name} Pakistan haram halal interest riba alcohol tobacco"
    try:
        search_results = list(google_search(query, num_results=5, lang="en", advanced=True))
    except TypeError:
        try:
            search_results = list(google_search(query, num_results=5, lang="en"))
        except Exception as exc:
            log_exception(f"News search failed for {ticker}")
            log_message(f"News search error for {ticker}: {exc}")
            return []
    except Exception as exc:
        log_exception(f"News search failed for {ticker}")
        log_message(f"News search error for {ticker}: {exc}")
        return []

    items: list[dict] = []
    for result in search_results[:5]:
        title, url, snippet = _extract_fields(result)
        if not snippet and url:
            snippet = _fetch_snippet(url)
        items.append(
            {
                "title": title or company_name,
                "url": url,
                "snippet": snippet,
                "date": datetime.now(timezone.utc).date().isoformat(),
            }
        )
    return items


def flag_news_concerns(ticker: str, company_name: str) -> dict:
    """Flag concern keywords in recent news snippets."""

    news_items = get_company_news(ticker, company_name)
    concern_keywords: list[str] = []
    for item in news_items:
        snippet = str(item.get("snippet", "")).lower()
        for keyword in NEWS_KEYWORDS:
            if keyword in snippet and keyword not in concern_keywords:
                concern_keywords.append(keyword)

    return {
        "has_concerns": bool(concern_keywords),
        "concern_keywords": concern_keywords,
        "news_items": news_items,
    }


def _extract_fields(result: object) -> tuple[str, str, str]:
    title = getattr(result, "title", "") if not isinstance(result, dict) else result.get("title", "")
    url = getattr(result, "url", "") if not isinstance(result, dict) else result.get("url", "")
    snippet = (
        getattr(result, "description", "")
        if not isinstance(result, dict)
        else result.get("description", result.get("snippet", ""))
    )
    return str(title or ""), str(url or ""), str(snippet or "")


def _fetch_snippet(url: str) -> str:
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return str(meta.get("content"))
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:240]
    except Exception:
        return ""