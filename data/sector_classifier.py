"""Sector classification helpers for PSX companies."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf
from anthropic import Anthropic
from bs4 import BeautifulSoup
from googlesearch import search as google_search

from .error_utils import log_exception, log_message


HARAM_SECTORS_KEYWORDS = {
    "Banking (Conventional)": ["bank", "banking", "financial services", "credit", "leasing", "microfinance"],
    "Insurance (Conventional)": ["insurance", "assurance"],
    "Alcohol": ["brewery", "beverages", "distillery", "wine", "spirits"],
    "Tobacco": ["tobacco", "cigarette", "smoking"],
    "Gambling": ["casino", "lottery", "gaming", "betting"],
    "Weapons": ["defence", "defense", "arms", "ammunition", "weapons"],
}

HALAL_SECTORS = [
    "Oil & Gas",
    "Cement",
    "Fertilizer",
    "Textile",
    "Pharmaceutical",
    "Food & Personal Care",
    "Technology",
    "Automobile",
    "Steel",
    "Power Generation",
    "Chemical",
    "Engineering",
    "Agriculture",
]

_HALAL_EXCEPTIONS = {
    "islamic banking": "Islamic Banking",
    "islamic bank": "Islamic Banking",
    "takaful": "Takaful Insurance",
    "family takaful": "Takaful Insurance",
    "general takaful": "Takaful Insurance",
}


def classify_sector_rule_based(ticker: str, company_name: str, sector_raw: str) -> dict:
    """Classify a PSX company with keyword rules."""

    text = _combined_text(company_name, sector_raw)
    lower_text = text.lower()

    for exception, label in _HALAL_EXCEPTIONS.items():
        if exception in lower_text:
            return {
                "ticker": ticker,
                "sector_classified": label,
                "is_potentially_haram": False,
                "confidence": "high",
                "method": "rule_based",
                "haram_reason": None,
                "business_summary": "",
            }

    for sector, keywords in HARAM_SECTORS_KEYWORDS.items():
        if _contains_any(lower_text, keywords):
            return {
                "ticker": ticker,
                "sector_classified": sector,
                "is_potentially_haram": True,
                "confidence": "high",
                "method": "rule_based",
                "haram_reason": f"Matched prohibited business keywords for {sector}.",
                "business_summary": "",
            }

    normalized_sector = _normalize_sector_label(sector_raw) or _infer_halal_sector(lower_text) or "Unclassified"
    confidence = "high" if normalized_sector != "Unclassified" else "low"
    return {
        "ticker": ticker,
        "sector_classified": normalized_sector,
        "is_potentially_haram": False,
        "confidence": confidence,
        "method": "rule_based",
        "haram_reason": None,
        "business_summary": "",
    }


def classify_sector_with_search(ticker: str, company_name: str) -> dict:
    """Use web search snippets to refine the sector classification."""

    query = f"{company_name} PSX Pakistan business activities"
    try:
        search_results = list(google_search(query, num_results=3, lang="en", advanced=True))
    except TypeError:
        try:
            search_results = list(google_search(query, num_results=3, lang="en"))
        except Exception as exc:
            log_exception(f"Search lookup failed for {ticker}")
            log_message(f"Search error for {ticker}: {exc}")
            return {
                "ticker": ticker,
                "sector_classified": "Unclassified",
                "business_summary": "",
                "is_potentially_haram": False,
                "confidence": "low",
                "method": "search",
                "haram_reason": None,
            }
    except Exception as exc:
        log_exception(f"Search lookup failed for {ticker}")
        log_message(f"Search error for {ticker}: {exc}")
        return {
            "ticker": ticker,
            "sector_classified": "Unclassified",
            "business_summary": "",
            "is_potentially_haram": False,
            "confidence": "low",
            "method": "search",
            "haram_reason": None,
        }

    snippets: list[str] = []
    for item in search_results[:3]:
        title, url, snippet = _extract_search_fields(item)
        if snippet:
            snippets.append(snippet)
        elif url:
            snippets.append(_fetch_meta_description(url))
        if title and title not in snippets:
            snippets.append(title)

    business_summary = _collapse_text(snippets)
    matched_sector, is_haram = _match_keywords_in_text(business_summary.lower())

    confidence = "high" if matched_sector else "low"
    return {
        "ticker": ticker,
        "sector_classified": matched_sector or "Unclassified",
        "business_summary": business_summary,
        "is_potentially_haram": is_haram,
        "confidence": confidence,
        "method": "search",
        "haram_reason": f"Search snippets suggest {matched_sector} business." if is_haram else None,
    }


def classify_sector_with_ai(ticker: str, company_name: str, business_summary: str) -> dict:
    """Ask Anthropic Claude to classify the company's business."""

    rule_based = classify_sector_rule_based(ticker, company_name, business_summary)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_based

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            max_tokens=400,
            temperature=0,
            system=(
                "You are an Islamic finance expert specializing in AAOIFI Shariah screening. "
                "Classify companies for a Pakistani Muslim investor tool. Be precise and conservative."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"""
Company: {company_name}
PSX Ticker: {ticker}
Business Description: {business_summary}

Task: Classify this company for Shariah screening purposes.

Respond in JSON only:
{{
  "sector": "<sector name>",
  "primary_business": "<one sentence description>",
  "is_haram_business": true/false,
  "haram_reason": "<reason if haram, else null>",
  "confidence": "high/medium/low",
  "mixed_business": true/false,
  "mixed_business_note": "<note if company has both halal and haram activities>"
}}

Haram business categories: conventional banking, conventional insurance,
alcohol production/distribution, tobacco, gambling, pornography, weapons manufacturing.
Note: Islamic banking, takaful insurance, and halal food are NOT haram.
Pakistani companies context: many conglomerates have mixed activities.
""",
                }
            ],
        )

        payload = _extract_json_object(_extract_text_response(response))
        if not payload:
            return rule_based

        sector = str(payload.get("sector") or rule_based.get("sector_classified") or "Unclassified")
        is_haram = bool(payload.get("is_haram_business", rule_based.get("is_potentially_haram", False)))
        haram_reason = payload.get("haram_reason") if is_haram else None
        confidence = str(payload.get("confidence") or rule_based.get("confidence") or "low")
        mixed_business = bool(payload.get("mixed_business", False))
        mixed_note = payload.get("mixed_business_note") or ""

        return {
            "ticker": ticker,
            "sector_classified": sector,
            "business_summary": str(payload.get("primary_business") or business_summary),
            "is_potentially_haram": is_haram,
            "confidence": confidence,
            "method": "ai",
            "haram_reason": haram_reason,
            "mixed_business": mixed_business,
            "mixed_business_note": mixed_note,
        }
    except Exception as exc:
        log_exception(f"AI classification failed for {ticker}")
        log_message(f"AI classification error for {ticker}: {exc}")
        return rule_based


def classify_all_companies(df: pd.DataFrame, log_callback=None) -> pd.DataFrame:
    """Classify all companies in a universe DataFrame."""

    frame = df.copy()
    if frame.empty:
        return frame

    results: list[dict] = []
    for _, row in frame.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        company_name = str(row.get("company_name", ticker)).strip()
        sector_raw = str(row.get("sector_raw", row.get("sector", ""))).strip()

        rule_based = classify_sector_rule_based(ticker, company_name, sector_raw)
        best = rule_based

        if log_callback and ticker:
            log_callback(f"{ticker}: classifying sector...")

        if rule_based.get("confidence") == "low":
            search_result = classify_sector_with_search(ticker, company_name)
            best = _merge_classification(rule_based, search_result)
            if best.get("confidence") == "low" and best.get("business_summary"):
                ai_result = classify_sector_with_ai(ticker, company_name, best.get("business_summary", ""))
                best = _merge_classification(best, ai_result)

        results.append(best)

    result_frame = pd.DataFrame(results)
    frame = frame.merge(result_frame, on="ticker", how="left", suffixes=("", "_classified"))

    frame["sector_classified"] = frame["sector_classified"].fillna(frame.get("sector_raw")).fillna(frame.get("sector"))
    frame["is_haram_business"] = frame["is_potentially_haram"].fillna(False)
    frame["haram_reason"] = frame["haram_reason"].fillna("")
    frame["classification_confidence"] = frame["confidence"].fillna("low")

    return frame


def research_unknown_stock(ticker: str) -> dict:
    """Research a PSX ticker that may not exist in the local database yet."""

    normalized_ticker = str(ticker).strip().upper().replace(".KA", "")
    yahoo_ticker = f"{normalized_ticker}.KA"

    company_name = normalized_ticker
    yfinance_sector = ""
    yfinance_summary = ""
    snippets: list[str] = []

    yfinance_context = _call_with_timeout(
        _fetch_yfinance_research_context,
        timeout_seconds=15,
        fallback={"company_name": company_name, "sector": "", "summary": ""},
        yahoo_ticker=yahoo_ticker,
    )
    company_name = str(yfinance_context.get("company_name") or company_name)
    yfinance_sector = str(yfinance_context.get("sector") or "")
    yfinance_summary = str(yfinance_context.get("summary") or "")

    if not yfinance_summary and not yfinance_sector:
        search_results = _call_with_timeout(
            _search_research_snippets,
            timeout_seconds=15,
            fallback=[],
            normalized_ticker=normalized_ticker,
        )

        for item in search_results[:3]:
            title, url, snippet = _extract_search_fields(item)
            if snippet:
                snippets.append(snippet)
            elif url:
                snippets.append(_fetch_meta_description(url))
            if title:
                snippets.append(title)

    business_context_parts = [part for part in [company_name, yfinance_sector, yfinance_summary, _collapse_text(snippets)] if part]
    business_context = "\n".join(dict.fromkeys(business_context_parts))

    prompt = f"""Given this information about a Pakistani company listed on PSX:
Ticker: {normalized_ticker}
Available Info: {business_context}

As an AAOIFI-certified Shariah screening expert:
1. What sector is this company in?
2. Is its primary business halal, haram, or mixed?
3. What specific AAOIFI concerns exist, if any?
4. What additional information would a Shariah board need to issue a ruling?

Respond in JSON format with keys:
sector, halal_status (halal/haram/mixed/unknown),
concerns, confidence, scholar_note"""

    ai_result = _call_claude_research(normalized_ticker, company_name, business_context, prompt)
    if ai_result:
        ai_result.setdefault("ticker", normalized_ticker)
        ai_result.setdefault("company_name", company_name)
        ai_result.setdefault("business_context", business_context)
        ai_result.setdefault("research_method", "ai")
        ai_result.setdefault("disclaimer", "AI analysis only, not a fatwa.")
        return ai_result

    fallback_classification = classify_sector_rule_based(normalized_ticker, company_name, yfinance_sector or business_context)
    halal_status = "unknown"
    if fallback_classification.get("is_potentially_haram"):
        halal_status = "haram"
    elif fallback_classification.get("sector_classified") and fallback_classification.get("sector_classified") != "Unclassified":
        halal_status = "mixed" if fallback_classification.get("confidence") == "low" else "halal"

    return {
        "ticker": normalized_ticker,
        "company_name": company_name,
        "sector": fallback_classification.get("sector_classified", "Unclassified"),
        "halal_status": halal_status,
        "concerns": fallback_classification.get("haram_reason") or "AI lookup failed; review manually.",
        "confidence": fallback_classification.get("confidence", "low"),
        "scholar_note": "AI analysis unavailable. This is a preliminary research summary, not a fatwa.",
        "business_context": business_context,
        "research_method": fallback_classification.get("method", "rule_based"),
        "disclaimer": "AI analysis only, not a fatwa.",
    }


def _merge_classification(base: dict, update: dict) -> dict:
    """Merge two classification dictionaries while keeping the stronger signal."""

    merged = dict(base)
    merged.update({key: value for key, value in update.items() if value not in (None, "")})
    if merged.get("confidence") == "low" and update.get("confidence") == "high":
        merged["confidence"] = "high"
    if merged.get("is_potentially_haram") is False and update.get("is_potentially_haram") is True:
        merged["is_potentially_haram"] = True
    return merged


def _match_keywords_in_text(text: str) -> tuple[str, bool]:
    """Return a likely sector and whether the text implies a haram business."""

    if not text:
        return "", False

    for exception, label in _HALAL_EXCEPTIONS.items():
        if exception in text:
            return label, False

    for sector, keywords in HARAM_SECTORS_KEYWORDS.items():
        if _contains_any(text, keywords):
            return sector, True

    for sector in HALAL_SECTORS:
        if sector.lower() in text:
            return sector, False

    return "", False


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _combined_text(company_name: str, sector_raw: str) -> str:
    return f"{company_name} {sector_raw}".strip()


def _normalize_sector_label(sector_raw: str) -> str:
    value = str(sector_raw).strip()
    if not value:
        return ""
    return " ".join(part.capitalize() if part.isalpha() else part for part in value.split())


def _infer_halal_sector(text: str) -> str:
    for sector in HALAL_SECTORS:
        if sector.lower() in text:
            return sector
    return ""


def _extract_search_fields(item: object) -> tuple[str, str, str]:
    title = getattr(item, "title", "") if not isinstance(item, dict) else item.get("title", "")
    url = getattr(item, "url", "") if not isinstance(item, dict) else item.get("url", "")
    snippet = (
        getattr(item, "description", "")
        if not isinstance(item, dict)
        else item.get("description", item.get("snippet", ""))
    )
    return str(title or ""), str(url or ""), str(snippet or "")


def _fetch_meta_description(url: str) -> str:
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return str(meta.get("content"))
        title = soup.title.get_text(strip=True) if soup.title else ""
        return title
    except Exception:
        return ""


def _collapse_text(parts: Iterable[str]) -> str:
    unique_parts: list[str] = []
    for part in parts:
        text = str(part).strip()
        if text and text not in unique_parts:
            unique_parts.append(text)
    return " | ".join(unique_parts)


def _extract_text_response(response: object) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            if text:
                texts.append(str(text))
        return "\n".join(texts)
    if isinstance(content, str):
        return content
    return str(response)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _call_claude_research(ticker: str, company_name: str, business_context: str, prompt: str) -> dict:
    """Call Claude with the research prompt and parse a JSON-only response."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {}

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            max_tokens=500,
            temperature=0,
            system="You are an AAOIFI-certified Shariah screening expert.",
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _extract_json_object(_extract_text_response(response))
        if not payload:
            return {}

        concerns = payload.get("concerns")
        if isinstance(concerns, list):
            concerns = "; ".join(str(item) for item in concerns if str(item).strip())

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": str(payload.get("sector") or "Unclassified"),
            "halal_status": str(payload.get("halal_status") or "unknown").lower(),
            "concerns": concerns or "",
            "confidence": str(payload.get("confidence") or "low").lower(),
            "scholar_note": str(payload.get("scholar_note") or ""),
            "business_context": business_context,
            "research_method": "ai",
            "disclaimer": "AI analysis only, not a fatwa.",
        }
    except Exception as exc:
        log_exception(f"Claude research failed for {ticker}")
        log_message(f"Claude research error for {ticker}: {exc}")
        return {}


def _fetch_yfinance_research_context(yahoo_ticker: str) -> dict:
    """Load the minimal yfinance context for research with graceful fallback."""

    try:
        company = yf.Ticker(yahoo_ticker)
        info = _safe_dict(_safe_getattr(company, "info"))
        return {
            "company_name": str(info.get("longName") or info.get("shortName") or yahoo_ticker.replace(".KA", "")),
            "sector": str(info.get("sector") or info.get("industry") or ""),
            "summary": str(info.get("longBusinessSummary") or ""),
        }
    except Exception as exc:
        log_exception(f"yfinance research lookup failed for {yahoo_ticker}")
        log_message(f"yfinance research error for {yahoo_ticker}: {exc}")
        return {"company_name": yahoo_ticker.replace(".KA", ""), "sector": "", "summary": ""}


def _search_research_snippets(normalized_ticker: str) -> list[tuple[str, str, str]]:
    """Collect top PSX search snippets for the research flow."""

    query = f"PSX {normalized_ticker} Pakistan company business"
    try:
        search_results = list(google_search(query, num_results=3, lang="en", advanced=True))
    except TypeError:
        try:
            search_results = list(google_search(query, num_results=3, lang="en"))
        except Exception as exc:
            log_exception(f"research search lookup failed for {normalized_ticker}")
            log_message(f"research search error for {normalized_ticker}: {exc}")
            return []
    except Exception as exc:
        log_exception(f"research search lookup failed for {normalized_ticker}")
        log_message(f"research search error for {normalized_ticker}: {exc}")
        return []

    snippets: list[tuple[str, str, str]] = []
    for item in search_results[:3]:
        snippets.append(_extract_search_fields(item))
    return snippets


def _call_with_timeout(function, timeout_seconds: int, fallback, *args, **kwargs):
    """Run a helper with a hard timeout and return a fallback on delay or error."""

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(function, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        return fallback
    except Exception:
        return fallback
    finally:
        executor.shutdown(wait=False, cancel_futures=True)