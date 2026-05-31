"""Live screening pipeline for PSX equities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.error_utils import log_exception, log_message
from data.news_monitor import flag_news_concerns
from data.psx_scraper import get_all_psx_tickers, get_financials_batch, get_financials_yfinance
from data.sector_classifier import classify_all_companies
from screener.aaoifi import screen_aaoifi


SCREENED_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "screened_results.csv"
CACHE_TTL_SECONDS = 24 * 60 * 60


def run_full_screening(use_cache: bool = True, log_callback=None) -> pd.DataFrame:
    """Run the full live screening pipeline for the PSX universe."""

    if use_cache:
        cached = _load_screened_cache()
        if not cached.empty:
            if log_callback:
                log_callback(f"Loaded {len(cached)} screened rows from cache")
            return cached

    try:
        tickers = get_all_psx_tickers(log_callback=log_callback)
        if tickers.empty:
            return _fallback_cached_screened()

        if log_callback:
            log_callback(f"Step 1 complete: {len(tickers)} PSX symbols discovered")

        financials = get_financials_batch(tickers["ticker"].tolist(), log_callback=log_callback)
        enriched = tickers.merge(financials, on="ticker", how="left")
        if log_callback:
            log_callback(f"Step 2 complete: {len(financials)} financial snapshots loaded")

        classified = classify_all_companies(enriched, log_callback=log_callback)
        if log_callback:
            classified_count = int(classified.get("sector_classified", pd.Series(dtype=object)).notna().sum())
            log_callback(f"Step 3 complete: {classified_count} companies classified")

        if "sector_classified" in classified.columns:
            classified["sector"] = classified["sector_classified"].fillna(classified.get("sector_raw")).fillna("Unknown")
        else:
            classified["sector"] = classified.get("sector_raw", "Unknown")

        screened = screen_aaoifi(classified)
        if log_callback:
            log_callback("Step 4 complete: AAOIFI screening finished")

        screened["purification_ratio"] = pd.to_numeric(screened["purification_ratio"], errors="coerce")
        screened["purification_ratio_percent"] = screened["purification_ratio"] * 100
        screened["screened_at"] = datetime.now(timezone.utc).isoformat()
        screened["classification_confidence"] = screened.get("classification_confidence", "low")
        screened["sector_classified"] = screened.get("sector_classified", screened.get("sector", "Unknown"))
        screened["is_haram_business"] = screened.get("is_haram_business", False)

        _save_screened_cache(screened)
        if log_callback:
            log_callback(f"Step 5 complete: results ready ({len(screened)} rows)")
        return screened
    except Exception as exc:
        log_exception("Full screening pipeline failed")
        log_message(f"Full screening error: {exc}")
        return _fallback_cached_screened()


def screen_single_company(ticker: str, log_callback=None) -> dict:
    """Run the full screening flow for one PSX ticker."""

    normalized_ticker = str(ticker).strip().upper().replace(".KA", "")
    if not normalized_ticker:
        return {}

    try:
        universe = get_all_psx_tickers(log_callback=log_callback)
        row = universe[universe["ticker"] == normalized_ticker]
        if not row.empty:
            company_name = str(row.iloc[0].get("company_name", normalized_ticker))
            sector_raw = str(row.iloc[0].get("sector_raw", ""))
        else:
            company_name = normalized_ticker
            sector_raw = ""

        financial_snapshot = _call_with_timeout(
            get_financials_yfinance,
            timeout_seconds=25,
            fallback={"ticker": normalized_ticker},
            ticker=normalized_ticker,
        )
        if log_callback:
            log_callback(f"{normalized_ticker}: financial snapshot loaded")
        financial = pd.DataFrame([financial_snapshot])
        base = pd.DataFrame([
            {
                "ticker": normalized_ticker,
                "company_name": company_name,
                "sector_raw": sector_raw,
            }
        ])
        merged = base.merge(financial, on="ticker", how="left")
        classified = classify_all_companies(merged, log_callback=log_callback)
        if log_callback:
            log_callback(f"{normalized_ticker}: sector classification complete")
        classified["sector"] = classified["sector_classified"].fillna(classified.get("sector_raw")).fillna("Unknown")
        screened = screen_aaoifi(classified)
        if log_callback:
            log_callback(f"{normalized_ticker}: AAOIFI screening complete")
        screened["purification_ratio_percent"] = pd.to_numeric(screened["purification_ratio"], errors="coerce") * 100
        screened["screening_explanation"] = screened.apply(generate_screening_explanation, axis=1)

        concern_data = _call_with_timeout(
            flag_news_concerns,
            timeout_seconds=20,
            fallback={"has_concerns": False, "concern_keywords": [], "news_items": []},
            ticker=normalized_ticker,
            company_name=company_name,
        )
        row_result = screened.iloc[0].to_dict()
        row_result["news_concerns"] = concern_data
        row_result["screening_explanation"] = screened.iloc[0]["screening_explanation"]
        return row_result
    except Exception as exc:
        log_exception(f"Single company screen failed for {normalized_ticker}")
        log_message(f"Single screening error for {normalized_ticker}: {exc}")
        return {
            "ticker": normalized_ticker,
            "screening_explanation": "Unable to complete screening for this ticker.",
        }


def generate_screening_explanation(row: pd.Series) -> str:
    """Create a plain-English explanation for a screened company."""

    ticker = str(row.get("ticker", "")).upper()
    sector = str(row.get("sector_classified") or row.get("sector") or "Unknown")
    status = str(row.get("overall_status", "UNKNOWN")).upper()
    score = row.get("shariah_score", np.nan)
    debt_ratio = _format_percent(row.get("debt_ratio"))
    interest_ratio = _format_percent(row.get("interest_ratio"))
    purification_ratio = _format_percent(row.get("purification_ratio"))

    business_text = str(row.get("business_screen", "PASS")).upper()
    debt_text = str(row.get("debt_screen", "PASS")).upper()
    interest_text = str(row.get("interest_screen", "PASS")).upper()
    securities_text = str(row.get("securities_screen", "PASS")).upper()
    receivables_text = str(row.get("receivables_screen", "PASS")).upper()

    passed = sum(value == "PASS" for value in [business_text, debt_text, interest_text, securities_text, receivables_text])
    sector_note = _friendly_sector(sector)

    if status == "HALAL":
        lead = f"{ticker} passed all 5 Shariah screens."
    elif status == "DOUBTFUL":
        lead = f"{ticker} passed the business screen but failed {5 - passed} financial screen(s), so it is classified as doubtful."
    else:
        lead = f"{ticker} failed the Shariah screening because the business or financial profile is not compliant."

    details = [
        lead,
        f"Its core business appears to be {sector_note}." if sector_note else "Its core business was not clearly classified.",
        f"Debt ratio is {debt_ratio}, interest income ratio is {interest_ratio}, and purification ratio is {purification_ratio}.",
        f"Shariah Score: {int(score) if pd.notna(score) else 0}/100.",
    ]

    if str(row.get("haram_reason", "")).strip():
        details.append(f"Reason: {row.get('haram_reason')}")

    if status == "HALAL" and pd.notna(row.get("purification_ratio")):
        details.append(f"Purification is estimated at {purification_ratio}.")

    return " ".join(details)


def _load_screened_cache() -> pd.DataFrame:
    if not SCREENED_CACHE_PATH.exists():
        return pd.DataFrame()

    try:
        age = datetime.now().timestamp() - SCREENED_CACHE_PATH.stat().st_mtime
        if age > CACHE_TTL_SECONDS:
            return pd.DataFrame()
        return pd.read_csv(SCREENED_CACHE_PATH)
    except Exception as exc:
        log_exception("Failed to load screened cache")
        log_message(f"Screened cache load error: {exc}")
        return pd.DataFrame()


def _save_screened_cache(frame: pd.DataFrame) -> None:
    try:
        SCREENED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(SCREENED_CACHE_PATH, index=False)
    except Exception as exc:
        log_exception("Failed to save screened cache")
        log_message(f"Screened cache save error: {exc}")


def _fallback_cached_screened() -> pd.DataFrame:
    cached = _load_screened_cache()
    if not cached.empty:
        return cached
    return pd.DataFrame()


def _format_percent(value: object) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "N/A"
    if np.isnan(numeric):
        return "N/A"
    return f"{numeric:.1%}"


def _friendly_sector(sector: str) -> str:
    cleaned = str(sector).strip()
    if not cleaned:
        return "an unclassified business"
    return cleaned.lower()


def _call_with_timeout(function, timeout_seconds: int, fallback, *args, **kwargs):
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