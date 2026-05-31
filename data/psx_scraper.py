"""Live PSX company discovery and Yahoo Finance data fetchers."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):  # type: ignore[override]
        return iterable

from .error_utils import log_exception, log_message


SYMBOLS_URL = "https://dps.psx.com.pk/symbols"
MARKET_WATCH_URL = "https://dps.psx.com.pk/market-watch"
CACHE_TTL_SECONDS = 24 * 60 * 60
TICKER_CACHE_PATH = Path(__file__).resolve().parent / "psx_tickers_cache.csv"
FINANCIALS_CACHE_PATH = Path(__file__).resolve().parent / "financials_cache.csv"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
_RATE_LIMIT_LOCK = threading.Lock()
_LAST_YFINANCE_CALL = 0.0

_SYMBOL_PATTERN = re.compile(
    r'\{[^{}]*"symbol"\s*:\s*"(?P<ticker>(?:\\.|[^"\\])*)"[^{}]*"name"\s*:\s*"(?P<name>(?:\\.|[^"\\])*)"[^{}]*"sectorName"\s*:\s*"(?P<sector>(?:\\.|[^"\\])*)"[^{}]*\}',
    re.DOTALL,
)

_NON_EQUITY_NAME_RE = re.compile(r"\b(?:sukuk|tfc|treasury|bond|etf|fund|right|warrant)\b", re.IGNORECASE)
_NON_EQUITY_SECTORS = {
    "BILLS AND BONDS",
    "EXCHANGE TRADED FUNDS",
    "CLOSE - END MUTUAL FUND",
    "MONEY MARKET FUND",
}


def get_all_psx_tickers(log_callback=None) -> pd.DataFrame:
    """Scrape all PSX tickers dynamically and cache the result."""

    if log_callback:
        log_callback("Connecting to dps.psx.com.pk...")

    cached = _load_ticker_cache()
    if not cached.empty:
        if log_callback:
            log_callback(f"Loaded {len(cached)} cached PSX symbols")
        print(f"Found {len(cached)} companies on PSX")
        return cached

    frames: list[pd.DataFrame] = []

    for url in (SYMBOLS_URL, MARKET_WATCH_URL):
        try:
            frame = _scrape_psx_page(url)
        except Exception as exc:
            log_exception(f"Failed to scrape PSX page: {url}")
            log_message(f"PSX scrape error for {url}: {exc}")
            frame = pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

        if not frame.empty:
            frames.append(frame)
            if log_callback:
                log_callback(f"Found {len(frame)} companies from {url}")
            if url == SYMBOLS_URL:
                break

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result = _clean_ticker_frame(result)
        _save_ticker_cache(result)
        if log_callback:
            log_callback(f"Found {len(result)} listed companies")
        print(f"Found {len(result)} companies on PSX")
        return result

    if not cached.empty:
        print(f"Found {len(cached)} companies on PSX")
        return cached

    empty = pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])
    print("Found 0 companies on PSX")
    return empty


def get_financials_yfinance(ticker: str) -> dict:
    """Fetch the latest available Yahoo Finance fundamentals for a PSX ticker."""

    base_ticker = str(ticker).strip().upper().replace(".KA", "")
    yahoo_ticker = f"{base_ticker}.KA"
    snapshot: dict[str, object] = {
        "ticker": base_ticker,
        "yahoo_ticker": yahoo_ticker,
        "total_debt": np.nan,
        "total_assets": np.nan,
        "accounts_receivable": np.nan,
        "non_compliant_investments": np.nan,
        "cash": np.nan,
        "total_revenue": np.nan,
        "interest_income": np.nan,
        "net_income": np.nan,
        "market_cap": np.nan,
        "current_price": np.nan,
        "shares_outstanding": np.nan,
        "long_business_summary": np.nan,
    }

    _respect_rate_limit()

    try:
        company = yf.Ticker(yahoo_ticker)
    except Exception as exc:
        log_exception(f"Failed to create yfinance ticker for {yahoo_ticker}")
        log_message(f"yfinance init error for {yahoo_ticker}: {exc}")
        return snapshot

    try:
        info = _safe_dict(_safe_getattr(company, "info"))
        fast_info = _safe_dict(_safe_getattr(company, "fast_info"))
        balance_sheet = _safe_frame(_safe_getattr(company, "balance_sheet"))
        financials = _safe_frame(_safe_getattr(company, "financials"))

        total_debt = info.get("totalDebt")
        if pd.isna(total_debt):
            long_term_debt = _statement_value(balance_sheet, [
                "Long Term Debt",
                "Long Term Debt And Capital Lease Obligation",
                "Long Term Debt Noncurrent",
            ])
            short_term_debt = _statement_value(balance_sheet, [
                "Short Long Term Debt",
                "Current Debt",
                "Current Portion Of Long Term Debt",
                "Short Term Debt",
            ])
            total_debt = _coalesce_numeric(long_term_debt, short_term_debt)

        total_assets = _coalesce_numeric(
            info.get("totalAssets"),
            _statement_value(balance_sheet, ["Total Assets"]),
        )

        accounts_receivable = _coalesce_numeric(
            _statement_value(balance_sheet, [
                "Net Receivables",
                "Accounts Receivable",
                "Trade Receivables",
            ]),
        )

        non_compliant_investments = _coalesce_numeric(
            info.get("shortLongTermDebt"),
            _statement_value(balance_sheet, ["Short Long Term Debt", "Short Term Debt"]),
            0.0,
        )

        cash = _coalesce_numeric(
            info.get("cashAndCashEquivalents"),
            info.get("totalCash"),
            _statement_value(balance_sheet, [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
                "Cash",
            ]),
        )

        total_revenue = _coalesce_numeric(
            info.get("totalRevenue"),
            _statement_value(financials, ["Total Revenue", "Operating Revenue", "Revenue"]),
        )

        interest_income = _coalesce_numeric(
            _statement_value(financials, [
                "Interest Income",
                "Total Interest Income",
                "Interest And Other Income",
                "Interest Income Expense",
            ]),
            0.0,
        )

        net_income = _coalesce_numeric(
            info.get("netIncomeToCommon"),
            info.get("netIncome"),
            _statement_value(financials, ["Net Income", "Net Income Common Stockholders"]),
        )

        market_cap = _coalesce_numeric(
            info.get("marketCap"),
            fast_info.get("market_cap"),
        )

        current_price = _coalesce_numeric(
            info.get("currentPrice"),
            info.get("regularMarketPrice"),
            fast_info.get("last_price"),
        )

        shares_outstanding = _coalesce_numeric(
            info.get("sharesOutstanding"),
            fast_info.get("shares_outstanding"),
        )

        long_business_summary = info.get("longBusinessSummary") or info.get("longName") or info.get("shortName")

        snapshot.update(
            {
                "total_debt": _nan_if_missing(total_debt),
                "total_assets": _nan_if_missing(total_assets),
                "accounts_receivable": _nan_if_missing(accounts_receivable),
                "non_compliant_investments": _nan_if_missing(non_compliant_investments),
                "cash": _nan_if_missing(cash),
                "total_revenue": _nan_if_missing(total_revenue),
                "interest_income": _nan_if_missing(interest_income),
                "net_income": _nan_if_missing(net_income),
                "market_cap": _nan_if_missing(market_cap),
                "current_price": _nan_if_missing(current_price),
                "shares_outstanding": _nan_if_missing(shares_outstanding),
                "long_business_summary": long_business_summary,
            }
        )
    except Exception as exc:
        log_exception(f"Failed to fetch financials for {yahoo_ticker}")
        log_message(f"yfinance fetch error for {yahoo_ticker}: {exc}")
    finally:
        time.sleep(1)

    return snapshot


def get_financials_batch(tickers: list, max_workers: int = 5, log_callback=None) -> pd.DataFrame:
    """Fetch many tickers concurrently and persist the result cache."""

    ticker_list = _unique_tickers(tickers)
    if not ticker_list:
        return pd.DataFrame(columns=[
            "ticker",
            "yahoo_ticker",
            "total_debt",
            "total_assets",
            "accounts_receivable",
            "non_compliant_investments",
            "cash",
            "total_revenue",
            "interest_income",
            "net_income",
            "market_cap",
            "current_price",
            "shares_outstanding",
            "long_business_summary",
        ])

    cached = _load_financials_cache()
    if not cached.empty:
        cached = cached.drop_duplicates(subset=["ticker"], keep="last")
        cached_tickers = set(cached["ticker"].astype(str).str.upper())
        if set(ticker_list).issubset(cached_tickers):
            merged_cached = pd.DataFrame({"ticker": ticker_list}).merge(cached, on="ticker", how="left")
            if log_callback:
                log_callback(f"Using cached financials for {len(merged_cached)} tickers")
            return merged_cached

    results: list[dict] = []
    max_workers = max(1, min(max_workers, len(ticker_list)))

    if log_callback:
        log_callback(f"Fetching financials for {len(ticker_list)} tickers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_financials_yfinance, ticker): ticker for ticker in ticker_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching PSX financials", unit="ticker"):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                log_exception(f"Financial batch fetch failed for {ticker}")
                log_message(f"Financial batch error for {ticker}: {exc}")
                results.append(get_financials_yfinance(ticker))
            if log_callback:
                log_callback(f"{ticker}: balance sheet loaded")

    frame = pd.DataFrame(results)
    if frame.empty:
        return frame

    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.drop_duplicates(subset=["ticker"], keep="last")
    frame = pd.DataFrame({"ticker": ticker_list}).merge(frame, on="ticker", how="left")
    frame["fetched_at"] = datetime.now(timezone.utc).isoformat()

    _save_financials_cache(frame)
    return frame


def _scrape_psx_page(url: str) -> pd.DataFrame:
    """Scrape a PSX endpoint and return symbol/company/sector rows."""

    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    if url.rstrip("/").endswith("symbols"):
        frame = _extract_symbols_from_sources(response.text, soup)
        if not frame.empty:
            return frame

    return _extract_tickers_from_market_watch(response.text, soup)


def _extract_symbols_from_sources(html: str, soup: BeautifulSoup) -> pd.DataFrame:
    """Extract PSX company rows from embedded JSON-like page data."""

    sources: list[str] = [html]
    sources.extend(script.get_text(" ", strip=True) for script in soup.find_all("script"))
    sources.append(soup.get_text(" ", strip=True))

    rows: list[dict[str, str]] = []
    for source in sources:
        rows.extend(_extract_records(source))

    if not rows:
        return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

    frame = pd.DataFrame(rows)
    return _clean_ticker_frame(frame)


def _extract_records(text: str) -> list[dict[str, str]]:
    """Extract JSON-style symbol records from text."""

    records: list[dict[str, str]] = []
    for match in _SYMBOL_PATTERN.finditer(text):
        ticker = _json_unescape(match.group("ticker"))
        company_name = _json_unescape(match.group("name"))
        sector_raw = _json_unescape(match.group("sector"))
        records.append({"ticker": ticker, "company_name": company_name, "sector_raw": sector_raw})
    return records


def _extract_tickers_from_market_watch(html: str, soup: BeautifulSoup) -> pd.DataFrame:
    """Fallback scraper for the market-watch page."""

    rows: list[dict[str, str]] = []
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        tables = []

    for table in tables:
        if table.empty:
            continue
        columns = [str(column).strip().lower() for column in table.columns]
        if not any("symbol" in column for column in columns):
            continue
        symbol_column = next((column for column in table.columns if "symbol" in str(column).lower()), table.columns[0])
        for value in table[symbol_column].dropna().astype(str):
            ticker = value.strip().split()[0].upper()
            if ticker:
                rows.append({"ticker": ticker, "company_name": ticker, "sector_raw": ""})

    if not rows:
        for cell in soup.find_all("td"):
            text = cell.get_text(" ", strip=True)
            if not text:
                continue
            ticker = text.split()[0].upper()
            if 1 <= len(ticker) <= 12 and ticker.replace(".", "").isalnum():
                rows.append({"ticker": ticker, "company_name": ticker, "sector_raw": ""})

    if not rows:
        return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

    frame = pd.DataFrame(rows)
    return _clean_ticker_frame(frame)


def _clean_ticker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize, de-duplicate, and filter the ticker frame."""

    frame = frame.copy()
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["company_name"] = frame["company_name"].astype(str).str.strip()
    frame["sector_raw"] = frame["sector_raw"].astype(str).str.strip()

    frame = frame[frame["ticker"].ne("")]
    frame = frame[~frame["sector_raw"].str.upper().isin(_NON_EQUITY_SECTORS)]
    frame = frame[~frame["company_name"].str.contains(_NON_EQUITY_NAME_RE, na=False)]
    frame = frame.drop_duplicates(subset=["ticker"], keep="first")
    frame = frame.sort_values("ticker").reset_index(drop=True)
    return frame[["ticker", "company_name", "sector_raw"]]


def _load_ticker_cache() -> pd.DataFrame:
    """Load cached PSX tickers if the file is still fresh."""

    if not TICKER_CACHE_PATH.exists():
        return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

    try:
        age = time.time() - TICKER_CACHE_PATH.stat().st_mtime
        if age > CACHE_TTL_SECONDS:
            return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])
        frame = pd.read_csv(TICKER_CACHE_PATH, dtype=str).fillna("")
    except Exception as exc:
        log_exception("Failed to load PSX ticker cache")
        log_message(f"Ticker cache load error: {exc}")
        return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

    if not {"ticker", "company_name", "sector_raw"}.issubset(frame.columns):
        return pd.DataFrame(columns=["ticker", "company_name", "sector_raw"])

    return _clean_ticker_frame(frame)


def _load_financials_cache() -> pd.DataFrame:
    """Load cached financials if available and recent."""

    if not FINANCIALS_CACHE_PATH.exists():
        return pd.DataFrame()

    try:
        age = time.time() - FINANCIALS_CACHE_PATH.stat().st_mtime
        if age > CACHE_TTL_SECONDS:
            return pd.DataFrame()
        frame = pd.read_csv(FINANCIALS_CACHE_PATH, dtype={"ticker": str})
    except Exception as exc:
        log_exception("Failed to load financials cache")
        log_message(f"Financials cache load error: {exc}")
        return pd.DataFrame()

    if "ticker" not in frame.columns:
        return pd.DataFrame()

    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    return frame


def _save_ticker_cache(frame: pd.DataFrame) -> None:
    """Persist PSX ticker cache to disk."""

    try:
        TICKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(TICKER_CACHE_PATH, index=False)
    except Exception as exc:
        log_exception("Failed to save PSX ticker cache")
        log_message(f"Ticker cache save error: {exc}")


def _save_financials_cache(frame: pd.DataFrame) -> None:
    """Persist the financial cache to disk."""

    try:
        FINANCIALS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(FINANCIALS_CACHE_PATH, index=False)
    except Exception as exc:
        log_exception("Failed to save financials cache")
        log_message(f"Financials cache save error: {exc}")


def _unique_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize a ticker iterable into unique uppercase symbols."""

    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in tickers:
        value = str(ticker).strip().upper().replace(".KA", "")
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _safe_getattr(obj: object, attribute: str) -> object:
    """Return an attribute value while swallowing lookup errors."""

    if obj is None:
        return None
    try:
        return getattr(obj, attribute)
    except Exception:
        return None


def _safe_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return dict(value)  # type: ignore[arg-type]
    except Exception:
        return {}


def _safe_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _statement_value(frame: pd.DataFrame, candidates: list[str]) -> float:
    """Return the first matching numeric statement value from a financial frame."""

    if frame.empty:
        return np.nan

    lookup = {_normalize_text(str(index)): index for index in frame.index}
    for candidate in candidates:
        key = lookup.get(_normalize_text(candidate))
        if key is None:
            continue
        series = pd.to_numeric(frame.loc[key], errors="coerce")
        if isinstance(series, pd.Series):
            non_missing = series.dropna()
            if not non_missing.empty:
                return float(non_missing.iloc[0])
        elif pd.notna(series):
            return float(series)
    return np.nan


def _coalesce_numeric(*values: object) -> float:
    """Return the first finite numeric value from the provided candidates."""

    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            return numeric
    return np.nan


def _nan_if_missing(value: object) -> float:
    """Normalize a value to float or NaN."""

    try:
        numeric = float(value)
    except Exception:
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _json_unescape(value: str) -> str:
    """Decode a JSON-style escaped string."""

    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _respect_rate_limit() -> None:
    """Keep Yahoo Finance requests at least one second apart."""

    global _LAST_YFINANCE_CALL
    with _RATE_LIMIT_LOCK:
        elapsed = time.monotonic() - _LAST_YFINANCE_CALL
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _LAST_YFINANCE_CALL = time.monotonic()