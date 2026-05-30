"""PSX Halal Screener data fetcher.

This module fetches a small set of financial fields from Yahoo Finance for
Pakistan Stock Exchange tickers. It also provides a built-in master list of
commonly tracked PSX names and caches fetched rows on disk for 24 hours.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

HARAM_SECTORS = {"Banking", "Insurance", "Alcohol", "Tobacco"}

PSX_TICKERS = {
    "ENGRO": {"sector": "Fertilizer"},
    "LUCK": {"sector": "Cement"},
    "HBL": {"sector": "Banking"},
    "MCB": {"sector": "Banking"},
    "UBL": {"sector": "Banking"},
    "ABL": {"sector": "Banking"},
    "BAHL": {"sector": "Banking"},
    "MEBL": {"sector": "Banking"},
    "NBP": {"sector": "Banking"},
    "BOP": {"sector": "Banking"},
    "FABL": {"sector": "Banking"},
    "PSO": {"sector": "Oil & Gas Marketing"},
    "OGDC": {"sector": "Oil & Gas Exploration"},
    "PPL": {"sector": "Oil & Gas Exploration"},
    "POL": {"sector": "Oil & Gas Exploration"},
    "MARI": {"sector": "Oil & Gas Exploration"},
    "HUBC": {"sector": "Power Generation"},
    "KAPCO": {"sector": "Power Generation"},
    "KEL": {"sector": "Power Generation"},
    "FFC": {"sector": "Fertilizer"},
    "EFERT": {"sector": "Fertilizer"},
    "FFBL": {"sector": "Fertilizer"},
    "FCCL": {"sector": "Cement"},
    "DGKC": {"sector": "Cement"},
    "CHCC": {"sector": "Cement"},
    "MLCF": {"sector": "Cement"},
    "ISL": {"sector": "Steel"},
    "ATRL": {"sector": "Oil & Gas Refining"},
    "NRL": {"sector": "Oil & Gas Refining"},
    "CNERGY": {"sector": "Oil & Gas Refining"},
    "SNGP": {"sector": "Gas Utilities"},
    "SSGC": {"sector": "Gas Utilities"},
    "TRG": {"sector": "Technology"},
    "SYS": {"sector": "Technology"},
    "HCAR": {"sector": "Automobile Assembler"},
    "INDU": {"sector": "Automobile Assembler"},
    "PSMC": {"sector": "Automobile Assembler"},
    "SEARL": {"sector": "Pharmaceuticals"},
    "GLAXO": {"sector": "Pharmaceuticals"},
    "ABOT": {"sector": "Pharmaceuticals"},
    "HINOON": {"sector": "Pharmaceuticals"},
    "COLG": {"sector": "Personal Goods"},
    "NESTLE": {"sector": "Food & Personal Care"},
    "UNITY": {"sector": "Food"},
    "EPCL": {"sector": "Chemicals"},
    "LOTCHEM": {"sector": "Chemicals"},
    "FATIMA": {"sector": "Fertilizer"},
    "PAEL": {"sector": "Electrical Goods"},
    "EFUL": {"sector": "Insurance"},
    "PAKT": {"sector": "Tobacco"},
}

_CACHE_TTL = timedelta(hours=24)
_CACHE_PATH = Path(__file__).with_name("cache.pkl")


def get_psx_data(tickers: list) -> pd.DataFrame:
    """Return a DataFrame with raw financial fields for the requested tickers.

    Tickers can be plain PSX symbols (for example, ``ENGRO``) or Yahoo Finance
    symbols with the ``.KA`` suffix. Unknown tickers are still fetched.
    """

    normalized_tickers = _unique_tickers(tickers)
    if not normalized_tickers:
        normalized_tickers = list(PSX_TICKERS.keys())

    cached_rows = _load_cache()
    rows: list[dict] = []
    row_map: dict[str, dict] = {}
    updated_cache = False

    pending_tickers: list[str] = []

    for ticker in normalized_tickers:
        cached_row = _find_cached_row(cached_rows, ticker)
        if cached_row is not None:
            row_map[ticker] = cached_row
            continue

        pending_tickers.append(ticker)

    if pending_tickers:
        max_workers = min(8, len(pending_tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_fetch_company_row, ticker): ticker for ticker in pending_tickers}
            done, not_done = wait(future_map.keys(), timeout=45)

            for future in done:
                ticker = future_map[future]
                try:
                    row_map[ticker] = future.result()
                except Exception:
                    row_map[ticker] = _empty_company_row(ticker)
                updated_cache = True

            for future in not_done:
                ticker = future_map[future]
                future.cancel()
                row_map[ticker] = _empty_company_row(ticker)
                updated_cache = True

    for ticker in normalized_tickers:
        if ticker in row_map:
            rows.append(row_map[ticker])

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.reindex(
        columns=[
            "ticker",
            "yahoo_ticker",
            "sector",
            "is_haram_sector",
            "total_debt",
            "total_assets",
            "interest_income",
            "total_revenue",
            "accounts_receivable",
            "cash_and_equivalents",
            "market_cap",
            "current_price",
        ]
    )

    if updated_cache:
        _save_cache(result)

    return result


def _fetch_company_row(ticker: str) -> dict:
    base_ticker = _strip_suffix(ticker)
    yahoo_ticker = _ensure_suffix(base_ticker)
    sector = PSX_TICKERS.get(base_ticker, {}).get("sector", "Unknown")

    try:
        company = yf.Ticker(yahoo_ticker)
    except Exception:
        company = None

    info = _safe_dict(_safe_getattr(company, "info"))
    fast_info = _safe_dict(_safe_getattr(company, "fast_info"))
    balance_sheet = _safe_frame(_safe_getattr(company, "balance_sheet"))
    income_stmt = _safe_frame(_safe_getattr(company, "income_stmt"))

    total_debt = _coalesce(
        info.get("totalDebt"),
        _statement_value(
            balance_sheet,
            [
                "Total Debt",
                "Net Debt",
                "Short Long Term Debt",
                "Long Term Debt",
                "Current Debt",
                "Current Portion of Long Term Debt",
            ],
        ),
    )

    total_assets = _coalesce(
        info.get("totalAssets"),
        _statement_value(balance_sheet, ["Total Assets"]),
    )

    interest_income = _coalesce(
        _statement_value(
            income_stmt,
            [
                "Interest Income",
                "Total Interest Income",
                "Interest And Other Income",
                "Interest Income Expense",
            ],
        ),
    )

    total_revenue = _coalesce(
        info.get("totalRevenue"),
        _statement_value(income_stmt, ["Total Revenue", "Operating Revenue", "Revenue"]),
    )

    accounts_receivable = _coalesce(
        _statement_value(
            balance_sheet,
            [
                "Accounts Receivable",
                "Net Receivables",
                "Trade Receivables",
            ],
        ),
    )

    cash_and_equivalents = _coalesce(
        info.get("totalCash"),
        info.get("cashAndCashEquivalents"),
        _statement_value(
            balance_sheet,
            [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
                "Cash",
            ],
        ),
    )

    market_cap = _coalesce(
        fast_info.get("market_cap"),
        info.get("marketCap"),
    )

    current_price = _coalesce(
        fast_info.get("last_price"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )

    return {
        "ticker": base_ticker,
        "yahoo_ticker": yahoo_ticker,
        "sector": sector,
        "is_haram_sector": sector in HARAM_SECTORS,
        "total_debt": _nan_if_missing(total_debt),
        "total_assets": _nan_if_missing(total_assets),
        "interest_income": _nan_if_missing(interest_income),
        "total_revenue": _nan_if_missing(total_revenue),
        "accounts_receivable": _nan_if_missing(accounts_receivable),
        "cash_and_equivalents": _nan_if_missing(cash_and_equivalents),
        "market_cap": _nan_if_missing(market_cap),
        "current_price": _nan_if_missing(current_price),
    }


def _empty_company_row(ticker: str) -> dict:
    """Return a placeholder row when a ticker cannot be fetched in time."""

    base_ticker = _strip_suffix(ticker)
    yahoo_ticker = _ensure_suffix(base_ticker)
    sector = PSX_TICKERS.get(base_ticker, {}).get("sector", "Unknown")

    return {
        "ticker": base_ticker,
        "yahoo_ticker": yahoo_ticker,
        "sector": sector,
        "is_haram_sector": sector in HARAM_SECTORS,
        "total_debt": np.nan,
        "total_assets": np.nan,
        "interest_income": np.nan,
        "total_revenue": np.nan,
        "accounts_receivable": np.nan,
        "cash_and_equivalents": np.nan,
        "market_cap": np.nan,
        "current_price": np.nan,
    }


def _safe_getattr(obj: object, attribute: str) -> object:
    """Return an attribute value while swallowing network-backed lookup errors."""

    if obj is None:
        return None

    try:
        return getattr(obj, attribute)
    except Exception:
        return None


def _load_cache() -> pd.DataFrame:
    if not _CACHE_PATH.exists():
        return pd.DataFrame()

    try:
        with _CACHE_PATH.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return pd.DataFrame()

    if not isinstance(payload, dict):
        return pd.DataFrame()

    cached_at = payload.get("cached_at")
    data = payload.get("data")

    if not isinstance(data, pd.DataFrame) or not isinstance(cached_at, datetime):
        return pd.DataFrame()

    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()].copy()

    if datetime.now(timezone.utc) - cached_at > _CACHE_TTL:
        return pd.DataFrame()

    return data.copy()


def _save_cache(data: pd.DataFrame) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cached_at": datetime.now(timezone.utc), "data": data.copy()}

    try:
        with _CACHE_PATH.open("wb") as handle:
            pickle.dump(payload, handle)
    except Exception:
        pass


def _find_cached_row(cached_rows: pd.DataFrame, ticker: str) -> dict | None:
    if cached_rows.empty or "ticker" not in cached_rows.columns:
        return None

    ticker_frame = cached_rows.loc[:, cached_rows.columns == "ticker"]
    if ticker_frame.empty:
        return None

    ticker_series = ticker_frame.iloc[:, 0]
    if not isinstance(ticker_series, pd.Series):
        return None

    base_ticker = _strip_suffix(ticker)
    match = cached_rows.loc[ticker_series == base_ticker]
    if match.empty:
        return None

    row = match.iloc[0].to_dict()
    row["is_haram_sector"] = bool(row.get("is_haram_sector", False))
    return row


def _unique_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []

    for ticker in tickers:
        if ticker is None:
            continue

        base_ticker = _strip_suffix(str(ticker))
        if not base_ticker or base_ticker in seen:
            continue

        seen.add(base_ticker)
        unique.append(base_ticker)

    return unique


def _strip_suffix(ticker: str) -> str:
    value = ticker.strip().upper()
    if value.endswith(".KA"):
        value = value[:-3]
    return value


def _ensure_suffix(ticker: str) -> str:
    value = _strip_suffix(ticker)
    return f"{value}.KA"


def _safe_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value

    try:
        return dict(value)  # type: ignore[arg-type]
    except Exception:
        return {}


def _safe_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value

    return pd.DataFrame()


def _statement_value(frame: pd.DataFrame, candidates: Iterable[str]) -> float | int | np.floating | np.integer | None:
    if frame.empty or frame.index.empty:
        return np.nan

    lookup = {_normalize_label(str(index)): index for index in frame.index}
    ordered_columns = _ordered_columns(frame)

    for candidate in candidates:
        index_key = lookup.get(_normalize_label(candidate))
        if index_key is None:
            continue

        row = frame.loc[index_key]
        if isinstance(row, pd.Series):
            for column in ordered_columns:
                if column not in row.index:
                    continue

                value = row[column]
                if pd.notna(value):
                    return value
        elif pd.notna(row):
            return row

    return np.nan


def _ordered_columns(frame: pd.DataFrame) -> list:
    columns: list[tuple[pd.Timestamp | pd.NaTType, object]] = []
    fallback: list[object] = []

    for column in frame.columns:
        timestamp = pd.to_datetime(column, errors="coerce")
        if pd.isna(timestamp):
            fallback.append(column)
        else:
            columns.append((timestamp, column))

    columns.sort(key=lambda item: item[0], reverse=True)
    ordered = [column for _, column in columns]
    ordered.extend(fallback)
    return ordered


def _normalize_label(value: str) -> str:
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    return normalized


def _coalesce(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, np.floating) and np.isnan(value):
            continue
        if pd.isna(value):
            continue
        return value
    return np.nan


def _nan_if_missing(value: object) -> float | int | np.floating | np.integer | None:
    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass

    return value
