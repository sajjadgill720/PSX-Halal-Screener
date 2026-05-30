"""Zakat calculation utilities for PSX stock portfolios.

The module provides two stock-level zakat methods and a portfolio aggregation
helper. It is designed for equities held in a conventional brokerage account,
where the relevant zakatable base is usually limited to cash and receivables
attributable to the shareholding.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

LUNAR_YEAR_RATE = 0.025775
MARKET_VALUE_RATE = 0.025
SILVER_FALLBACK_PKR_PER_GRAM = 9500.0
TROY_OUNCE_TO_GRAMS = 31.1034768


def calculate_zakat_per_stock(ticker: str, shares_held: float, method: str = "assets") -> dict:
    """Calculate zakat due for a single stock position.

    Parameters
    ----------
    ticker:
        PSX ticker symbol. ``.KA`` is added automatically when missing.
    shares_held:
        Number of shares owned by the investor.
    method:
        ``"assets"`` for the AAOIFI-preferred zakatable assets method or
        ``"market"`` for the conservative market value method.

    Returns
    -------
    dict
        A dictionary containing ``ticker``, ``shares_held``, ``market_value``,
        ``zakatable_base``, ``zakat_due``, and ``method_used``.

    Notes
    -----
    Method 1 follows the scholarly preference of zakating only the cash and
    receivables attributable to the shareholding. Method 2 is conservative and
    applies zakat to the entire market value of the position.
    """

    normalized_ticker = _normalize_ticker(ticker)
    shares = _to_float(shares_held)
    snapshot = _fetch_stock_snapshot(normalized_ticker)

    market_price = _to_float(snapshot.get("current_price"))
    market_value = market_price * shares if np.isfinite(market_price) and np.isfinite(shares) else np.nan

    cash_per_share = _to_float(snapshot.get("cash_per_share"))
    receivables_per_share = _to_float(snapshot.get("receivables_per_share"))
    zakatable_per_share = cash_per_share + receivables_per_share
    zakatable_base = zakatable_per_share * shares if np.isfinite(zakatable_per_share) and np.isfinite(shares) else np.nan

    method_used = _normalize_method(method)
    if method_used == "market":
        # Conservative method: scholars sometimes permit zakat on the full market value
        # of tradable shares when the investor wants to err on the side of caution.
        zakatable_base = market_value
        zakat_due = market_value * MARKET_VALUE_RATE if np.isfinite(market_value) else np.nan
    else:
        # AAOIFI-preferred method: zakat is based only on zakatable assets such as
        # cash and receivables attributable to the ownership position.
        zakat_due = zakatable_base * LUNAR_YEAR_RATE if np.isfinite(zakatable_base) else np.nan
        method_used = "assets"

    return {
        "ticker": snapshot.get("ticker", normalized_ticker),
        "shares_held": shares,
        "market_value": market_value,
        "zakatable_base": zakatable_base,
        "zakat_due": zakat_due,
        "method_used": method_used,
    }


def calculate_portfolio_zakat(holdings: dict, method: str = "assets") -> pd.DataFrame:
    """Calculate zakat across a portfolio of stock holdings.

    Parameters
    ----------
    holdings:
        Mapping of ``{ticker: shares_held}``.
    method:
        ``"assets"`` for zakatable assets or ``"market"`` for market value.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per stock and a final ``TOTAL`` row.
    """

    rows = [calculate_zakat_per_stock(ticker, shares, method=method) for ticker, shares in holdings.items()]
    frame = pd.DataFrame(rows)

    if frame.empty:
        return pd.DataFrame(
            columns=["ticker", "shares_held", "market_value", "zakatable_base", "zakat_due", "method_used"]
        )

    total_row = {
        "ticker": "TOTAL",
        "shares_held": pd.to_numeric(frame["shares_held"], errors="coerce").sum(min_count=1),
        "market_value": pd.to_numeric(frame["market_value"], errors="coerce").sum(min_count=1),
        "zakatable_base": pd.to_numeric(frame["zakatable_base"], errors="coerce").sum(min_count=1),
        "zakat_due": pd.to_numeric(frame["zakat_due"], errors="coerce").sum(min_count=1),
        "method_used": _normalize_method(method),
    }

    return pd.concat([frame, pd.DataFrame([total_row])], ignore_index=True)


@lru_cache(maxsize=1)
def get_nisab_pkr() -> float:
    """Return the nisab threshold in PKR using silver as the benchmark.

    The silver benchmark is widely used in contemporary scholarship because it
    is more accessible, which makes zakat due earlier and therefore more
    protective of the poor. The preferred target is 595 grams of silver.
    """

    silver_price_per_gram = _fetch_silver_price_pkr_per_gram()
    if not np.isfinite(silver_price_per_gram) or silver_price_per_gram <= 0:
        silver_price_per_gram = SILVER_FALLBACK_PKR_PER_GRAM

    return 595.0 * silver_price_per_gram


def is_zakat_due(total_portfolio_value: float) -> bool:
    """Return whether a portfolio value reaches the nisab threshold."""

    portfolio_value = _to_float(total_portfolio_value)
    if not np.isfinite(portfolio_value):
        return False

    return portfolio_value >= get_nisab_pkr()


def _fetch_stock_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch the minimal stock snapshot required for zakat calculations."""

    try:
        company = yf.Ticker(ticker)
    except Exception:
        company = None

    if company is None:
        return {
            "ticker": ticker,
            "current_price": np.nan,
            "cash_per_share": np.nan,
            "receivables_per_share": np.nan,
        }

    info = _safe_dict(getattr(company, "info", None))
    fast_info = _safe_dict(getattr(company, "fast_info", None))
    balance_sheet = _safe_frame(getattr(company, "balance_sheet", None))

    shares_outstanding = _first_numeric(
        info.get("sharesOutstanding"),
        fast_info.get("shares"),
        fast_info.get("shares_outstanding"),
    )

    current_price = _first_numeric(
        fast_info.get("last_price"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )

    total_cash = _first_numeric(
        info.get("totalCash"),
        info.get("cashAndCashEquivalents"),
        _statement_value(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments", "Cash"]),
    )

    accounts_receivable = _first_numeric(
        _statement_value(balance_sheet, ["Accounts Receivable", "Net Receivables", "Trade Receivables"]),
    )

    cash_per_share = total_cash / shares_outstanding if np.isfinite(total_cash) and np.isfinite(shares_outstanding) and shares_outstanding > 0 else np.nan
    receivables_per_share = accounts_receivable / shares_outstanding if np.isfinite(accounts_receivable) and np.isfinite(shares_outstanding) and shares_outstanding > 0 else np.nan

    return {
        "ticker": ticker,
        "current_price": current_price,
        "cash_per_share": cash_per_share,
        "receivables_per_share": receivables_per_share,
    }


def _fetch_silver_price_pkr_per_gram() -> float:
    """Fetch the silver price in PKR per gram using Yahoo Finance.

    The function tries a direct PKR quote first and then falls back to USD
    silver plus USD/PKR conversion. If all network lookups fail, the caller
    should use the hardcoded fallback.
    """

    direct = _yahoo_last_price("XAGPKR=X")
    if np.isfinite(direct) and direct > 0:
        return direct / TROY_OUNCE_TO_GRAMS

    silver_usd_per_ounce = _yahoo_last_price("XAGUSD=X")
    usd_pkr = _yahoo_last_price("USDPKR=X")

    if np.isfinite(silver_usd_per_ounce) and silver_usd_per_ounce > 0 and np.isfinite(usd_pkr) and usd_pkr > 0:
        return (silver_usd_per_ounce * usd_pkr) / TROY_OUNCE_TO_GRAMS

    return np.nan


def _yahoo_last_price(ticker: str) -> float:
    """Return the latest available Yahoo Finance price for a ticker."""

    try:
        symbol = yf.Ticker(ticker)
    except Exception:
        return np.nan

    info = _safe_dict(getattr(symbol, "info", None))
    fast_info = _safe_dict(getattr(symbol, "fast_info", None))

    return _first_numeric(
        fast_info.get("last_price"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )


def _normalize_ticker(ticker: str) -> str:
    """Normalize a PSX ticker to Yahoo Finance's ``.KA`` form."""

    value = str(ticker).strip().upper()
    if not value.endswith(".KA"):
        value = f"{value}.KA"
    return value


def _normalize_method(method: str) -> str:
    """Normalize the method selector to one of the supported keys."""

    value = str(method).strip().lower()
    if value in {"market", "market_value", "market value"}:
        return "market"
    return "assets"


def _safe_dict(value: object) -> dict:
    """Convert a mapping-like object to ``dict`` if possible."""

    if isinstance(value, dict):
        return value

    try:
        return dict(value)  # type: ignore[arg-type]
    except Exception:
        return {}


def _safe_frame(value: object) -> pd.DataFrame:
    """Return a DataFrame or an empty frame when the input is unusable."""

    if isinstance(value, pd.DataFrame):
        return value

    return pd.DataFrame()


def _statement_value(frame: pd.DataFrame, candidates: list[str]) -> float:
    """Pull the first matching statement value from a financial statement."""

    if frame.empty or frame.index.empty:
        return np.nan

    lookup = {_normalize_text(str(index)): index for index in frame.index}
    ordered_columns = _ordered_columns(frame)

    for candidate in candidates:
        index_key = lookup.get(_normalize_text(candidate))
        if index_key is None:
            continue

        row = frame.loc[index_key]
        if isinstance(row, pd.Series):
            for column in ordered_columns:
                if column not in row.index:
                    continue
                value = row[column]
                if pd.notna(value):
                    return _to_float(value)
        elif pd.notna(row):
            return _to_float(row)

    return np.nan


def _ordered_columns(frame: pd.DataFrame) -> list:
    """Return statement columns ordered from newest to oldest."""

    dated: list[tuple[pd.Timestamp, object]] = []
    fallback: list[object] = []

    for column in frame.columns:
        timestamp = pd.to_datetime(column, errors="coerce")
        if pd.isna(timestamp):
            fallback.append(column)
        else:
            dated.append((timestamp, column))

    dated.sort(key=lambda item: item[0], reverse=True)
    ordered = [column for _, column in dated]
    ordered.extend(fallback)
    return ordered


def _first_numeric(*values: object) -> float:
    """Return the first finite numeric value from a sequence of candidates."""

    for value in values:
        numeric = _to_float(value)
        if np.isfinite(numeric):
            return numeric
    return np.nan


def _to_float(value: object) -> float:
    """Convert a value to float while preserving ``NaN`` on failure."""

    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return np.nan


def _normalize_text(value: str) -> str:
    """Normalize text for reliable label matching."""

    return " ".join(value.strip().lower().split())
