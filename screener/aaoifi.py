"""AAOIFI Shariah screening engine for PSX stocks.

This module applies the core AAOIFI-style business and financial screens to a
Pandas DataFrame of companies. It is designed to work with raw financial data
retrieved from Yahoo Finance or a similar upstream fetcher.

Expected input columns are:

- ticker
- sector
- total_debt
- total_assets
- interest_income
- total_revenue
- accounts_receivable
- non_compliant_investments

The main entry point, :func:`screen_aaoifi`, returns the original rows enriched
with per-screen PASS/FAIL results, ratios, an overall status, a purification
ratio, and a simple 0-100 Shariah score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUSINESS_HARAM_SECTORS = {
    "banking (conventional)",
    "insurance (conventional)",
    "alcohol",
    "tobacco",
    "weapons",
    "pornography",
    "gambling",
    "banking",
    "insurance",
}

DEBT_THRESHOLD = 0.33
INTEREST_THRESHOLD = 0.05
SECURITIES_THRESHOLD = 0.33
RECEIVABLES_THRESHOLD = 0.49


def screen_aaoifi(data: pd.DataFrame) -> pd.DataFrame:
    """Apply AAOIFI Shariah screens to a company DataFrame.

    Parameters
    ----------
    data:
        Input DataFrame containing company-level financial data. The function
        expects columns for ticker, sector, debt, assets, interest income,
        revenue, receivables, and non-compliant investments, but it will handle
        missing columns gracefully by treating them as ``NaN``.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with the following additional columns:
        ``business_screen``, ``debt_ratio``, ``debt_screen``,
        ``interest_ratio``, ``interest_screen``, ``securities_ratio``,
        ``securities_screen``, ``receivables_ratio``, ``receivables_screen``,
        ``overall_status``, ``purification_ratio``, and ``shariah_score``.

    Notes
    -----
    The overall status is computed as follows:

    - ``HALAL`` when all five screens pass.
    - ``DOUBTFUL`` when the business screen passes and one or two financial
      screens fail.
    - ``HARAM`` when the business screen fails or three or more financial
      screens fail.

    The Shariah score is a simple 0-100 comfort score. Each of the five screens
    contributes up to 20 points, with a linear decrease from full points at
    ratio 0 to zero points at the respective threshold. Business screens are
    scored as 20 for pass and 0 for fail.
    """

    frame = data.copy()

    sector = frame.get("sector", pd.Series(index=frame.index, dtype="object"))
    total_debt = _numeric_series(frame, "total_debt")
    total_assets = _numeric_series(frame, "total_assets")
    interest_income = _numeric_series(frame, "interest_income")
    total_revenue = _numeric_series(frame, "total_revenue")
    accounts_receivable = _numeric_series(frame, "accounts_receivable")
    non_compliant_investments = _numeric_series(frame, "non_compliant_investments")

    business_screen = sector.map(_business_result)

    debt_ratio = _safe_divide(total_debt, total_assets)
    debt_screen = debt_ratio.map(lambda value: _ratio_result(value, DEBT_THRESHOLD))

    interest_ratio = _safe_divide(interest_income, total_revenue)
    interest_screen = interest_ratio.map(lambda value: _ratio_result(value, INTEREST_THRESHOLD))

    securities_ratio = _safe_divide(non_compliant_investments, total_assets)
    securities_screen = securities_ratio.map(lambda value: _ratio_result(value, SECURITIES_THRESHOLD))

    receivables_ratio = _safe_divide(accounts_receivable, total_assets)
    receivables_screen = receivables_ratio.map(lambda value: _ratio_result(value, RECEIVABLES_THRESHOLD))

    financial_fail_count = (
        (debt_screen == "FAIL").astype(int)
        + (interest_screen == "FAIL").astype(int)
        + (securities_screen == "FAIL").astype(int)
        + (receivables_screen == "FAIL").astype(int)
    )

    overall_status = np.where(
        business_screen == "FAIL",
        "HARAM",
        np.where(
            financial_fail_count == 0,
            "HALAL",
            np.where(financial_fail_count <= 2, "DOUBTFUL", "HARAM"),
        ),
    )

    purification_ratio = interest_ratio.where(overall_status == "HALAL", np.nan)

    shariah_score = (
        _screen_score(business_screen, None)
        + _screen_score_from_ratio(debt_ratio, DEBT_THRESHOLD)
        + _screen_score_from_ratio(interest_ratio, INTEREST_THRESHOLD)
        + _screen_score_from_ratio(securities_ratio, SECURITIES_THRESHOLD)
        + _screen_score_from_ratio(receivables_ratio, RECEIVABLES_THRESHOLD)
    ).round().clip(0, 100).astype(int)

    frame["business_screen"] = business_screen
    frame["debt_ratio"] = debt_ratio
    frame["debt_screen"] = debt_screen
    frame["interest_ratio"] = interest_ratio
    frame["interest_screen"] = interest_screen
    frame["securities_ratio"] = securities_ratio
    frame["securities_screen"] = securities_screen
    frame["receivables_ratio"] = receivables_ratio
    frame["receivables_screen"] = receivables_screen
    frame["overall_status"] = overall_status
    frame["purification_ratio"] = purification_ratio
    frame["shariah_score"] = shariah_score

    return frame


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series for ``column`` or ``NaN`` if the column is missing."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    return pd.to_numeric(frame[column], errors="coerce")


def _business_result(value: object) -> str:
    """Return the business screen result for a sector label."""

    if value is None or pd.isna(value):
        return "PASS"

    normalized = _normalize_text(str(value))
    return "FAIL" if normalized in BUSINESS_HARAM_SECTORS else "PASS"


def _ratio_result(value: float | np.floating | np.integer | None, threshold: float) -> str:
    """Convert a ratio into PASS/FAIL using the provided threshold."""

    if value is None or pd.isna(value):
        return "FAIL"

    return "PASS" if float(value) < threshold else "FAIL"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while avoiding invalid division results."""

    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")

    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.where(denominator > 0, np.nan)
    return result


def _screen_score_from_ratio(ratio: pd.Series, threshold: float) -> pd.Series:
    """Score a financial screen from 0 to 20 based on headroom to threshold."""

    ratio = pd.to_numeric(ratio, errors="coerce")
    score = (1 - (ratio / threshold)) * 20
    score = score.clip(lower=0, upper=20)
    score = score.where(ratio.notna(), 0)
    return score


def _screen_score(business_screen: pd.Series, threshold: None) -> pd.Series:
    """Score the business screen as 20 for PASS and 0 for FAIL.

    The ``threshold`` parameter is accepted for signature consistency with the
    ratio-based scoring helper and is unused.
    """

    return business_screen.map(lambda value: 20 if value == "PASS" else 0)


def _normalize_text(value: str) -> str:
    """Normalize text for comparison against prohibited sector labels."""

    return " ".join(value.strip().lower().split())
