"""Persistent SQLite verdict store for PSX Shariah screening.

This module keeps company metadata, screening verdicts, financial snapshots,
and zakat calculations in a local SQLite database so the application can
reuse prior work instead of re-screening the same company unnecessarily.
"""

from __future__ import annotations

import difflib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "verdict_store.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _coerce_number(value: object, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _coerce_text(value: object, default: str = "") -> str:
    if _is_missing(value):
        return default
    return str(value)


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


@contextmanager
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        _initialize_schema(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            sector_raw TEXT,
            sector_classified TEXT,
            listed_date TEXT,
            last_updated TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS verdicts (
            ticker TEXT PRIMARY KEY,
            overall_status TEXT,
            shariah_score INTEGER,
            business_screen TEXT,
            business_reason TEXT,
            debt_screen TEXT,
            debt_ratio REAL,
            debt_reason TEXT,
            interest_screen TEXT,
            interest_ratio REAL,
            interest_reason TEXT,
            securities_screen TEXT,
            securities_ratio REAL,
            securities_reason TEXT,
            receivables_screen TEXT,
            receivables_ratio REAL,
            receivables_reason TEXT,
            purification_ratio REAL,
            ai_explanation TEXT,
            haram_reason TEXT,
            halal_conditions TEXT,
            screened_at TIMESTAMP,
            data_source TEXT,
            data_quality TEXT
        );

        CREATE TABLE IF NOT EXISTS financials (
            ticker TEXT PRIMARY KEY,
            total_assets REAL,
            total_debt REAL,
            interest_income REAL,
            total_revenue REAL,
            accounts_receivable REAL,
            non_compliant_investments REAL,
            cash REAL,
            market_cap REAL,
            current_price REAL,
            shares_outstanding REAL,
            fiscal_year TEXT,
            fetched_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS zakat_data (
            ticker TEXT PRIMARY KEY,
            cash_per_share REAL,
            receivables_per_share REAL,
            inventory_per_share REAL,
            zakatable_per_share REAL,
            calculated_at TIMESTAMP
        );
        """
    )


def save_verdict(ticker: str, verdict: dict) -> None:
    """Persist a screening verdict and related company metadata."""

    ticker_norm = _normalize_ticker(ticker)
    company_name = _coerce_text(verdict.get("company_name") or verdict.get("name") or ticker_norm)
    sector_raw = _coerce_text(verdict.get("sector_raw") or verdict.get("sector"))
    sector_classified = _coerce_text(verdict.get("sector_classified") or verdict.get("sector"))
    listed_date = _coerce_text(verdict.get("listed_date"))
    timestamp = _coerce_text(verdict.get("screened_at")) or _utc_now()

    verdict_row = {
        "ticker": ticker_norm,
        "overall_status": _coerce_text(verdict.get("overall_status")).upper(),
        "shariah_score": _coerce_int(verdict.get("shariah_score")),
        "business_screen": _coerce_text(verdict.get("business_screen")),
        "business_reason": _coerce_text(verdict.get("business_reason") or verdict.get("business_summary") or verdict.get("screening_explanation")),
        "debt_screen": _coerce_text(verdict.get("debt_screen")),
        "debt_ratio": verdict.get("debt_ratio"),
        "debt_reason": _coerce_text(verdict.get("debt_reason") or verdict.get("debt_explanation") or verdict.get("debt_note")),
        "interest_screen": _coerce_text(verdict.get("interest_screen")),
        "interest_ratio": verdict.get("interest_ratio"),
        "interest_reason": _coerce_text(verdict.get("interest_reason") or verdict.get("interest_explanation") or verdict.get("interest_note")),
        "securities_screen": _coerce_text(verdict.get("securities_screen")),
        "securities_ratio": verdict.get("securities_ratio"),
        "securities_reason": _coerce_text(verdict.get("securities_reason") or verdict.get("securities_explanation") or verdict.get("securities_note")),
        "receivables_screen": _coerce_text(verdict.get("receivables_screen")),
        "receivables_ratio": verdict.get("receivables_ratio"),
        "receivables_reason": _coerce_text(verdict.get("receivables_reason") or verdict.get("receivables_explanation") or verdict.get("receivables_note")),
        "purification_ratio": verdict.get("purification_ratio"),
        "ai_explanation": _coerce_text(verdict.get("ai_explanation") or verdict.get("screening_explanation") or verdict.get("explanation")),
        "haram_reason": _coerce_text(verdict.get("haram_reason")),
        "halal_conditions": _coerce_text(verdict.get("halal_conditions")),
        "screened_at": timestamp,
        "data_source": _coerce_text(verdict.get("data_source") or "ai_estimated"),
        "data_quality": _coerce_text(verdict.get("data_quality") or "estimated"),
    }

    company_row = {
        "ticker": ticker_norm,
        "company_name": company_name,
        "sector_raw": sector_raw,
        "sector_classified": sector_classified,
        "listed_date": listed_date,
        "last_updated": timestamp,
    }

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO companies (ticker, company_name, sector_raw, sector_classified, listed_date, last_updated)
            VALUES (:ticker, :company_name, :sector_raw, :sector_classified, :listed_date, :last_updated)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                sector_raw = COALESCE(excluded.sector_raw, companies.sector_raw),
                sector_classified = COALESCE(excluded.sector_classified, companies.sector_classified),
                listed_date = COALESCE(excluded.listed_date, companies.listed_date),
                last_updated = excluded.last_updated
            """,
            company_row,
        )

        connection.execute(
            """
            INSERT INTO verdicts (
                ticker, overall_status, shariah_score, business_screen, business_reason,
                debt_screen, debt_ratio, debt_reason, interest_screen, interest_ratio, interest_reason,
                securities_screen, securities_ratio, securities_reason, receivables_screen, receivables_ratio,
                receivables_reason, purification_ratio, ai_explanation, haram_reason, halal_conditions,
                screened_at, data_source, data_quality
            ) VALUES (
                :ticker, :overall_status, :shariah_score, :business_screen, :business_reason,
                :debt_screen, :debt_ratio, :debt_reason, :interest_screen, :interest_ratio, :interest_reason,
                :securities_screen, :securities_ratio, :securities_reason, :receivables_screen, :receivables_ratio,
                :receivables_reason, :purification_ratio, :ai_explanation, :haram_reason, :halal_conditions,
                :screened_at, :data_source, :data_quality
            )
            ON CONFLICT(ticker) DO UPDATE SET
                overall_status = excluded.overall_status,
                shariah_score = excluded.shariah_score,
                business_screen = excluded.business_screen,
                business_reason = excluded.business_reason,
                debt_screen = excluded.debt_screen,
                debt_ratio = excluded.debt_ratio,
                debt_reason = excluded.debt_reason,
                interest_screen = excluded.interest_screen,
                interest_ratio = excluded.interest_ratio,
                interest_reason = excluded.interest_reason,
                securities_screen = excluded.securities_screen,
                securities_ratio = excluded.securities_ratio,
                securities_reason = excluded.securities_reason,
                receivables_screen = excluded.receivables_screen,
                receivables_ratio = excluded.receivables_ratio,
                receivables_reason = excluded.receivables_reason,
                purification_ratio = excluded.purification_ratio,
                ai_explanation = excluded.ai_explanation,
                haram_reason = excluded.haram_reason,
                halal_conditions = excluded.halal_conditions,
                screened_at = excluded.screened_at,
                data_source = excluded.data_source,
                data_quality = excluded.data_quality
            """,
            verdict_row,
        )


def get_verdict(ticker: str) -> dict | None:
    """Return a stored verdict and related company metadata for a ticker."""

    ticker_norm = _normalize_ticker(ticker)
    with _connect() as connection:
        verdict = connection.execute("SELECT * FROM verdicts WHERE ticker = ?", (ticker_norm,)).fetchone()
        if verdict is None:
            return None

        company = connection.execute("SELECT * FROM companies WHERE ticker = ?", (ticker_norm,)).fetchone()
        financials = connection.execute("SELECT * FROM financials WHERE ticker = ?", (ticker_norm,)).fetchone()
        zakat_data = connection.execute("SELECT * FROM zakat_data WHERE ticker = ?", (ticker_norm,)).fetchone()

    result = dict(verdict)
    if company is not None:
        result.update({f"company_{key}": company[key] for key in company.keys() if key != "ticker"})
    if financials is not None:
        result.update({f"financial_{key}": financials[key] for key in financials.keys() if key != "ticker"})
    if zakat_data is not None:
        result.update({f"zakat_{key}": zakat_data[key] for key in zakat_data.keys() if key != "ticker"})
    return result


def get_all_verdicts() -> pd.DataFrame:
    """Return all stored verdicts as a DataFrame."""

    with _connect() as connection:
        query = """
            SELECT
                v.*, c.company_name, c.sector_raw, c.sector_classified,
                c.listed_date, c.last_updated,
                f.total_assets, f.total_debt, f.interest_income, f.total_revenue,
                f.accounts_receivable, f.non_compliant_investments, f.cash,
                f.market_cap, f.current_price, f.shares_outstanding, f.fiscal_year,
                f.fetched_at,
                z.cash_per_share, z.receivables_per_share, z.inventory_per_share,
                z.zakatable_per_share, z.calculated_at
            FROM verdicts v
            LEFT JOIN companies c ON c.ticker = v.ticker
            LEFT JOIN financials f ON f.ticker = v.ticker
            LEFT JOIN zakat_data z ON z.ticker = v.ticker
            ORDER BY v.screened_at DESC
        """
        rows = connection.execute(query).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def search_verdicts(query: str) -> pd.DataFrame:
    """Search verdicts using ticker/company name fuzzy matching and SQL filtering."""

    normalized = str(query).strip()
    if not normalized:
        return get_all_verdicts()

    frame = get_all_verdicts()
    if frame.empty:
        return frame

    ticker_series = frame.get("ticker", pd.Series(dtype=object)).astype(str)
    company_series = frame.get("company_name", pd.Series(dtype=object)).astype(str)
    sector_series = frame.get("sector_classified", pd.Series(dtype=object)).astype(str)
    mask = (
        ticker_series.str.contains(normalized, case=False, na=False)
        | company_series.str.contains(normalized, case=False, na=False)
        | sector_series.str.contains(normalized, case=False, na=False)
    )
    filtered = frame.loc[mask].copy()

    if not filtered.empty:
        return filtered.reset_index(drop=True)

    candidates = list(dict.fromkeys(ticker_series.tolist() + company_series.tolist() + sector_series.tolist()))
    fuzzy_matches = difflib.get_close_matches(normalized, candidates, n=10, cutoff=0.45)
    if not fuzzy_matches:
        return filtered.reset_index(drop=True)

    fuzzy_mask = ticker_series.isin(fuzzy_matches) | company_series.isin(fuzzy_matches) | sector_series.isin(fuzzy_matches)
    return frame.loc[fuzzy_mask].reset_index(drop=True)


def get_verdict_reason(ticker: str) -> str:
    """Return a complete human-readable reason string built from verdict fields."""

    verdict = get_verdict(ticker)
    if not verdict:
        return "No verdict stored for this ticker."

    sections: list[str] = []
    business_reason = _coerce_text(verdict.get("business_reason"))
    debt_reason = _coerce_text(verdict.get("debt_reason"))
    interest_reason = _coerce_text(verdict.get("interest_reason"))
    securities_reason = _coerce_text(verdict.get("securities_reason"))
    receivables_reason = _coerce_text(verdict.get("receivables_reason"))
    ai_explanation = _coerce_text(verdict.get("ai_explanation"))
    haram_reason = _coerce_text(verdict.get("haram_reason"))
    halal_conditions = _coerce_text(verdict.get("halal_conditions"))

    sections.append(f"Status: {_coerce_text(verdict.get('overall_status')) or 'UNKNOWN'}")

    if business_reason:
        sections.append(f"Business screen: {business_reason}")
    if debt_reason:
        sections.append(f"Debt screen: {debt_reason}")
    if interest_reason:
        sections.append(f"Interest screen: {interest_reason}")
    if securities_reason:
        sections.append(f"Securities screen: {securities_reason}")
    if receivables_reason:
        sections.append(f"Receivables screen: {receivables_reason}")
    if ai_explanation:
        sections.append(f"AI explanation: {ai_explanation}")
    if haram_reason:
        sections.append(f"Haram reason: {haram_reason}")
    if halal_conditions:
        sections.append(f"Halal conditions: {halal_conditions}")

    return "\n".join(sections)


def needs_refresh(ticker: str, days: int = 90) -> bool:
    """Return True when a verdict should be refreshed."""

    verdict = get_verdict(ticker)
    if not verdict:
        return True

    screened_at = _parse_datetime(verdict.get("screened_at"))
    if screened_at is None:
        return True
    if datetime.now(timezone.utc) - screened_at > timedelta(days=days):
        return True

    financials = get_financials(ticker)
    if not financials:
        return False

    fetched_at = _parse_datetime(financials.get("fetched_at"))
    if fetched_at is None:
        return True

    if fetched_at > screened_at:
        return True

    # Treat a very recent financial filing as a refresh trigger, since the
    # screening verdict may now be stale relative to the new quarterly data.
    if datetime.now(timezone.utc) - fetched_at <= timedelta(days=21):
        return True

    fiscal_year = _coerce_text(financials.get("fiscal_year"))
    if fiscal_year:
        year_token = fiscal_year.strip().split()[0]
        if year_token.isdigit():
            try:
                fiscal_year_int = int(year_token)
                if fiscal_year_int >= datetime.now().year:
                    return True
            except Exception:
                pass

    return False


def get_screening_stats() -> dict:
    """Return aggregate screening statistics."""

    frame = get_all_verdicts()
    if frame.empty:
        return {
            "total": 0,
            "halal_count": 0,
            "haram_count": 0,
            "doubtful_count": 0,
            "last_run": None,
            "coverage_pct": 0.0,
        }

    statuses = frame.get("overall_status", pd.Series(dtype=object)).astype(str).str.upper()
    total = int(len(frame))
    last_run = frame.get("screened_at", pd.Series(dtype=object)).dropna().astype(str).max() if "screened_at" in frame.columns else None

    with _connect() as connection:
        total_companies = int(connection.execute("SELECT COUNT(*) AS count FROM companies").fetchone()["count"])

    coverage_pct = round((total / total_companies) * 100, 2) if total_companies > 0 else 0.0

    return {
        "total": total,
        "halal_count": int((statuses == "HALAL").sum()),
        "haram_count": int((statuses == "HARAM").sum()),
        "doubtful_count": int((statuses == "DOUBTFUL").sum()),
        "last_run": last_run,
        "coverage_pct": coverage_pct,
    }


def save_financials(ticker: str, data: dict) -> None:
    """Persist the latest financial snapshot for a ticker."""

    ticker_norm = _normalize_ticker(ticker)
    row = {
        "ticker": ticker_norm,
        "total_assets": data.get("total_assets"),
        "total_debt": data.get("total_debt"),
        "interest_income": data.get("interest_income"),
        "total_revenue": data.get("total_revenue"),
        "accounts_receivable": data.get("accounts_receivable"),
        "non_compliant_investments": data.get("non_compliant_investments"),
        "cash": data.get("cash"),
        "market_cap": data.get("market_cap"),
        "current_price": data.get("current_price"),
        "shares_outstanding": data.get("shares_outstanding"),
        "fiscal_year": _coerce_text(data.get("fiscal_year")),
        "fetched_at": _coerce_text(data.get("fetched_at")) or _utc_now(),
    }

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO financials (
                ticker, total_assets, total_debt, interest_income, total_revenue,
                accounts_receivable, non_compliant_investments, cash, market_cap,
                current_price, shares_outstanding, fiscal_year, fetched_at
            ) VALUES (
                :ticker, :total_assets, :total_debt, :interest_income, :total_revenue,
                :accounts_receivable, :non_compliant_investments, :cash, :market_cap,
                :current_price, :shares_outstanding, :fiscal_year, :fetched_at
            )
            ON CONFLICT(ticker) DO UPDATE SET
                total_assets = excluded.total_assets,
                total_debt = excluded.total_debt,
                interest_income = excluded.interest_income,
                total_revenue = excluded.total_revenue,
                accounts_receivable = excluded.accounts_receivable,
                non_compliant_investments = excluded.non_compliant_investments,
                cash = excluded.cash,
                market_cap = excluded.market_cap,
                current_price = excluded.current_price,
                shares_outstanding = excluded.shares_outstanding,
                fiscal_year = excluded.fiscal_year,
                fetched_at = excluded.fetched_at
            """,
            row,
        )


def get_financials(ticker: str) -> dict | None:
    """Return a stored financial snapshot for a ticker."""

    ticker_norm = _normalize_ticker(ticker)
    with _connect() as connection:
        row = connection.execute("SELECT * FROM financials WHERE ticker = ?", (ticker_norm,)).fetchone()
    return dict(row) if row is not None else None


def save_zakat_data(ticker: str, data: dict) -> None:
    """Persist per-share zakat data for a ticker."""

    ticker_norm = _normalize_ticker(ticker)
    row = {
        "ticker": ticker_norm,
        "cash_per_share": data.get("cash_per_share"),
        "receivables_per_share": data.get("receivables_per_share"),
        "inventory_per_share": data.get("inventory_per_share"),
        "zakatable_per_share": data.get("zakatable_per_share"),
        "calculated_at": _coerce_text(data.get("calculated_at")) or _utc_now(),
    }

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO zakat_data (
                ticker, cash_per_share, receivables_per_share, inventory_per_share,
                zakatable_per_share, calculated_at
            ) VALUES (
                :ticker, :cash_per_share, :receivables_per_share, :inventory_per_share,
                :zakatable_per_share, :calculated_at
            )
            ON CONFLICT(ticker) DO UPDATE SET
                cash_per_share = excluded.cash_per_share,
                receivables_per_share = excluded.receivables_per_share,
                inventory_per_share = excluded.inventory_per_share,
                zakatable_per_share = excluded.zakatable_per_share,
                calculated_at = excluded.calculated_at
            """,
            row,
        )


def get_zakat_data(ticker: str) -> dict | None:
    """Return stored per-share zakat data for a ticker."""

    ticker_norm = _normalize_ticker(ticker)
    with _connect() as connection:
        row = connection.execute("SELECT * FROM zakat_data WHERE ticker = ?", (ticker_norm,)).fetchone()
    return dict(row) if row is not None else None
