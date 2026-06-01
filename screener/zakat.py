"""Production-grade Islamic zakat calculator for stock portfolios.

This module follows AAOIFI Standard No. 7 principles for equity zakat
calculation and provides:

- live nisab pricing for silver and gold
- three zakat methods for stock portfolios
- per-stock and portfolio-level calculations
- HTML report generation
- PDF export via fpdf2

Defaults are intentionally conservative and Pakistan-friendly:
- silver nisab uses 595g
- gold nisab uses 85g
- lunar-year annualization is available through ``ZAKAT_RATE_LUNAR``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fpdf import FPDF, HTMLMixin

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

NISAB_SILVER_GRAMS = 595
NISAB_GOLD_GRAMS = 85
LUNAR_YEAR_FACTOR = 354.367 / 365.25
ZAKAT_RATE = 0.025
ZAKAT_RATE_LUNAR = 1 / 40

METAL_CACHE_TTL_HOURS = 6
SILVER_CACHE_FILE = DATA_DIR / "silver_cache.json"
GOLD_CACHE_FILE = DATA_DIR / "gold_cache.json"

METALS_LIVE_URLS = {
    "silver": "https://api.metals.live/v1/spot/silver",
    "gold": "https://api.metals.live/v1/spot/gold",
}
USD_PKR_URL = "https://open.er-api.com/v6/latest/USD"

DEFAULT_FALLBACKS = {
    "silver": 310.0,  # PKR per gram, conservative fallback to be updated quarterly (2026-06-01)
    "gold": 24000.0,  # PKR per gram, conservative fallback to be updated quarterly (2026-06-01)
}

PROHIBITED_BUSINESS_KEYWORDS = (
    "bank",
    "insurance",
    "alcohol",
    "liquor",
    "tobacco",
    "casino",
    "gambling",
    "pork",
    "weapon",
    "porn",
    "adult entertainment",
    "conventional finance",
)

METHOD_NAMES = {
    "net_assets": "Net Assets Method (AAOIFI)",
    "market_value": "Market Value Method",
    "dividend_income": "Dividend / Income Method",
}


def get_silver_price_pkr() -> float:
    """Return live silver price in PKR per gram using the documented fallback chain."""

    return float(_get_metal_price_pkr("silver", DEFAULT_FALLBACKS["silver"])["price_per_gram_pkr"])


def get_gold_price_pkr() -> float:
    """Return live gold price in PKR per gram using the documented fallback chain."""

    return float(_get_metal_price_pkr("gold", DEFAULT_FALLBACKS["gold"])["price_per_gram_pkr"])


def calculate_nisab_pkr(metal: str = "silver") -> dict:
    """Calculate the nisab threshold in PKR for the selected benchmark metal."""

    normalized = str(metal).strip().lower()
    if normalized not in {"silver", "gold"}:
        normalized = "silver"

    grams_required = NISAB_SILVER_GRAMS if normalized == "silver" else NISAB_GOLD_GRAMS
    price_info = _get_metal_price_pkr(normalized, DEFAULT_FALLBACKS[normalized])
    nisab_value = grams_required * float(price_info["price_per_gram_pkr"])

    note = (
        "Nisab using silver is lower and more conservative. It is commonly preferred by Pakistani Hanafi scholars, including Mufti Taqi Usmani."
        if normalized == "silver"
        else "Gold nisab is the alternative benchmark used by some scholars and institutions."
    )

    return {
        "metal": normalized,
        "grams_required": grams_required,
        "price_per_gram_pkr": float(price_info["price_per_gram_pkr"]),
        "nisab_pkr": float(nisab_value),
        "source": price_info["source"],
        "fetched_at": price_info["fetched_at"],
        "note": note,
        "lunar_year_factor": LUNAR_YEAR_FACTOR,
    }


def get_nisab_pkr() -> float:
    """Compatibility helper used by the dashboard."""

    return float(calculate_nisab_pkr("silver")["nisab_pkr"])


def calculate_stock_zakat(
    ticker: str,
    shares_held: int,
    purchase_price: float,
    current_price: float,
    method: str = "net_assets",
    financial_data: dict | None = None,
    nisab_metal: str = "silver",
) -> dict:
    """Calculate zakat for a single stock position."""

    method_used = _normalize_method(method)
    ticker_normalized = _normalize_ticker(ticker)
    shares = _to_float(shares_held)
    purchase = _to_float(purchase_price)
    current = _to_float(current_price)
    snapshot = _resolve_financial_snapshot(ticker_normalized, financial_data)

    if np.isnan(current):
        current = _first_numeric(snapshot.get("current_price"), snapshot.get("last_price"))
    if np.isnan(purchase):
        purchase = _first_numeric(snapshot.get("purchase_price"), current)

    market_value = current * shares if np.isfinite(current) and np.isfinite(shares) else np.nan
    cost_basis = purchase * shares if np.isfinite(purchase) and np.isfinite(shares) else np.nan
    unrealized_gain = market_value - cost_basis if np.isfinite(market_value) and np.isfinite(cost_basis) else np.nan

    shares_outstanding = _first_numeric(snapshot.get("shares_outstanding"), snapshot.get("sharesOutstanding"))
    cash = _first_numeric(snapshot.get("cash"), snapshot.get("total_cash"), snapshot.get("cash_and_cash_equivalents"))
    receivables = _first_numeric(snapshot.get("receivables"), snapshot.get("accounts_receivable"), snapshot.get("net_receivables"))
    inventory = _first_numeric(snapshot.get("inventory"), snapshot.get("inventories"))
    annual_dividend_per_share = _first_numeric(
        snapshot.get("annual_dividend_per_share"),
        snapshot.get("dividend_per_share"),
        snapshot.get("trailing_annual_dividend_rate"),
        snapshot.get("dividend_rate"),
    )

    cash_per_share = cash / shares_outstanding if np.isfinite(cash) and np.isfinite(shares_outstanding) and shares_outstanding > 0 else np.nan
    receivables_per_share = receivables / shares_outstanding if np.isfinite(receivables) and np.isfinite(shares_outstanding) and shares_outstanding > 0 else np.nan
    inventory_per_share = inventory / shares_outstanding if np.isfinite(inventory) and np.isfinite(shares_outstanding) and shares_outstanding > 0 else np.nan
    zakatable_per_share = cash_per_share + receivables_per_share + inventory_per_share

    dividend_income = _first_numeric(snapshot.get("annual_dividends_received"), snapshot.get("dividend_income"))
    if not np.isfinite(dividend_income) and np.isfinite(annual_dividend_per_share) and np.isfinite(shares):
        dividend_income = annual_dividend_per_share * shares

    if method_used == "market_value":
        zakatable_total = market_value
        zakat_amount = market_value * ZAKAT_RATE_LUNAR if np.isfinite(market_value) else np.nan
    elif method_used == "dividend_income":
        zakatable_total = dividend_income
        zakat_amount = dividend_income * 0.10 if np.isfinite(dividend_income) else np.nan
    else:
        zakatable_total = zakatable_per_share * shares if np.isfinite(zakatable_per_share) and np.isfinite(shares) else np.nan
        zakat_amount = zakatable_total * ZAKAT_RATE_LUNAR if np.isfinite(zakatable_total) else np.nan

    purification_rate = _resolve_purification_rate(snapshot)
    purification_amount = market_value * purification_rate if np.isfinite(market_value) and purification_rate > 0 else 0.0
    total_obligation = (zakat_amount if np.isfinite(zakat_amount) else 0.0) + purification_amount

    nisab_info = calculate_nisab_pkr(nisab_metal)
    portfolio_value = market_value
    zakat_due = bool(np.isfinite(portfolio_value) and portfolio_value >= nisab_info["nisab_pkr"])

    overall_status = str(snapshot.get("overall_status", snapshot.get("status", ""))).upper().strip()
    business_screen = _booleanish(snapshot.get("business_screen"))
    haram_reason = str(snapshot.get("haram_reason", "")).strip()
    non_compliant = _is_non_compliant(overall_status, business_screen, snapshot)

    return {
        "ticker": ticker_normalized,
        "company_name": str(snapshot.get("company_name", ticker_normalized)).strip() or ticker_normalized,
        "sector": str(snapshot.get("sector", snapshot.get("sector_classified", "Unknown"))).strip() or "Unknown",
        "shares_held": float(shares) if np.isfinite(shares) else np.nan,
        "purchase_price": float(purchase) if np.isfinite(purchase) else np.nan,
        "current_price": float(current) if np.isfinite(current) else np.nan,
        "market_value": float(market_value) if np.isfinite(market_value) else np.nan,
        "cost_basis": float(cost_basis) if np.isfinite(cost_basis) else np.nan,
        "unrealized_gain": float(unrealized_gain) if np.isfinite(unrealized_gain) else np.nan,
        "method_used": method_used,
        "method_name": METHOD_NAMES[method_used],
        "cash_per_share": float(cash_per_share) if np.isfinite(cash_per_share) else np.nan,
        "receivables_per_share": float(receivables_per_share) if np.isfinite(receivables_per_share) else np.nan,
        "inventory_per_share": float(inventory_per_share) if np.isfinite(inventory_per_share) else np.nan,
        "zakatable_per_share": float(zakatable_per_share) if np.isfinite(zakatable_per_share) else np.nan,
        "zakatable_total": float(zakatable_total) if np.isfinite(zakatable_total) else np.nan,
        "zakat_rate": ZAKAT_RATE_LUNAR,
        "solar_zakat_rate": ZAKAT_RATE,
        "lunar_year_factor": LUNAR_YEAR_FACTOR,
        "zakat_amount": float(zakat_amount) if np.isfinite(zakat_amount) else np.nan,
        "purification_amount": float(purification_amount),
        "purification_rate": float(purification_rate),
        "total_obligation": float(total_obligation),
        "nisab_threshold": float(nisab_info["nisab_pkr"]),
        "nisab_metal": nisab_info["metal"],
        "portfolio_value": float(portfolio_value) if np.isfinite(portfolio_value) else np.nan,
        "zakat_due": zakat_due,
        "nisab_note": (
            "This position alone is below nisab. Include all assets to determine if Zakat is due."
            if not zakat_due
            else "This position is above nisab on its standalone market value."
        ),
        "scholarly_note": (
            "Calculated using AAOIFI Standard No. 7. Net assets method treats the investor as a proportionate owner in the company's zakatable assets rather than its full market value."
            if method_used == "net_assets"
            else (
                "Calculated using AAOIFI Standard No. 7 with the market value method. Some scholars accept this conservative retail approach, although it may include unrealized gains."
                if method_used == "market_value"
                else "Calculated using a minority dividend/income approach that treats dividends as zakatable income."
            )
        ),
        "overall_status": overall_status,
        "business_screen": business_screen,
        "haram_reason": haram_reason,
        "non_compliant": non_compliant,
        "warnings": _build_warnings(snapshot, method_used),
        "dividend_income": float(dividend_income) if np.isfinite(dividend_income) else np.nan,
        "calculation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def calculate_portfolio_zakat(
    holdings: list[dict] | dict,
    method: str = "net_assets",
    include_other_assets: float = 0.0,
    debts_deductible: float = 0.0,
    nisab_metal: str = "silver",
) -> dict:
    """Calculate zakat across a portfolio of holdings and return a structured result."""

    holdings_list = _coerce_holdings(holdings)
    method_used = _normalize_method(method)
    zakat_details: list[dict] = []
    breakdown_by_sector: dict[str, dict[str, float]] = {}
    non_compliant_holdings: list[dict] = []

    for holding in holdings_list:
        shares_value = _to_float(holding.get("shares", holding.get("shares_held", 0)))
        shares_held = int(shares_value) if np.isfinite(shares_value) else 0
        detail = calculate_stock_zakat(
            ticker=str(holding.get("ticker", "")).strip(),
            shares_held=shares_held,
            purchase_price=_to_float(holding.get("purchase_price", holding.get("cost_basis_per_share", np.nan))),
            current_price=_to_float(holding.get("current_price", np.nan)),
            method=method_used,
            financial_data=holding.get("financial_data") or holding,
        )
        detail["sector"] = str(holding.get("sector") or detail.get("sector") or "Unknown")
        detail["company_name"] = str(holding.get("company_name") or detail.get("company_name") or detail["ticker"])
        detail["purification_rate"] = float(detail.get("purification_rate", 0.0) or 0.0)
        detail["source_holding"] = _sanitize_source_holding(holding)

        if detail.get("non_compliant"):
            non_compliant_holdings.append(
                {
                    "ticker": detail["ticker"],
                    "shares": detail.get("shares_held", 0),
                    "note": _non_compliant_note(detail),
                }
            )

        sector_key = str(detail.get("sector", "Unknown")) or "Unknown"
        sector_bucket = breakdown_by_sector.setdefault(sector_key, {"value": 0.0, "zakat": 0.0})
        sector_bucket["value"] += float(detail.get("market_value", 0.0) or 0.0)
        sector_bucket["zakat"] += float(detail.get("zakat_amount", 0.0) or 0.0)

        zakat_details.append(detail)

    frame = pd.DataFrame(zakat_details)
    total_market_value = float(pd.to_numeric(frame.get("market_value", pd.Series(dtype=float)), errors="coerce").sum(min_count=1) or 0.0)
    total_zakatable_base = float(pd.to_numeric(frame.get("zakatable_total", pd.Series(dtype=float)), errors="coerce").sum(min_count=1) or 0.0)
    total_purification = float(pd.to_numeric(frame.get("purification_amount", pd.Series(dtype=float)), errors="coerce").sum(min_count=1) or 0.0)
    total_zakat = float(pd.to_numeric(frame.get("zakat_amount", pd.Series(dtype=float)), errors="coerce").sum(min_count=1) or 0.0)
    total_assets = total_market_value + _to_float(include_other_assets)
    net_zakatable = total_zakatable_base + _to_float(include_other_assets) - _to_float(debts_deductible)
    nisab_info = calculate_nisab_pkr(nisab_metal)
    zakat_due = bool(np.isfinite(net_zakatable) and net_zakatable >= nisab_info["nisab_pkr"])

    summary = {
        "total_market_value": float(total_market_value),
        "total_zakatable_base": float(total_zakatable_base),
        "other_assets": float(_to_float(include_other_assets)),
        "total_assets": float(total_assets),
        "deductible_debts": float(_to_float(debts_deductible)),
        "net_zakatable": float(net_zakatable),
        "nisab_pkr": float(nisab_info["nisab_pkr"]),
        "nisab_metal": nisab_info["metal"],
        "zakat_due": zakat_due,
        "total_zakat": float(total_zakat),
        "total_purification": float(total_purification),
        "total_obligation": float(total_zakat + total_purification),
        "method": method_used,
        "method_name": METHOD_NAMES[method_used],
        "calculation_date": datetime.now().strftime("%Y-%m-%d"),
        "calculation_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "hawl_reminder": "Zakat is only due if this wealth has been held for one full lunar year (hawl). Ensure your holdings have been above nisab for 354 days or more.",
        "nisab_source": nisab_info["source"],
        "nisab_note": nisab_info["note"],
        "lunar_year_factor": LUNAR_YEAR_FACTOR,
        "zakat_rate": ZAKAT_RATE_LUNAR,
        "solar_zakat_rate": ZAKAT_RATE,
    }

    return {
        "holdings_detail": zakat_details,
        "summary": summary,
        "breakdown_by_sector": {
            sector: {"value": round(values["value"], 2), "zakat": round(values["zakat"], 2)}
            for sector, values in sorted(breakdown_by_sector.items())
        },
        "non_compliant_holdings": non_compliant_holdings,
        "nisab_info": nisab_info,
    }


def generate_zakat_report(portfolio_result: dict) -> str:
    """Generate a professional HTML zakat report suitable for scholar review."""

    summary = portfolio_result.get("summary", {}) if isinstance(portfolio_result, dict) else {}
    holdings = portfolio_result.get("holdings_detail", []) if isinstance(portfolio_result, dict) else []
    non_compliant = portfolio_result.get("non_compliant_holdings", []) if isinstance(portfolio_result, dict) else []
    nisab_info = portfolio_result.get("nisab_info", {}) if isinstance(portfolio_result, dict) else {}

    title = "Zakat Calculation Report"
    generated_at = summary.get("calculation_datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    method_name = summary.get("method_name") or METHOD_NAMES.get(summary.get("method", "net_assets"), "Net Assets Method (AAOIFI)")
    metal = str(nisab_info.get("metal", "silver")).title()
    nisab_threshold = _format_pkr(summary.get("nisab_pkr", nisab_info.get("nisab_pkr", 0.0)))
    total_obligation = _format_pkr(summary.get("total_obligation", 0.0))
    total_zakat = _format_pkr(summary.get("total_zakat", 0.0))
    total_purification = _format_pkr(summary.get("total_purification", 0.0))
    total_assets = _format_pkr(summary.get("total_assets", 0.0))
    net_zakatable = _format_pkr(summary.get("net_zakatable", 0.0))
    hawl_reminder = str(summary.get("hawl_reminder", ""))
    disclaimer = "This calculation is based on AAOIFI Standard No. 7. Please verify with a qualified Islamic scholar."

    rows_html = []
    for holding in holdings:
        rows_html.append(
            "<tr>"
            f"<td>{_escape_html(holding.get('ticker'))}</td>"
            f"<td class='num'>{_format_number(holding.get('shares_held'))}</td>"
            f"<td class='num'>{_format_pkr(holding.get('market_value'))}</td>"
            f"<td class='num'>{_format_pkr(holding.get('zakatable_total'))}</td>"
            f"<td class='num'>{_format_pkr(holding.get('zakat_amount'))}</td>"
            f"<td class='num'>{_format_pkr(holding.get('purification_amount'))}</td>"
            f"<td>{_escape_html(holding.get('method_name'))}</td>"
            "</tr>"
        )

    non_compliant_html = ""
    if non_compliant:
        items = "".join(
            f"<li><b>{_escape_html(item.get('ticker'))}</b> - {_escape_html(item.get('note'))}</li>"
            for item in non_compliant
        )
        non_compliant_html = f"<div class='warning'><h3>Non-compliant holdings</h3><ul>{items}</ul></div>"

    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>{_escape_html(title)}</title>
        <style>
            body {{ font-family: Arial, Helvetica, sans-serif; color: #17301f; background: #ffffff; margin: 0; padding: 28px; }}
            .page {{ max-width: 980px; margin: 0 auto; }}
            .title {{ font-size: 30px; font-weight: 800; letter-spacing: 0.02em; margin: 0 0 6px; }}
            .subtitle {{ color: #5c6b63; margin: 0 0 18px; }}
            .banner {{ border: 1px solid #d9e0d8; border-radius: 16px; padding: 18px; background: linear-gradient(135deg, #f6fbf7, #ffffff); margin-bottom: 18px; }}
            .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0; }}
            .card {{ border: 1px solid #d9e0d8; border-radius: 14px; padding: 14px; background: #fff; }}
            .label {{ text-transform: uppercase; font-size: 11px; letter-spacing: 0.08em; color: #6f7b74; margin-bottom: 6px; }}
            .value {{ font-size: 18px; font-weight: 800; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
            th, td {{ border: 1px solid #d9e0d8; padding: 10px 9px; text-align: left; vertical-align: top; }}
            th {{ background: #f2f7f3; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
            td.num {{ text-align: right; white-space: nowrap; }}
            .section {{ margin-top: 22px; }}
            .section h2 {{ font-size: 18px; margin: 0 0 10px; }}
            .note {{ border-left: 4px solid #1a6b3c; background: #f3faf5; padding: 12px 14px; border-radius: 10px; }}
            .warning {{ border-left: 4px solid #c9a84c; background: #fffdf5; padding: 12px 14px; border-radius: 10px; margin-top: 12px; }}
            .footer {{ margin-top: 26px; padding-top: 14px; border-top: 1px solid #d9e0d8; color: #5c6b63; font-size: 12px; }}
            .signature {{ margin-top: 24px; display: flex; justify-content: space-between; gap: 20px; }}
            .sig-line {{ border-top: 1px solid #27352c; padding-top: 8px; min-width: 280px; }}
            .muted {{ color: #5c6b63; }}
        </style>
    </head>
    <body>
        <div class="page">
            <div class="banner">
                <div class="title">{title}</div>
                <p class="subtitle">Generated on {generated_at}. Method: {method_name}. Nisab benchmark: {metal}.</p>
            </div>

            <div class="grid">
                <div class="card"><div class="label">Nisab threshold</div><div class="value">{nisab_threshold}</div><div class="muted">Using {metal} benchmark</div></div>
                <div class="card"><div class="label">Total Zakat</div><div class="value">{total_zakat}</div><div class="muted">Before purification</div></div>
                <div class="card"><div class="label">Total obligation</div><div class="value">{total_obligation}</div><div class="muted">Zakat + purification</div></div>
            </div>

            <div class="grid">
                <div class="card"><div class="label">Total assets</div><div class="value">{total_assets}</div></div>
                <div class="card"><div class="label">Net zakatable</div><div class="value">{net_zakatable}</div></div>
                <div class="card"><div class="label">Purification</div><div class="value">{total_purification}</div></div>
            </div>

            <div class="section">
                <h2>Method Reference</h2>
                <div class="note">
                    <b>{_escape_html(method_name)}</b><br />
                    AAOIFI Standard No. 7 treats equity ownership as a proportionate claim on zakatable assets. The calculator applies the selected method to each holding and aggregates the result across the portfolio.
                </div>
            </div>

            <div class="section">
                <h2>Per-Holding Table</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Shares</th>
                            <th>Value</th>
                            <th>Zakatable</th>
                            <th>Zakat</th>
                            <th>Purification</th>
                            <th>Method</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html) if rows_html else '<tr><td colspan="7">No holdings supplied.</td></tr>'}
                    </tbody>
                </table>
            </div>

            {non_compliant_html}

            <div class="section">
                <h2>Hawl Reminder</h2>
                <div class="note">{_escape_html(hawl_reminder)}</div>
            </div>

            <div class="section">
                <h2>Purification and Verification</h2>
                <div class="note">
                    Purification amounts are shown separately for any impermissible income exposures. If a holding is non-compliant, scholars may differ on whether to exclude it from Zakat or dispose of it entirely; seek scholar review before acting.
                </div>
            </div>

            <div class="signature">
                <div class="sig-line">Investor signature</div>
                <div class="sig-line">Scholar verification</div>
            </div>

            <div class="footer">
                <p><b>Disclaimer:</b> {_escape_html(disclaimer)}</p>
                <p class="muted">This document is a calculation aid, not a fatwa.</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_report.strip()


def export_zakat_pdf(portfolio_result: dict, output_path: str) -> str:
    """Convert a zakat report to PDF using fpdf2 and return the saved path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pdf = _ZakatPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.write_html(generate_zakat_report(portfolio_result))
    pdf.output(str(output))
    return str(output)


def calculate_zakat_per_stock(ticker: str, shares_held: float, method: str = "assets") -> dict:
    """Backward-compatible helper that maps the old API to the new stock calculator."""

    mapped_method = _normalize_method(method)
    current_price = _guess_current_price(ticker)
    return calculate_stock_zakat(
        ticker=ticker,
        shares_held=int(_safe_int(shares_held)),
        purchase_price=current_price,
        current_price=current_price,
        method=mapped_method,
    )


def is_zakat_due(total_portfolio_value: float) -> bool:
    """Return whether a portfolio value reaches the nisab threshold."""

    value = _to_float(total_portfolio_value)
    return bool(np.isfinite(value) and value >= get_nisab_pkr())


class _ZakatPDF(FPDF, HTMLMixin):
    """PDF layout for zakat reports with a light decorative watermark."""

    def header(self) -> None:  # type: ignore[override]
        self.set_draw_color(26, 107, 60)
        self.set_line_width(0.7)
        self.rect(8, 8, 194, 281)
        self.line(8, 18, 202, 18)

        self.set_fill_color(235, 241, 236)
        self.ellipse(164, 14, 24, 24, style="F")
        self.set_fill_color(255, 255, 255)
        self.ellipse(171, 14, 20, 24, style="F")

        self.set_font("Helvetica", "B", 13)
        self.set_text_color(26, 107, 60)
        self.cell(0, 8, "Zakat Calculation Report", align="C", ln=1)
        self.set_text_color(80, 80, 80)
        self.set_font("Helvetica", size=9)
        self.cell(0, 5, "PSX Halal Screener | AAOIFI Standard No. 7", align="C", ln=1)
        self.ln(2)

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="C", ln=1)
        self.cell(0, 5, "Generated by PSX Halal Screener | Not a fatwa - consult a qualified scholar", align="C")


def _get_metal_price_pkr(metal: str, fallback_price: float) -> dict:
    normalized = str(metal).strip().lower()
    cache_file = SILVER_CACHE_FILE if normalized == "silver" else GOLD_CACHE_FILE
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached = _read_cached_metal_price(cache_file, normalized)
    if cached is not None:
        return cached

    source_label = ""
    price_per_gram = np.nan
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        metals_response = requests.get(METALS_LIVE_URLS[normalized], timeout=15)
        metals_response.raise_for_status()
        metals_payload = metals_response.json()
        metals_oz = _extract_numeric_price(metals_payload)
        usd_pkr = _fetch_usd_pkr_rate()

        if np.isfinite(metals_oz) and metals_oz > 0 and np.isfinite(usd_pkr) and usd_pkr > 0:
            price_per_gram = (metals_oz * usd_pkr) / 31.1035
            source_label = "metals.live API + open.er-api USD/PKR"
    except Exception as exc:
        logger.info("Failed to fetch %s price from primary source: %s", normalized, exc)

    if not np.isfinite(price_per_gram) or price_per_gram <= 0:
        cached = _read_cached_metal_price(cache_file, normalized)
        if cached is not None:
            logger.info("Using cached %s price", normalized)
            return cached

    if not np.isfinite(price_per_gram) or price_per_gram <= 0:
        price_per_gram = float(fallback_price)
        source_label = f"hardcoded fallback PKR {fallback_price:.2f} per gram"

    result = {
        "metal": normalized,
        "price_per_gram_pkr": float(price_per_gram),
        "source": source_label or "fallback",
        "fetched_at": fetched_at,
    }
    _write_cached_metal_price(cache_file, result)
    logger.info("Using %s price source: %s", normalized, result["source"])
    return result


def _fetch_usd_pkr_rate() -> float:
    try:
        response = requests.get(USD_PKR_URL, timeout=15)
        response.raise_for_status()
        payload = response.json()
        rates = payload.get("rates", {}) if isinstance(payload, dict) else {}
        return _to_float(rates.get("PKR"))
    except Exception as exc:
        logger.info("USD/PKR lookup failed: %s", exc)
        return np.nan


def _extract_numeric_price(payload: object) -> float:
    if isinstance(payload, dict):
        for key in ("price", "value", "last", "close"):
            value = _to_float(payload.get(key))
            if np.isfinite(value) and value > 0:
                return value
        for value in payload.values():
            nested = _extract_numeric_price(value)
            if np.isfinite(nested) and nested > 0:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (list, tuple)) and item:
                for value in reversed(item):
                    numeric = _to_float(value)
                    if np.isfinite(numeric) and numeric > 0:
                        return numeric
            else:
                numeric = _extract_numeric_price(item)
                if np.isfinite(numeric) and numeric > 0:
                    return numeric
    return np.nan


def _read_cached_metal_price(cache_file: Path, metal: str) -> dict | None:
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if str(payload.get("metal", "")).lower() != metal:
            return None
        fetched_at_raw = payload.get("fetched_at")
        if fetched_at_raw:
            fetched_at = _parse_timestamp(fetched_at_raw)
            if fetched_at and (datetime.now(timezone.utc) - fetched_at).total_seconds() > METAL_CACHE_TTL_HOURS * 3600:
                return None
        price = _to_float(payload.get("price_per_gram_pkr"))
        if not np.isfinite(price) or price <= 0:
            return None
        payload["price_per_gram_pkr"] = float(price)
        return payload
    except Exception as exc:
        logger.info("Failed to read cached %s price: %s", metal, exc)
        return None


def _write_cached_metal_price(cache_file: Path, payload: dict) -> None:
    try:
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("Failed to write cached %s price: %s", payload.get("metal", "metal"), exc)


def _parse_timestamp(value: object) -> datetime | None:
    try:
        parsed = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return None


def _resolve_financial_snapshot(ticker: str, financial_data: dict | None) -> dict:
    if isinstance(financial_data, dict) and financial_data:
        return dict(financial_data)

    company = None
    try:
        company = yf.Ticker(_normalize_ticker(ticker))
    except Exception as exc:
        logger.info("Could not initialize yfinance ticker for %s: %s", ticker, exc)

    if company is None:
        return {"ticker": ticker}

    info = _safe_dict(getattr(company, "info", None))
    fast_info = _safe_dict(getattr(company, "fast_info", None))
    balance_sheet = _safe_frame(getattr(company, "balance_sheet", None))
    financials = _safe_frame(getattr(company, "financials", None))
    cashflow = _safe_frame(getattr(company, "cashflow", None))

    return {
        "ticker": ticker,
        "company_name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector") or info.get("industry") or "Unknown",
        "current_price": _first_numeric(fast_info.get("last_price"), info.get("currentPrice"), info.get("regularMarketPrice")),
        "shares_outstanding": _first_numeric(info.get("sharesOutstanding"), fast_info.get("shares"), fast_info.get("shares_outstanding")),
        "cash": _first_numeric(
            info.get("totalCash"),
            info.get("cashAndCashEquivalents"),
            _statement_value(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments", "Cash"]),
        ),
        "receivables": _first_numeric(_statement_value(balance_sheet, ["Accounts Receivable", "Net Receivables", "Trade Receivables"])),
        "inventory": _first_numeric(_statement_value(balance_sheet, ["Inventory", "Inventories", "Merchandise Inventory"])),
        "annual_dividends_received": _first_numeric(info.get("trailingAnnualDividendRate"), info.get("dividendRate")),
        "trailing_annual_dividend_rate": _first_numeric(info.get("trailingAnnualDividendRate"), info.get("dividendRate")),
        "business_screen": _infer_business_screen(info.get("sector"), info.get("industry")),
        "overall_status": str(info.get("overall_status", "")).upper(),
        "haram_reason": info.get("haram_reason", ""),
        "purification_rate": _first_numeric(info.get("purification_rate"), info.get("purificationRatio")),
        "financials_frame": financials,
        "cashflow_frame": cashflow,
    }


def _resolve_purification_rate(snapshot: dict) -> float:
    candidates = [
        snapshot.get("purification_rate"),
        snapshot.get("purification_ratio"),
        snapshot.get("purification_ratio_percent"),
        snapshot.get("haram_income_ratio"),
        snapshot.get("impure_income_ratio"),
    ]
    rate = _first_numeric(*candidates)
    if not np.isfinite(rate) or rate <= 0:
        return 0.0
    if rate > 1:
        rate = rate / 100.0
    return float(rate)


def _build_warnings(snapshot: dict, method_used: str) -> list[str]:
    warnings: list[str] = []
    if method_used == "dividend_income":
        warnings.append("Dividend method is a minority scholarly approach and should be verified with a qualified scholar.")
    if not _has_numeric(snapshot.get("shares_outstanding")):
        warnings.append("Shares outstanding could not be verified; per-share zakatable assets may be estimated.")
    if not _has_numeric(snapshot.get("current_price")):
        warnings.append("Current market price is missing; market-value-based outputs may be approximate.")
    if not any(_has_numeric(snapshot.get(key)) for key in ("cash", "receivables", "inventory")):
        warnings.append("Balance-sheet data was incomplete; consider supplying financial_data manually for accuracy.")
    return warnings


def _infer_business_screen(sector: object, industry: object) -> bool:
    text = f"{sector or ''} {industry or ''}".lower()
    if not text.strip():
        return True
    return not any(keyword in text for keyword in PROHIBITED_BUSINESS_KEYWORDS)


def _is_non_compliant(overall_status: str, business_screen: bool, snapshot: dict) -> bool:
    if not business_screen:
        return True
    if overall_status in {"HARAM", "NON_COMPLIANT", "NON-COMPLIANT"}:
        return True
    haram_reason = str(snapshot.get("haram_reason", "")).lower()
    return any(keyword in haram_reason for keyword in PROHIBITED_BUSINESS_KEYWORDS)


def _non_compliant_note(detail: dict) -> str:
    if str(detail.get("haram_reason", "")).strip():
        return (
            f"{detail.get('ticker')} is non-compliant. {detail.get('haram_reason')} "
            "Scholars differ on zakat treatment for haram investments; the safest view is to liquidate and give the proceeds to charity rather than count them as zakat."
        )
    return (
        f"{detail.get('ticker')} is marked non-compliant or outside the halal screen. "
        "Scholars differ on zakat treatment for haram investments; the safest view is to liquidate and give the proceeds to charity rather than count them as zakat."
    )


def _sanitize_source_holding(holding: dict) -> dict:
    allowed = {
        "ticker",
        "shares",
        "shares_held",
        "purchase_price",
        "current_price",
        "company_name",
        "sector",
    }
    return {key: holding[key] for key in holding.keys() if key in allowed}


def _coerce_holdings(holdings: list[dict] | dict) -> list[dict]:
    if isinstance(holdings, dict):
        return [{"ticker": ticker, "shares": shares} for ticker, shares in holdings.items()]

    normalized: list[dict] = []
    for item in holdings or []:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        shares = item.get("shares", item.get("shares_held", 0))
        if ticker:
            normalized.append({**item, "ticker": ticker, "shares": shares})
    return normalized


def _guess_current_price(ticker: str) -> float:
    snapshot = _resolve_financial_snapshot(ticker, None)
    current = _first_numeric(snapshot.get("current_price"))
    if np.isfinite(current):
        return float(current)
    return np.nan


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


def _statement_value(frame: pd.DataFrame, candidates: list[str]) -> float:
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
    for value in values:
        numeric = _to_float(value)
        if np.isfinite(numeric):
            return numeric
    return np.nan


def _has_numeric(value: object) -> bool:
    numeric = _to_float(value)
    return bool(np.isfinite(numeric))


def _booleanish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "halal", "compliant"}


def _safe_int(value: object) -> int:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return 0
    return int(numeric)


def _to_float(value: object) -> float:
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


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).strip().upper()
    if not value.endswith(".KA"):
        value = f"{value}.KA"
    return value


def _normalize_method(method: str) -> str:
    value = str(method).strip().lower()
    mapping = {
        "assets": "net_assets",
        "net_assets": "net_assets",
        "net assets": "net_assets",
        "market": "market_value",
        "market_value": "market_value",
        "market value": "market_value",
        "income": "dividend_income",
        "dividend": "dividend_income",
        "dividend_income": "dividend_income",
        "dividend income": "dividend_income",
    }
    return mapping.get(value, "net_assets")


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _format_pkr(value: object) -> str:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"PKR {numeric:,.2f}"


def _format_number(value: object) -> str:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return "N/A"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


def _escape_html(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
