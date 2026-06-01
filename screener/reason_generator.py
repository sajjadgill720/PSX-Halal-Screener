"""Generate full plain-English Shariah verdict explanations.

This module turns the raw AAOIFI screening output into a readable report that
can be shown in the dashboard or stored alongside cached verdicts.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping

from screener.aaoifi import (
    BUSINESS_HARAM_SECTORS,
    DEBT_THRESHOLD,
    INTEREST_THRESHOLD,
    RECEIVABLES_THRESHOLD,
    SECURITIES_THRESHOLD,
)

HEADER_LINE = "=" * 40
SECTION_LINE = "-" * 14


def generate_full_reason(ticker: str, verdict: dict, financials: dict) -> str:
    """Generate a complete, readable Shariah verdict explanation."""

    verdict_data = _safe_mapping(verdict)
    financial_data = _safe_mapping(financials)
    combined = (financial_data, verdict_data)

    ticker_text = _non_empty_text(_lookup(combined, "ticker"), ticker).upper()
    company_name = _non_empty_text(
        _lookup(combined, "company_name", "name", "company_company_name", "financial_company_name"),
        ticker_text,
    )
    status = _upper_text(_lookup(combined, "overall_status"), default="UNKNOWN")
    sector = _non_empty_text(
        _lookup(combined, "sector", "sector_classified", "sector_raw", "company_sector_classified", "company_sector_raw"),
        "Unknown",
    )
    screened_at = _format_date(_lookup(combined, "screened_at", "calculated_at", "fetched_at", "company_last_updated"))
    fiscal_year = _non_empty_text(_lookup(combined, "fiscal_year", "financial_fiscal_year"), "Unknown")
    data_source = _non_empty_text(_lookup(combined, "data_source"), "Unknown")
    data_quality = _non_empty_text(_lookup(combined, "data_quality"), "Unknown")

    shariah_score = _coerce_int(_lookup(combined, "shariah_score"))
    business_screen = _screen_result(_lookup(combined, "business_screen"))
    debt_screen = _screen_result(_lookup(combined, "debt_screen"))
    interest_screen = _screen_result(_lookup(combined, "interest_screen"))
    securities_screen = _screen_result(_lookup(combined, "securities_screen"))
    receivables_screen = _screen_result(_lookup(combined, "receivables_screen"))
    purification_ratio = _coerce_float(_lookup(combined, "purification_ratio", "financial_purification_ratio"))

    business_reason = _non_empty_text(
        _lookup(combined, "business_reason", "haram_reason", "business_summary", "screening_explanation"),
        "",
    )
    debt_reason = _non_empty_text(_lookup(combined, "debt_reason", "debt_explanation", "debt_note"), "")
    interest_reason = _non_empty_text(_lookup(combined, "interest_reason", "interest_explanation", "interest_note"), "")
    securities_reason = _non_empty_text(_lookup(combined, "securities_reason", "securities_explanation", "securities_note"), "")
    receivables_reason = _non_empty_text(_lookup(combined, "receivables_reason", "receivables_explanation", "receivables_note"), "")
    halal_conditions = _non_empty_text(_lookup(combined, "halal_conditions"), "")
    ai_explanation = _non_empty_text(_lookup(combined, "ai_explanation", "explanation"), "")

    total_assets = _coerce_float(_lookup(combined, "total_assets", "financial_total_assets"))
    total_debt = _coerce_float(_lookup(combined, "total_debt", "financial_total_debt"))
    interest_income = _coerce_float(_lookup(combined, "interest_income", "financial_interest_income"))
    total_revenue = _coerce_float(_lookup(combined, "total_revenue", "financial_total_revenue"))
    accounts_receivable = _coerce_float(_lookup(combined, "accounts_receivable", "financial_accounts_receivable"))
    non_compliant_investments = _coerce_float(
        _lookup(combined, "non_compliant_investments", "financial_non_compliant_investments")
    )

    debt_ratio = _ratio_from_value(_coerce_float(_lookup(combined, "debt_ratio", "financial_debt_ratio")), total_debt, total_assets)
    interest_ratio = _ratio_from_value(
        _coerce_float(_lookup(combined, "interest_ratio", "financial_interest_ratio")), interest_income, total_revenue
    )
    securities_ratio = _ratio_from_value(
        _coerce_float(_lookup(combined, "securities_ratio", "financial_securities_ratio")),
        non_compliant_investments,
        total_assets,
    )
    receivables_ratio = _ratio_from_value(
        _coerce_float(_lookup(combined, "receivables_ratio", "financial_receivables_ratio")),
        accounts_receivable,
        total_assets,
    )

    business_points = 20 if business_screen == "PASS" else 0
    debt_points = _screen_points(debt_ratio, DEBT_THRESHOLD)
    interest_points = _screen_points(interest_ratio, INTEREST_THRESHOLD)
    securities_points = _screen_points(securities_ratio, SECURITIES_THRESHOLD)
    receivables_points = _screen_points(receivables_ratio, RECEIVABLES_THRESHOLD)

    if status == "HARAM" and business_screen == "FAIL":
        display_score = 0
    elif shariah_score is not None:
        display_score = shariah_score
    else:
        display_score = int(round(business_points + debt_points + interest_points + securities_points + receivables_points))

    lines: list[str] = [
        HEADER_LINE,
        f"{company_name} — {status} {'✅' if status == 'HALAL' else '❌' if status == 'HARAM' else '⚠'}",
        f"Shariah Score: {display_score}/100",
        f"Screened: {screened_at}",
        HEADER_LINE,
        "",
    ]

    lines.append(_verdict_lead(status, business_screen, debt_screen, interest_screen, securities_screen, receivables_screen))
    lines.append("")

    lines.extend(_business_section(ticker_text, sector, business_screen, business_reason, ai_explanation, status))

    if business_screen == "FAIL" or status == "HARAM":
        lines.extend(_haram_followup(company_name, sector, business_reason, status))
    else:
        lines.extend(_financial_sections(
            ticker_text,
            debt_screen,
            debt_ratio,
            total_debt,
            total_assets,
            debt_reason,
            interest_screen,
            interest_ratio,
            interest_income,
            total_revenue,
            interest_reason,
            securities_screen,
            securities_ratio,
            non_compliant_investments,
            total_assets,
            securities_reason,
            receivables_screen,
            receivables_ratio,
            accounts_receivable,
            total_assets,
            receivables_reason,
            purification_ratio,
            status,
        ))
        lines.extend(_score_breakdown(
            business_points,
            debt_points,
            interest_points,
            securities_points,
            receivables_points,
            display_score,
            debt_ratio,
            interest_ratio,
            securities_ratio,
            receivables_ratio,
        ))
        if status == "HALAL":
            lines.extend(_purification_section(purification_ratio, interest_ratio))

    lines.extend(_data_quality_section(data_source, data_quality, fiscal_year, screened_at, total_assets, total_debt, interest_income, total_revenue, accounts_receivable, non_compliant_investments, verdict_data, financial_data))
    if status == "DOUBTFUL":
        lines.extend(_doubtful_note())

    lines.extend(_disclaimer_section())
    lines.append(HEADER_LINE)
    return "\n".join(line.rstrip() for line in lines if line is not None).strip()


def _verdict_lead(
    status: str,
    business_screen: str,
    debt_screen: str,
    interest_screen: str,
    securities_screen: str,
    receivables_screen: str,
) -> str:
    if status == "HALAL":
        return "VERDICT: This stock passes all 5 AAOIFI Shariah screening criteria and is permissible to invest in."
    if status == "HARAM" and business_screen == "FAIL":
        return "VERDICT: This stock fails Shariah screening and is not permissible to invest in under AAOIFI standards."
    if status == "HARAM":
        failed = [name for name, result in (("debt", debt_screen), ("interest", interest_screen), ("securities", securities_screen), ("receivables", receivables_screen)) if result == "FAIL"]
        failed_text = ", ".join(failed) if failed else "one or more financial screens"
        return f"VERDICT: This stock fails Shariah screening because {failed_text} are outside AAOIFI limits."
    return "VERDICT: This stock passes the business screen but has borderline or incomplete financial data, so it should be treated as doubtful and reviewed carefully."


def _business_section(ticker: str, sector: str, business_screen: str, business_reason: str, ai_explanation: str, status: str) -> list[str]:
    lines = [
        f"{SECTION_LINE} BUSINESS ACTIVITY {'✅ PASSED' if business_screen == 'PASS' else '❌ FAILED' if business_screen == 'FAIL' else '⚠ UNKNOWN'} {SECTION_LINE}",
    ]

    if business_screen == "PASS":
        if business_reason:
            lines.append(_wrap_text(business_reason))
        else:
            lines.append(_wrap_text(f"{ticker} does not appear to be operating in a prohibited industry. Its primary business is { _sector_description(sector) }."))
    elif business_screen == "FAIL":
        if business_reason:
            lines.append(_wrap_text(business_reason))
        else:
            lines.append(_wrap_text(f"{ticker} is operating in a prohibited or clearly non-compliant business line. Primary business: {sector}."))
        if ai_explanation and ai_explanation != business_reason:
            lines.append("")
            lines.append(_wrap_text(ai_explanation))
    else:
        lines.append(_wrap_text("The business classification is not clear enough to make a confident ruling."))
    return lines + [""]


def _financial_sections(
    ticker: str,
    debt_screen: str,
    debt_ratio: float | None,
    debt_amount: float | None,
    asset_amount: float | None,
    debt_reason: str,
    interest_screen: str,
    interest_ratio: float | None,
    interest_amount: float | None,
    revenue_amount: float | None,
    interest_reason: str,
    securities_screen: str,
    securities_ratio: float | None,
    securities_amount: float | None,
    securities_base: float | None,
    securities_reason: str,
    receivables_screen: str,
    receivables_ratio: float | None,
    receivables_amount: float | None,
    receivables_base: float | None,
    receivables_reason: str,
    purification_ratio: float | None,
    status: str,
) -> list[str]:
    lines: list[str] = []

    lines.append(
        f"{SECTION_LINE} DEBT RATIO {'✅ PASSED' if debt_screen == 'PASS' else '❌ FAILED' if debt_screen == 'FAIL' else '⚠ UNKNOWN'} {SECTION_LINE}"
    )
    lines.append(_debt_text(ticker, debt_screen, debt_ratio, debt_amount, asset_amount, debt_reason))
    lines.append("")

    lines.append(
        f"{SECTION_LINE} INTEREST INCOME {'✅ PASSED' if interest_screen == 'PASS' else '❌ FAILED' if interest_screen == 'FAIL' else '⚠ UNKNOWN'} {SECTION_LINE}"
    )
    lines.append(_interest_text(ticker, interest_screen, interest_ratio, interest_amount, revenue_amount, interest_reason, status, purification_ratio))
    lines.append("")

    lines.append(
        f"{SECTION_LINE} INTEREST-BEARING SECURITIES {'✅ PASSED' if securities_screen == 'PASS' else '❌ FAILED' if securities_screen == 'FAIL' else '⚠ UNKNOWN'} {SECTION_LINE}"
    )
    lines.append(_securities_text(ticker, securities_screen, securities_ratio, securities_amount, securities_base, securities_reason))
    lines.append("")

    lines.append(
        f"{SECTION_LINE} RECEIVABLES RATIO {'✅ PASSED' if receivables_screen == 'PASS' else '❌ FAILED' if receivables_screen == 'FAIL' else '⚠ UNKNOWN'} {SECTION_LINE}"
    )
    lines.append(_receivables_text(ticker, receivables_screen, receivables_ratio, receivables_amount, receivables_base, receivables_reason))
    lines.append("")

    return lines


def _debt_text(ticker: str, status: str, ratio: float | None, debt_amount: float | None, asset_amount: float | None, reason: str) -> str:
    if ratio is None:
        return _wrap_text(reason or f"{ticker}'s debt ratio could not be calculated from the available data.")
    return _build_ratio_paragraph(
        ticker=ticker,
        label="Debt ratio",
        status=status,
        ratio=ratio,
        threshold=DEBT_THRESHOLD,
        numerator_label="total debt",
        denominator_label="total assets",
        numerator_value=debt_amount,
        denominator_value=asset_amount,
        pass_message="This is comfortably below the AAOIFI threshold of 33%. The company is not excessively funded through interest-based borrowing.",
        fail_message="This exceeds the AAOIFI threshold of 33%, which indicates an excessive reliance on interest-based borrowing.",
        extra_reason=reason,
    )


def _interest_text(ticker: str, status: str, ratio: float | None, interest_amount: float | None, revenue_amount: float | None, reason: str, overall_status: str, purification_ratio: float | None) -> str:
    text = _build_ratio_paragraph(
        ticker=ticker,
        label="Interest income",
        status=status,
        ratio=ratio,
        threshold=INTEREST_THRESHOLD,
        numerator_label="interest income",
        denominator_label="total revenue",
        numerator_value=interest_amount,
        denominator_value=revenue_amount,
        pass_message="This is well within the 5% AAOIFI tolerance for incidental interest earnings.",
        fail_message="This exceeds the 5% AAOIFI tolerance for incidental interest earnings.",
        extra_reason=reason,
    )
    if overall_status == "HALAL" and purification_ratio is not None:
        text += f"\n{_wrap_text(f'Purification required: Donate {_format_percent(purification_ratio)} of any dividends received to charity.')}"
    return text


def _securities_text(ticker: str, status: str, ratio: float | None, securities_amount: float | None, securities_base: float | None, reason: str) -> str:
    return _build_ratio_paragraph(
        ticker=ticker,
        label="Interest-bearing securities",
        status=status,
        ratio=ratio,
        threshold=SECURITIES_THRESHOLD,
        numerator_label="non-compliant investments",
        denominator_label="total assets",
        numerator_value=securities_amount,
        denominator_value=securities_base,
        pass_message="This is well below the AAOIFI threshold of 33%.",
        fail_message="This exceeds the AAOIFI threshold of 33%, so too much of the asset base is in non-compliant investments.",
        extra_reason=reason,
    )


def _receivables_text(ticker: str, status: str, ratio: float | None, receivables_amount: float | None, receivables_base: float | None, reason: str) -> str:
    return _build_ratio_paragraph(
        ticker=ticker,
        label="Receivables ratio",
        status=status,
        ratio=ratio,
        threshold=RECEIVABLES_THRESHOLD,
        numerator_label="accounts receivable",
        denominator_label="total assets",
        numerator_value=receivables_amount,
        denominator_value=receivables_base,
        pass_message="The company's assets are not excessively illiquid or concentrated in receivables.",
        fail_message="This exceeds the AAOIFI threshold, suggesting too much of the asset base is tied up in receivables.",
        extra_reason=reason,
    )


def _build_ratio_paragraph(
    *,
    ticker: str,
    label: str,
    status: str,
    ratio: float | None,
    threshold: float,
    numerator_label: str,
    denominator_label: str,
    numerator_value: float | None,
    denominator_value: float | None,
    pass_message: str,
    fail_message: str,
    extra_reason: str,
) -> str:
    if ratio is None:
        return _wrap_text(extra_reason or f"{ticker}'s {label.lower()} could not be calculated from the available data.")

    ratio_pct = _format_percent(ratio)
    limit_pct = _format_percent(threshold)
    points = _screen_points(ratio, threshold)

    if status == "PASS":
        margin = max(0.0, (threshold - ratio) * 100)
        amount_sentence = _ratio_amount_sentence(ticker, numerator_label, denominator_label, numerator_value, denominator_value)
        return _wrap_text(
            f"{amount_sentence} This is {ratio_pct}, which is comfortably below the AAOIFI threshold of {limit_pct}. {pass_message} Margin of safety: {margin:.1f} percentage points below the limit."
        )

    if status == "FAIL":
        amount_sentence = _ratio_amount_sentence(ticker, numerator_label, denominator_label, numerator_value, denominator_value)
        excess = max(0.0, (ratio - threshold) * 100)
        return _wrap_text(
            f"{amount_sentence} This is {ratio_pct}, which is above the AAOIFI threshold of {limit_pct}. {fail_message} Excess over limit: {excess:.1f} percentage points."
        )

    return _wrap_text(
        extra_reason or f"{ticker}'s {label.lower()} is approximately {ratio_pct}, worth about {points}/20 points under the AAOIFI scoring model."
    )


def _ratio_amount_sentence(
    ticker: str,
    numerator_label: str,
    denominator_label: str,
    numerator_value: float | None,
    denominator_value: float | None,
) -> str:
    if numerator_value is None or denominator_value is None:
        return f"{ticker}'s {numerator_label} represents part of its {denominator_label}."
    return f"{ticker}'s {numerator_label} is {_format_currency_pkr(numerator_value)} against {_format_currency_pkr(denominator_value)} of {denominator_label}."


def _haram_followup(company_name: str, sector: str, business_reason: str, status: str) -> list[str]:
    lines = [
        f"{SECTION_LINE} REMAINING SCREENS: NOT EVALUATED {SECTION_LINE}",
        _wrap_text("Once the business activity screen fails, further ratio analysis is not meaningful. The core business model itself is impermissible."),
        "",
    ]
    if business_reason:
        lines.append(_wrap_text(business_reason))
        lines.append("")
    if status == "HARAM":
        lines.extend(_what_you_should_do(company_name))
        lines.extend(_alternatives_section(sector))
    return lines


def _score_breakdown(
    business_points: int,
    debt_points: int,
    interest_points: int,
    securities_points: int,
    receivables_points: int,
    display_score: int,
    debt_ratio: float | None,
    interest_ratio: float | None,
    securities_ratio: float | None,
    receivables_ratio: float | None,
) -> list[str]:
    def _part(name: str, points: int, ratio: float | None, threshold: float) -> str:
        if ratio is None:
            return f"{name}: {points}/20 points (data unavailable)"
        ratio_pct = _format_percent(ratio)
        if name == "Business Activity":
            note = "permissible" if points else "not permissible"
        else:
            note = "comfortable" if ratio < threshold * 0.75 else "acceptable" if ratio < threshold else "over limit"
        return f"{name}: {points}/20 points ({ratio_pct} — {note})"

    return [
        f"{SECTION_LINE} SHARIAH SCORE BREAKDOWN {SECTION_LINE}",
        f"Business Activity:  {business_points}/20 points",
        _part("Debt Ratio", debt_points, debt_ratio, DEBT_THRESHOLD),
        _part("Interest Income", interest_points, interest_ratio, INTEREST_THRESHOLD),
        _part("Securities", securities_points, securities_ratio, SECURITIES_THRESHOLD),
        _part("Receivables", receivables_points, receivables_ratio, RECEIVABLES_THRESHOLD),
        f"Total: {display_score}/100",
        "",
    ]


def _purification_section(purification_ratio: float | None, interest_ratio: float | None) -> list[str]:
    ratio = purification_ratio if purification_ratio is not None else interest_ratio
    if ratio is None:
        return []

    example_dividend = 10000.0
    example_donation = example_dividend * ratio
    return [
        f"{SECTION_LINE} PURIFICATION {SECTION_LINE}",
        f"Rate: {_format_percent(ratio, places=3)}",
        f"On PKR 10,000 dividends -> donate {_format_currency_pkr(example_donation)} to charity",
        "",
    ]


def _data_quality_section(
    data_source: str,
    data_quality: str,
    fiscal_year: str,
    screened_at: str,
    total_assets: float | None,
    total_debt: float | None,
    interest_income: float | None,
    total_revenue: float | None,
    accounts_receivable: float | None,
    non_compliant_investments: float | None,
    verdict_data: Mapping[str, Any],
    financial_data: Mapping[str, Any],
) -> list[str]:
    fields = [total_assets, total_debt, interest_income, total_revenue, accounts_receivable, non_compliant_investments]
    available = sum(value is not None for value in fields)
    quality = "Complete" if available == 6 else f"Partial ({available}/6 financial fields available)"

    company_label = _non_empty_text(
        _lookup((verdict_data, financial_data), "company_name", "name", "company_company_name"),
        "Unknown",
    )

    return [
        f"{SECTION_LINE} DATA QUALITY {SECTION_LINE}",
        f"Company: {company_label}",
        f"Data source: {data_source}",
        f"Fiscal year: {fiscal_year}",
        f"Fetched: {screened_at}",
        f"Quality: {data_quality if data_quality and data_quality != 'Unknown' else quality}",
        "",
    ]


def _what_you_should_do(company_name: str) -> list[str]:
    return [
        f"{SECTION_LINE} WHAT YOU SHOULD DO {SECTION_LINE}",
        _wrap_text(
            f"If you currently hold {company_name} shares, consider exiting the position as soon as practical and consult a qualified Shariah scholar about purification and disposal of any profit portion."
        ),
        "",
    ]


def _alternatives_section(sector: str) -> list[str]:
    sector_text = sector.lower()
    if "bank" in sector_text:
        suggestions = "Shariah-compliant banking options such as Meezan Bank, Bank Islami, or Dubai Islamic Bank Pakistan."
    elif "insurance" in sector_text:
        suggestions = "Shariah-compliant takaful or Islamic finance options instead of conventional insurance."
    else:
        suggestions = "Shariah-compliant alternatives in the same industry, if available."
    return [
        f"{SECTION_LINE} HALAL ALTERNATIVES {SECTION_LINE}",
        _wrap_text(suggestions),
        "",
    ]


def _doubtful_note() -> list[str]:
    return [
        f"{SECTION_LINE} NOTE {SECTION_LINE}",
        _wrap_text("Because the company is classified as doubtful, you should review the underlying financial statements and consult a qualified Shariah scholar before investing."),
        "",
    ]


def _disclaimer_section() -> list[str]:
    return [
        f"{SECTION_LINE} DISCLAIMER {SECTION_LINE}",
        _wrap_text(
            "This screening is based on AAOIFI Standard No. 21. It is guidance only and does not constitute a fatwa. Consult a qualified Shariah scholar for a ruling."
        ),
        "",
    ]


def _screen_points(ratio: float | None, threshold: float) -> int:
    if ratio is None:
        return 0
    score = (1 - (ratio / threshold)) * 20
    return int(round(max(0.0, min(20.0, score))))


def _ratio_from_value(ratio: float | None, numerator: float | None, denominator: float | None) -> float | None:
    if ratio is not None:
        return ratio
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return None


def _format_percent(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}%"


def _format_currency_pkr(value: float | None) -> str:
    if value is None:
        return "PKR 0"
    amount = float(value)
    abs_amount = abs(amount)
    if abs_amount >= 1_000_000_000:
        text = f"PKR {amount / 1_000_000_000:.1f} billion"
    elif abs_amount >= 1_000_000:
        text = f"PKR {amount / 1_000_000:.1f} million"
    elif abs_amount >= 1_000:
        text = f"PKR {amount:,.0f}"
    else:
        text = f"PKR {amount:.2f}"
    return text.replace(".0 billion", " billion").replace(".0 million", " million")


def _wrap_text(text: str) -> str:
    return " ".join(str(text).split())


def _format_date(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%B %d, %Y")
    except Exception:
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, pattern).strftime("%B %d, %Y")
            except Exception:
                continue
    return text


def _safe_mapping(value: Mapping[str, Any] | dict | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return dict(value)


def _lookup(sources: tuple[Mapping[str, Any], ...], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            if key in source:
                value = source.get(key)
                if not _is_missing(value):
                    return value
    return None


def _non_empty_text(value: Any, default: str) -> str:
    if _is_missing(value):
        return default
    text = str(value).strip()
    return text if text else default


def _upper_text(value: Any, default: str = "") -> str:
    text = _non_empty_text(value, default)
    return text.upper()


def _coerce_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            value = float(text)
        else:
            value = float(value)
        return None if math.isnan(value) else value
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    try:
        return int(round(number))
    except Exception:
        return None


def _screen_result(value: Any) -> str:
    text = _non_empty_text(value, "UNKNOWN").upper()
    if text in {"PASS", "FAIL", "UNKNOWN"}:
        return text
    if text in {"HALAL", "HARAM", "DOUBTFUL"}:
        return text
    return text


def _sector_description(sector: str) -> str:
    normalized = sector.strip().lower()
    if not normalized or normalized == "unknown":
        return "an unclassified business"
    if normalized in {item.lower() for item in BUSINESS_HARAM_SECTORS}:
        return f"a prohibited sector: {sector}"
    return sector


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null"}
    try:
        return math.isnan(float(value))
    except Exception:
        return False
