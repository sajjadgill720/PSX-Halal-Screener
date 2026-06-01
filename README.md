# PSX Halal Screener

PSX Halal Screener is a Streamlit application for screening Pakistan Stock Exchange equities using AAOIFI-style Shariah filters and estimating zakat on stock portfolios.

## Installation

```bash
git clone <your-repository-url>
cd "PSX Halal Screener"
pip install -r requirements.txt
# Recommended: use the provided virtual environment helpers to run Streamlit
".venv\Scripts\Activate.ps1"  # (Windows PowerShell)
python -m streamlit run dashboard/app.py

Or double-click run_streamlit.bat in the repo root (Windows) which runs Streamlit using the project venv.
```

## AAOIFI Screening Ratios

The screener applies the following AAOIFI-inspired checks:

- Business screen: excludes sectors that are not Shariah-compliant by default, such as conventional banking, conventional insurance, alcohol, tobacco, weapons, pornography, and gambling.
- Debt ratio: total debt / total assets must be below 0.33.
- Interest income ratio: interest income / total revenue must be below 0.05.
- Interest-bearing securities ratio: non-compliant investments / total assets must be below 0.33.
- Receivables ratio: accounts receivable / total assets must be below 0.49.

A company that passes all screens is treated as Halal. A company that passes the business screen but fails one or two financial screens is marked Doubtful. A company that fails the business screen or three or more financial screens is marked Haram.

## Disclaimer

This project provides screening guidance only and does not constitute a fatwa or formal Shariah ruling. Always consult a qualified Shariah scholar before making investment decisions.

## Credits

This project references AAOIFI Standard No. 21 (Financial Paper/Share) as a key source for Shariah screening principles.
