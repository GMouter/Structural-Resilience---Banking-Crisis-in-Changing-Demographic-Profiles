from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "world_bank"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COVERAGE_DIR = PROJECT_ROOT / "outputs" / "coverage"
RAW_FILE = RAW_DIR / "world_bank_raw.csv"
PANEL_FILE = PROCESSED_DIR / "structural_panel.csv"
DERIVED_PANEL_FILE = PROCESSED_DIR / "structural_panel_derived.csv"
PHASE3_PANEL_FILE = PROCESSED_DIR / "structural_panel_phase3.csv"
PHASE3_WB_RAW_FILE = RAW_DIR / "world_bank_phase3.csv"
PWT_DIR = PROJECT_ROOT / "data" / "raw" / "pwt"
PWT_FILE = PWT_DIR / "pwt100.xlsx"
IMF_DIR = PROJECT_ROOT / "data" / "raw" / "imf"
BIS_DIR = PROJECT_ROOT / "data" / "raw" / "bis"
IMF_FISCAL_FILE = PROCESSED_DIR / "imf_fiscal.csv"
BIS_FINANCIAL_FILE = PROCESSED_DIR / "bis_financial.csv"
STRUCTURAL_FINANCIAL_PANEL_FILE = PROCESSED_DIR / "structural_financial_panel.csv"

START_YEAR = 1960
END_YEAR = 2025

COUNTRIES = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "CAN": "Canada",
    "AUS": "Australia",
    "NZL": "New Zealand",
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "ESP": "Spain",
    "NLD": "Netherlands",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "CHE": "Switzerland",
    "POL": "Poland",
    "JPN": "Japan",
    "KOR": "South Korea",
    "CHN": "China",
    "IND": "India",
    "IDN": "Indonesia",
    "SGP": "Singapore",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "CHL": "Chile",
    "ZAF": "South Africa",
    "NGA": "Nigeria",
    "SAU": "Saudi Arabia",
    "TUR": "Türkiye",
    "ISR": "Israel",
}

INDICATORS = {
    "working_age_share": {
        "code": "SP.POP.1564.TO.ZS",
        "name": "Population ages 15-64 (% total)",
    },
    "old_age_dependency": {
        "code": "SP.POP.DPND.OL",
        "name": "Age dependency ratio, old (% working-age population)",
    },
    "fertility": {
        "code": "SP.DYN.TFRT.IN",
        "name": "Fertility rate, total (births per woman)",
    },
    "population_growth": {
        "code": "SP.POP.GROW",
        "name": "Population growth (annual %)",
    },
    "real_gdp_pc_growth": {
        "code": "NY.GDP.PCAP.KD.ZG",
        "name": "GDP per capita growth (annual %)",
    },
    "real_gdp_growth": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "name": "GDP growth (annual %)",
    },
    "investment_gdp": {
        "code": "NE.GDI.TOTL.ZS",
        "name": "Gross capital formation (% GDP)",
    },
    "inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "name": "Inflation, consumer prices (annual %)",
    },
    "current_account_gdp": {
        "code": "BN.CAB.XOKA.GD.ZS",
        "name": "Current account balance (% GDP)",
    },
}

PANEL_COLUMNS = ["country_code", "country_name", "year", *INDICATORS]
