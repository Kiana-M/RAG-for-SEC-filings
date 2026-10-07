import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date

from coverage import fiscal_period, parse_header
from ingest import chunk_text, clean, find_headings, normalize_tables, ntok, split_sections

HEADER = "Company: Apple Inc\nTicker: AAPL\nFiling Type: 10-K (Annual Report)\n" + "=" * 60 + "\n"


def test_parse_header_and_xbrl_strip():
    raw = HEADER + "aapl-20240928false2024FY0000320193us-gaap:FooMember2024-01-01\nUNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nForm 10-K"
    meta, body = parse_header(raw)
    assert meta["Ticker"] == "AAPL" and meta["Filing Type"].startswith("10-K")
    text = clean(body)
    assert text.startswith("UNITED STATES") and "us-gaap" not in text


def test_fiscal_period_52_53_week_and_offset_years():
    assert fiscal_period(date(2024, 6, 29), "09-27", "10-Q") == (2024, "Q3")  # Apple fiscal Q3
    assert fiscal_period(date(2024, 4, 28), "01-26", "10-Q") == (2025, "Q1")  # NVIDIA FY ends late Jan
    assert fiscal_period(date(2024, 1, 28), "01-26", "10-K") == (2024, "FY")
    assert fiscal_period(date(2022, 4, 1), "12-31", "10-Q") == (2022, "Q1")  # KO header says 2022Q2


def test_running_headers_and_footers_stripped():
    body = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nItem 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL "
            "CONDITION AND RESULTS OF OPERATIONSRevenue grew.Apple Inc. | 2024 Form 10-K | 20More text 53Table of "
            "ContentsCISCO SYSTEMS, INC.MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF "
            "OPERATIONS (Continued)by comparing. 14NVIDIA Corporation and SubsidiariesNotes to Condensed Consolidated "
            "Financial Statements (Continued)(Unaudited)Note 9")
    text = clean(body)
    assert text.count("MANAGEMENT’S DISCUSSION") == 1  # the Item 7 title stays, the page-top repeat goes
    assert "Form 10-K | 20" not in text and "Table of Contents" not in text and "(Continued)" not in text


def test_normalize_tables():
    assert normalize_tables("Net sales |  | $ | 215 |  |  | $ | 322 |") == "Net sales | $215 | $322"
    assert normalize_tables("ITEM 1A. RISK FACTORS |  |  |") == "ITEM 1A. RISK FACTORS"
    assert normalize_tables("Margin | 46.2 | % |  | (1.5 | ) |") == "Margin | 46.2% | (1.5)"


FAKE_10K = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 10-K\nTABLE OF CONTENTS\n"
    "Item 1. | Business | 1\nItem 1A. | Risk Factors | 5\nItem 7. | Management’s Discussion | 20\n"
    "Item 8. | Financial Statements | 28\n" + "Cover text. " * 40
    + "\nItem 1. BusinessWe make phones. " + "Business text. " * 40
    + "only.Item 1A.    Risk FactorsThe Company faces risks, as discussed in Part II, Item 7 of this Form 10-K "
      "and see the Item 1A. Risk Factors section. " + "Risk text. " * 40
    + "| 20Item 7.    Management’s Discussion and AnalysisSales rose. " + "MD&A text. " * 40
    + "None.30FINANCIAL SECTION\nTABLE OF CONTENTS\nBusiness Profile | 32\n" + "Profile. " * 40
    + "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\nWe audited. " + "Statements. " * 40
)


def test_sections_skip_toc_and_cross_references():
    labels = [lab for lab, _ in split_sections(FAKE_10K, "10-K")]
    assert labels == ["other", "business", "risk_factors", "mdna", "financials"]
    risk = dict(split_sections(FAKE_10K, "10-K"))["risk_factors"]
    assert "Part II, Item 7 of this Form" in risk  # cross-reference did not start a new section


def test_10q_part2_items():
    text = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION " + "x. " * 100
            + "PART I — FINANCIAL INFORMATIONItem 1.    Financial StatementsBalance. " + "Num. " * 100
            + "Item 2.    Management’s Discussion and AnalysisGrew. " + "Talk. " * 100
            + "PART II — OTHER INFORMATIONItem 1.    Legal ProceedingsSued. " + "Law. " * 100
            + "Item 1A.    Risk FactorsRisky. " + "Risk. " * 100)
    assert [lab for _, lab in find_headings(text, "10-Q")] == ["financials", "mdna", "legal", "risk_factors"]


def test_chunking_size_overlap_and_tables():
    para = " ".join(f"Sentence number {i} talks about the U.S. market." for i in range(400))
    table = "\n".join(f"Row {i} | {i * 10} | {i * 20}" for i in range(10))
    chunks = chunk_text(para + "\n" + table, size=200, overlap=30)
    assert all(ntok(c) <= 200 for c in chunks)
    assert "U.S. market" in chunks[0]  # no split after "U.S."
    assert chunks[1].split(".")[0] in chunks[0]  # overlap carries the tail of the previous chunk
    assert any(table in c for c in chunks)  # a table that fits stays in one chunk
