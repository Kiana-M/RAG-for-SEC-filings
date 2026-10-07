import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from retrieve import chroma_where, matches, rrf, validate_plan

COMPANIES = {"AAPL": {}, "JPM": {}}


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["c", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert list(dict(rrf([["a", "b", "c"], ["c", "a", "d"]])))[:2] == ["a", "c"]
    assert fused["d"] < fused["b"]  # one rank-3 hit scores below one rank-2 hit


def test_filters():
    flt = {"ticker": "AAPL", "fiscal_year": [2024, 2025], "section": ["risk_factors"]}
    assert chroma_where({"ticker": "AAPL"}) == {"ticker": {"$eq": "AAPL"}}
    assert chroma_where(flt)["$and"][1] == {"fiscal_year": {"$in": [2024, 2025]}}
    assert matches({"ticker": "AAPL", "fiscal_year": 2025, "section": "risk_factors"}, flt)
    assert not matches({"ticker": "AAPL", "fiscal_year": 2023, "section": "risk_factors"}, flt)


def test_validate_plan():
    plan = validate_plan({"tickers": ["aapl"], "needs_comparison": False, "period_types": ["fy"],
                          "sub_queries": [{"ticker": "AAPL", "query": "Apple risks"}]}, COMPANIES)
    assert plan["tickers"] == ["AAPL"] and plan["period_types"] == ["FY"]
    with pytest.raises(ValueError):
        validate_plan({"tickers": ["MRNA"], "needs_comparison": False, "sub_queries": []}, COMPANIES)
    # a ticker without a sub-query still gets one
    plan = validate_plan({"tickers": ["AAPL", "JPM"], "needs_comparison": True,
                          "sub_queries": [{"ticker": "AAPL", "query": "q"}]}, COMPANIES)
    assert [s["ticker"] for s in plan["sub_queries"]] == ["AAPL", "JPM"]
