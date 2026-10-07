"""Planner -> filtered hybrid search (Chroma + BM25) -> RRF -> per-company quotas.

python retrieve.py "How has NVIDIA's revenue and growth outlook changed over the last two years?"
"""
import argparse
import json
import pickle
from collections import defaultdict
from datetime import date

import numpy as np

import config
from coverage import load_coverage
from ingest import get_collection, tokenize
from llm import embed, llm

QUARTERS = ["FY", "Q1", "Q2", "Q3"]

PLANNER_SYSTEM = """You plan retrieval over a corpus of SEC 10-K and 10-Q filings. Return only JSON.

Companies in the corpus (ticker | name | aliases | sector / industry | fiscal year end | filings by fiscal year):
{companies}

Rules:
- Map company names, aliases and sector/industry phrases ONLY to tickers listed above. "Major pharma" or
  "pharmaceutical companies" means industry Pharmaceuticals; "banks" means industry Banking; etc.
  Companies the question names that are not in the list go in "unknown_companies" (never invent tickers).
- fiscal_years: ONLY when the question names specific years (e.g. "in 2023"), as the integer fiscal years
  shown above; otherwise [].
  If it says "last N years" / "recent" / "over time", set "last_n_years" to N (recent = 1, over time = 3)
  and leave fiscal_years empty: the system resolves it per company. If no time is given, leave both empty
  (the system then uses each company's latest filings).
- period_types: "FY" = 10-K, "Q1"/"Q2"/"Q3" = 10-Q. Risk factors, business and legal questions: ["FY"]
  (10-Q risk factor sections mostly refer back to the 10-K). Trends, growth, outlook: [] (all).
- sections: choose from {sections}. risk -> risk_factors; revenue, growth, outlook, results -> mdna and
  financials; regulation -> risk_factors, business, legal; litigation -> legal; leave [] if unsure.
- needs_comparison: true when the answer combines several companies or periods.
- sub_queries: one per ticker, a self-contained search query rewritten for that company
  (e.g. "NVIDIA revenue growth and outlook by segment"), with its own sections and fiscal_years
  (empty = use the top-level values).

JSON schema:
{{"tickers": [str], "unknown_companies": [str], "fiscal_years": [int], "last_n_years": int|null,
  "period_types": [str], "sections": [str], "needs_comparison": bool,
  "sub_queries": [{{"ticker": str, "query": str, "sections": [str], "fiscal_years": [int]}}]}}"""


# ---------------------------------------------------------------- planner

def coverage_by_ticker():
    rows = [r for r in load_coverage() if r["status"] == "ok"]
    by = defaultdict(list)
    for r in rows:
        by[r["ticker"]].append(r)
    return by


def company_table(companies, cov):
    lines = []
    for t, c in companies.items():
        if t not in cov:
            continue
        years = defaultdict(list)
        for r in sorted(cov[t], key=lambda r: r["period_end"]):
            years[r["fiscal_year"]].append(r["fiscal_quarter"])
        filings = "; ".join(f"FY{y}: {','.join(qs)}" for y, qs in sorted(years.items()))
        lines.append(f"{t} | {c['name']} | {', '.join(c['aliases'])} | {c['sector']} / {c['industry']} | "
                     f"FYE {c['fiscal_year_end']} | {filings}")
    return "\n".join(lines)


def validate_plan(plan, companies):
    """Raise ValueError with a readable message if the plan is malformed; normalize it otherwise."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    for key, typ in [("tickers", list), ("sub_queries", list), ("needs_comparison", bool)]:
        if not isinstance(plan.get(key), typ):
            raise ValueError(f"'{key}' must be a {typ.__name__}")
    plan["tickers"] = [t.upper() for t in plan["tickers"]]
    bad = [t for t in plan["tickers"] if t not in companies]
    if bad:
        raise ValueError(f"tickers not in the corpus: {bad}")
    plan["unknown_companies"] = list(plan.get("unknown_companies") or [])
    plan["fiscal_years"] = [int(y) for y in plan.get("fiscal_years") or []]
    n = plan.get("last_n_years")
    plan["last_n_years"] = int(n) if n else None
    plan["period_types"] = [p.upper() for p in plan.get("period_types") or []]
    if any(p not in QUARTERS for p in plan["period_types"]):
        raise ValueError(f"period_types must be from {QUARTERS}")
    plan["sections"] = list(plan.get("sections") or [])
    if any(s not in config.SECTIONS for s in plan["sections"]):
        raise ValueError(f"sections must be from {config.SECTIONS}")
    subs = []
    for sq in plan["sub_queries"]:
        t = str(sq.get("ticker", "")).upper()
        if t not in plan["tickers"] or not sq.get("query"):
            raise ValueError(f"bad sub_query {sq}: ticker must be one of 'tickers' and query non-empty")
        secs = list(sq.get("sections") or [])
        if any(s not in config.SECTIONS for s in secs):
            raise ValueError(f"sub_query sections must be from {config.SECTIONS}")
        subs.append({"ticker": t, "query": sq["query"], "sections": secs,
                     "fiscal_years": [int(y) for y in sq.get("fiscal_years") or []]})
    for t in plan["tickers"]:  # every ticker gets a sub-query
        if not any(s["ticker"] == t for s in subs):
            subs.append({"ticker": t, "query": None, "sections": [], "fiscal_years": []})
    plan["sub_queries"] = subs
    return plan


def plan_question(question, companies, cov):
    """One LLM call -> validated plan; on invalid JSON or schema, retry once with the error."""
    system = PLANNER_SYSTEM.format(companies=company_table(companies, cov), sections=config.SECTIONS)
    prompt = f"Question: {question}"  # no date: "last N years" is resolved from coverage, and the cache stays valid
    err = None
    for _ in range(2):
        text = llm(prompt if err is None else f"{prompt}\n\nYour previous plan was invalid: {err}. Fix it.",
                   system=system, json_mode=True)
        try:
            return validate_plan(json.loads(text), companies)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            err = str(e)
    raise ValueError(f"planner failed twice: {err}")


def resolve_years(plan, cov):
    """Fill each sub-query's fiscal_years: explicit years, else last_n_years from the company's latest filing."""
    for sq in plan["sub_queries"]:
        rows = cov.get(sq["ticker"], [])
        if not sq["fiscal_years"]:
            sq["fiscal_years"] = list(plan["fiscal_years"])
        if not sq["fiscal_years"] and plan["last_n_years"] and rows:
            latest = max(date.fromisoformat(r["period_end"]) for r in rows)
            cutoff = latest.replace(year=latest.year - plan["last_n_years"])
            sq["fiscal_years"] = sorted({r["fiscal_year"] for r in rows if date.fromisoformat(r["period_end"]) > cutoff})
        if not sq["fiscal_years"] and rows:  # no time given: the latest fiscal year with a matching filing
            typed = [r for r in rows if not plan["period_types"] or r["fiscal_quarter"] in plan["period_types"]] or rows
            sq["fiscal_years"] = [max(typed, key=lambda r: r["period_end"])["fiscal_year"]]
        if not sq["sections"]:
            sq["sections"] = list(plan["sections"])
    return plan


# ---------------------------------------------------------------- search

_bm25 = None


def bm25_index():
    global _bm25
    if _bm25 is None:
        with open(config.BM25_PATH, "rb") as f:
            _bm25 = pickle.load(f)
    return _bm25


def chroma_where(flt):
    """{'ticker': 'AAPL', 'fiscal_year': [2024, 2025], ...} -> Chroma where clause."""
    clauses = [{k: {"$in": v} if isinstance(v, list) else {"$eq": v}} for k, v in flt.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches(meta, flt):
    return all(meta[k] in v if isinstance(v, list) else meta[k] == v for k, v in flt.items())


def dense_search(col, qvec, flt, k):
    r = col.query(query_embeddings=[qvec], n_results=k, where=chroma_where(flt), include=[])
    return r["ids"][0]


def bm25_search(query, flt, k):
    idx = bm25_index()
    scores = idx["bm25"].get_scores(tokenize(query))
    allowed = np.array([matches(m, flt) for m in idx["metadatas"]])
    scores = np.where(allowed, scores, -np.inf)
    top = np.argsort(-scores)[:k]
    return [idx["ids"][i] for i in top if np.isfinite(scores[i]) and scores[i] > 0]


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def rerank(query, chunks):
    import cohere

    resp = cohere.ClientV2().rerank(model=config.RERANK_MODEL, query=query,
                                    documents=[c["text"] for c in chunks], top_n=len(chunks))
    return [chunks[r.index] | {"rerank": r.relevance_score} for r in resp.results]


def hybrid_search(col, query, flt, k=config.TOP_K):
    """Dense + BM25 under the same metadata filter, fused with RRF."""
    dense = dense_search(col, embed([query], task="query")[0], flt, k)
    sparse = bm25_search(query, flt, k)
    fused = rrf([dense, sparse], k=config.RRF_K)
    if not fused:
        return []
    ids = [i for i, _ in fused]
    got = col.get(ids=ids, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    out = []
    for i, s in fused:
        d, m = by_id[i]
        out.append({"id": i, "text": d, "meta": m, "rrf": round(s, 5),
                    "dense_rank": dense.index(i) + 1 if i in dense else None,
                    "bm25_rank": sparse.index(i) + 1 if i in sparse else None})
    return out


# ---------------------------------------------------------------- retrieve

def retrieve(question, plan=None):
    """Returns {"plan", "results": {ticker: [chunk]}, "gaps": [str], "notes": [str]}."""
    companies = json.loads(config.COMPANIES.read_text())
    cov = coverage_by_ticker()
    plan = resolve_years(plan or plan_question(question, companies, cov), cov)
    col = get_collection()
    gaps = [f"{c}: not in the corpus" for c in plan["unknown_companies"]]
    notes, results = [], {}

    for sq in plan["sub_queries"]:
        t = sq["ticker"]
        query = sq["query"] or f"{companies[t]['name']} {question}"
        have = cov.get(t, [])
        periods = plan["period_types"]
        # coverage gaps: requested years / periods with no filing
        for y in sq["fiscal_years"]:
            rows = [r for r in have if r["fiscal_year"] == y and (not periods or r["fiscal_quarter"] in periods)]
            if not rows:
                kinds = "/".join("10-K" if p == "FY" else f"{p} 10-Q" for p in periods) or "filing"
                gaps.append(f"{t}: no {kinds} for FY{y} in the corpus")
        years = [y for y in sq["fiscal_years"] if any(r["fiscal_year"] == y for r in have)]
        if sq["fiscal_years"] and not years:
            results[t] = []
            continue
        # temporal questions: search each year separately so every year gets its quota
        groups = [[y] for y in years] if len(years) > 1 else [years]
        chunks = []
        for ys in groups:
            base = {"ticker": t}
            if ys:
                base["fiscal_year"] = ys if len(ys) > 1 else ys[0]
            # relax filters progressively if nothing matches: sections, then period types
            attempts = [(sq["sections"], periods), ([], periods), ([], [])]
            for secs, pers in attempts:
                flt = dict(base)
                if pers:
                    flt["fiscal_quarter"] = pers
                if secs:
                    flt["section"] = secs
                found = hybrid_search(col, query, flt)
                if found:
                    if (secs, pers) != attempts[0] and (sq["sections"] or periods):
                        notes.append(f"{t} {ys or ''}: relaxed filters to sections={secs or 'any'}, periods={pers or 'any'}")
                    break
            if config.RERANK and found:
                found = rerank(query, found)
            chunks += found[:config.PER_TICKER]
        if not chunks:
            gaps.append(f"{t}: no matching chunks")
        results[t] = chunks
    return {"plan": plan, "results": results, "gaps": gaps, "notes": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--text", type=int, default=160, help="characters of chunk text to show")
    args = ap.parse_args()
    out = retrieve(args.question)
    print("PLAN\n" + json.dumps(out["plan"], indent=1))
    print("\nGAPS:", out["gaps"] or "none")
    if out["notes"]:
        print("NOTES:", out["notes"])
    for t, chunks in out["results"].items():
        print(f"\n== {t}: {len(chunks)} chunks")
        for c in chunks:
            m = c["meta"]
            print(f"  {c['id']:42} {m['fiscal_label']:10} rrf={c['rrf']:.4f} dense#{c['dense_rank']} bm25#{c['bm25_rank']}")
            if args.text:
                print("     " + c["text"][:args.text].replace("\n", " ") + " ...")


if __name__ == "__main__":
    main()
