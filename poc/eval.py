"""Build the evaluation set, run it through answer(), write results.csv and a summary by question type.

python eval.py                     # build eval_set.json if missing, run everything
python eval.py --limit 1           # first question of each type (cheap smoke test)
python eval.py --types synthetic   # one type only
python eval.py --no-judge          # skip the faithfulness judge
python eval.py --rebuild-set       # regenerate eval_set.json

Question types: example (3), template (10), synthetic (10), unanswerable (3).
"""
import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict

import config
from answer import answer, extract_citations
from coverage import fully_covered, load_coverage
from ingest import get_collection, ntok
from llm import llm
from retrieve import bm25_index

EVAL_SET = config.ROOT / "eval_set.json"
RESULTS = config.ROOT / "results.csv"
SUMMARY = config.ROOT / "eval_summary.csv"
SEED = 7

EXAMPLES = [
    ("What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
     ["AAPL", "TSLA", "JPM"], ["risk_factors"], [], None, True),
    ("How has NVIDIA's revenue and growth outlook changed over the last two years?",
     ["NVDA"], ["mdna"], [], 2, True),
    ("What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
     ["ABBV", "JNJ", "LLY", "MRK", "PFE"], ["risk_factors"], [], None, True),
]


def q(qtype, question, tickers, sections=(), years=(), last_n=None, comparison=None, **extra):
    return {"type": qtype, "question": question, "expected_tickers": list(tickers),
            "expected_sections": list(sections), "expected_years": list(years),
            "expected_last_n": last_n, "expected_comparison": comparison, **extra}


# ---------------------------------------------------------------- build the set

def template_questions(companies, cov, rng):
    name = {t: c["aliases"][0] if c["aliases"] else c["name"] for t, c in companies.items()}
    with_10k = sorted({r["ticker"] for r in cov if r["form"] == "10-K"})
    full = fully_covered(cov)
    out = []
    # multi-company comparisons (different sectors)
    for template, section in [("What are the main risk factors for {a} and {b}, and how do they differ?", "risk_factors"),
                              ("Compare the legal proceedings disclosed by {a} and {b}.", "legal"),
                              ("How do {a} and {b} describe their competitive risks?", "risk_factors")]:
        while True:
            a, b = rng.sample(with_10k, 2)
            if companies[a]["sector"] != companies[b]["sector"]:
                break
        out.append(q("template", template.format(a=name[a], b=name[b]), [a, b], [section], comparison=True,
                     template=template))
    # industry phrase -> tickers
    industries = defaultdict(list)
    for t, c in companies.items():
        if t in with_10k:
            industries[c["industry"]].append(t)
    ind = rng.choice(sorted(i for i, ts in industries.items() if len(ts) >= 3))
    out.append(q("template", f"What risks do the {ind.lower()} companies in the corpus highlight?",
                 sorted(industries[ind]), ["risk_factors"], comparison=True, template="industry risks"))
    # temporal, fully covered companies only
    for t in rng.sample(full, 4)[:3]:
        y1, y2 = 2023, 2025
        out.append(q("template", f"How did {name[t]}'s revenue change between fiscal {y1} and fiscal {y2}?",
                     [t], ["mdna"], years=[y1, y2], comparison=True, template="revenue change years"))
    t = rng.choice(full)
    out.append(q("template", f"How has {name[t]}'s operating margin evolved over the last two years?",
                 [t], ["mdna"], last_n=2, comparison=True, template="margin last two years"))
    # single company, single section
    for template, section in [("What cybersecurity risks does {a} describe in its latest annual report?", "risk_factors"),
                              ("What legal proceedings is {a} involved in according to its latest 10-K?", "legal")]:
        a = rng.choice(with_10k)
        out.append(q("template", template.format(a=name[a]), [a], [section], comparison=False, template=template))
    return out


def synthetic_questions(n, rng):
    """Sample prose chunks and have the LLM write a paraphrased question each one answers."""
    idx = bm25_index()
    pool = [i for i, m in enumerate(idx["metadatas"]) if m["section"] in ("risk_factors", "mdna", "business", "legal")]
    rng.shuffle(pool)
    col = get_collection()
    out, seen = [], set()
    for i in pool:
        if len(out) == n:
            break
        cid, m = idx["ids"][i], idx["metadatas"][i]
        if m["ticker"] in seen:
            continue
        text = col.get(ids=[cid], include=["documents"])["documents"][0]
        lines = text.split("\n")
        if ntok(text) < 300 or sum(" | " in ln for ln in lines) > len(lines) * 0.3:
            continue  # prefer prose chunks
        prompt = (f"Excerpt from {m['company']}'s {m['form']} for {m['fiscal_label']} (section: {m['section']}):\n\n"
                  f"{text}\n\nWrite ONE question an analyst might ask that this excerpt answers specifically. "
                  f"Name the company and the period (e.g. 'in its {m['fiscal_label']} {m['form']}'). Paraphrase: "
                  'do not copy distinctive phrases. Return JSON {"question": "..."}')
        try:
            question = json.loads(llm(prompt, json_mode=True))["question"]
        except (ValueError, KeyError) as e:
            print(f"  skipped {cid}: {e}")
            continue
        seen.add(m["ticker"])
        out.append(q("synthetic", question, [m["ticker"]], [m["section"]], gold_chunk=cid))
        print(f"  synthetic: {cid} -> {question}")
    return out


def unanswerable_questions(companies, cov, rng):
    """Two-year changes for companies with a single 10-K, about years that filing cannot cover."""
    by = defaultdict(list)
    for r in cov:
        by[r["ticker"]].append(r)
    singles = [t for t, rs in by.items() if len(rs) == 1 and rs[0]["form"] == "10-K"]
    out, sectors = [], set()
    for t in rng.sample(singles, len(singles)):
        if companies[t]["sector"] in sectors:
            continue
        sectors.add(companies[t]["sector"])
        y = by[t][0]["fiscal_year"]
        out.append(q("unanswerable", f"How did {companies[t]['name']}'s revenue change between fiscal {y - 4} and fiscal {y - 3}?",
                     [t], ["mdna"], years=[y - 4, y - 3], comparison=True))
        if len(out) == 3:
            break
    return out


def build_set():
    rng = random.Random(SEED)
    companies = json.loads(config.COMPANIES.read_text())
    cov = [r for r in load_coverage() if r["status"] == "ok"]
    items = [q("example", qq, t, s, y, n, c) for qq, t, s, y, n, c in EXAMPLES]
    items += template_questions(companies, cov, rng)
    items += synthetic_questions(10, rng)
    items += unanswerable_questions(companies, cov, rng)
    for i, it in enumerate(items):
        it["id"] = f"{it['type'][:3]}-{i:02d}"
    EVAL_SET.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} questions -> {EVAL_SET.name}")
    return items


# ---------------------------------------------------------------- metrics

JUDGE_SYSTEM = """You check whether an answer is supported by the excerpts it cites. Return only JSON."""
ABSTAIN_RE = re.compile(r"(not (?:available|provided|included|contain|cover|answer)|no (?:data|information|filings?|excerpts?|relevant)"
                        r"|cannot|unable to|does not (?:contain|include|cover|provide|answer)|gaps?:(?! none))", re.I)


def planner_scores(item, plan):
    """Fraction of checked plan fields that match expectations (tickers, sections, years, comparison)."""
    checks = {"tickers": set(plan["tickers"]) == set(item["expected_tickers"])}
    if item["expected_sections"]:
        secs = set(plan["sections"]) | {s for sq in plan["sub_queries"] for s in sq["sections"]}
        checks["sections"] = not secs or set(item["expected_sections"]) <= secs
    if item["expected_years"]:
        years = set(plan["fiscal_years"]) | {y for sq in plan["sub_queries"] for y in sq["fiscal_years"]}
        checks["years"] = set(item["expected_years"]) <= years
    if item["expected_last_n"]:
        checks["years"] = plan["last_n_years"] == item["expected_last_n"]
    if item["expected_comparison"] is not None:
        checks["comparison"] = plan["needs_comparison"] == item["expected_comparison"]
    return sum(checks.values()) / len(checks), [k for k, v in checks.items() if not v]


def coverage_score(item, out):
    """Multi-company: share of expected tickers that are cited in the answer or named as a gap."""
    cited = {c.split("-")[0] for c in extract_citations(out["answer"])}
    gaps = " ".join(out["gaps"])
    hit = [t for t in item["expected_tickers"] if t in cited or t in gaps]
    return len(hit) / len(item["expected_tickers"])


def faithfulness(out):
    """LLM judge: share of factual claims in the answer supported by the cited excerpts (0-1)."""
    if not out["citations"]:
        return None, "no citations"
    excerpts = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in out["citations"])
    prompt = (f"Answer:\n{out['answer']}\n\nCited excerpts:\n{excerpts}\n\n"
              "List the factual claims in the answer (ignore statements about missing data or gaps). For each, "
              "decide if the cited excerpts support it. Return JSON "
              '{"supported": int, "total": int, "unsupported_claims": [str]}')
    res = json.loads(llm(prompt, system=JUDGE_SYSTEM, json_mode=True, judge=True))
    total = max(int(res.get("total", 0)), 1)
    return round(min(int(res.get("supported", 0)) / total, 1.0), 3), "; ".join(res.get("unsupported_claims", []))[:500]


def abstained(item, out):
    t = item["expected_tickers"][0]
    return any(t in g for g in out["gaps"]) or bool(ABSTAIN_RE.search(out["answer"]))


# ---------------------------------------------------------------- run

FIELDS = ["id", "type", "question", "expected_tickers", "plan_tickers", "recall_at_5", "company_coverage",
          "planner_accuracy", "planner_misses", "faithfulness", "unsupported", "abstained", "n_citations",
          "invalid_citations", "gaps", "latency_s", "error", "answer"]


def run_one(item, judge):
    row = {"id": item["id"], "type": item["type"], "question": item["question"],
           "expected_tickers": " ".join(item["expected_tickers"])}
    t0 = time.time()
    try:
        out = answer(item["question"])
    except Exception as e:  # quota exhaustion etc.: record and continue
        row.update(error=f"{type(e).__name__}: {e}"[:300], latency_s=round(time.time() - t0, 1))
        return row
    row.update(latency_s=round(time.time() - t0, 1), plan_tickers=" ".join(out["plan"]["tickers"]),
               n_citations=len(out["citations"]), invalid_citations=" ".join(out["invalid_citations"]),
               gaps=" | ".join(out["gaps"]), answer=out["answer"])
    if item["type"] == "synthetic":
        top5 = [c["id"] for c in out["retrieved"].get(item["expected_tickers"][0], [])[:5]]
        row["recall_at_5"] = int(item["gold_chunk"] in top5)
    if len(item["expected_tickers"]) > 1:
        row["company_coverage"] = round(coverage_score(item, out), 3)
    if item["type"] in ("template", "example"):
        acc, misses = planner_scores(item, out["plan"])
        row["planner_accuracy"], row["planner_misses"] = round(acc, 3), " ".join(misses)
    if item["type"] == "unanswerable":
        row["abstained"] = int(abstained(item, out))
    if judge and item["type"] != "unanswerable":
        try:
            row["faithfulness"], row["unsupported"] = faithfulness(out)
        except Exception as e:
            row["error"] = f"judge: {type(e).__name__}: {e}"[:300]
    return row


def mean(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    return round(sum(xs) / len(xs), 3) if xs else ""


def summarize(rows):
    metrics = ["recall_at_5", "company_coverage", "planner_accuracy", "faithfulness", "abstained", "latency_s"]
    table = []
    for t in ["example", "template", "synthetic", "unanswerable", "all"]:
        rs = [r for r in rows if t == "all" or r["type"] == t]
        if rs:
            table.append({"type": t, "n": len(rs), "errors": sum(bool(r.get("error")) for r in rs),
                          **{m: mean(r.get(m) for r in rs) for m in metrics}})
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    cols = list(table[0])
    print("\n" + " | ".join(f"{c:>16}" for c in cols))
    for r in table:
        print(" | ".join(f"{str(r[c]):>16}" for c in cols))
    print(f"\nrecall_at_5: synthetic gold chunk in the company's top 5 | company_coverage: expected companies cited "
          f"or named as gaps | planner_accuracy: share of plan fields correct | faithfulness: LLM judge | "
          f"abstained: unanswerable questions answered with a gap/refusal\n-> {RESULTS.name}, {SUMMARY.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-set", action="store_true")
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="questions per type")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    items = build_set() if args.rebuild_set or not EVAL_SET.exists() else json.loads(EVAL_SET.read_text())
    if args.types:
        items = [it for it in items if it["type"] in args.types]
    if args.limit:
        by_type = defaultdict(list)
        for it in items:
            by_type[it["type"]].append(it)
        items = [it for its in by_type.values() for it in its[:args.limit]]

    rows = []
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for it in items:
            print(f"[{it['id']}] {it['question']}", flush=True)
            row = run_one(it, judge=not args.no_judge)
            print(f"   -> " + ", ".join(f"{k}={row[k]}" for k in ("recall_at_5", "company_coverage", "planner_accuracy",
                                                                    "faithfulness", "abstained", "latency_s", "error")
                                        if row.get(k) not in (None, "")), flush=True)
            w.writerow(row)
            f.flush()
            rows.append(row)
    summarize(rows)


if __name__ == "__main__":
    main()
