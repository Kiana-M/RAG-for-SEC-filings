"""Per-company summaries with citations, then a comparison answer.

python answer.py "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import config
from llm import llm
from retrieve import retrieve

CHUNK_ID_RE = re.compile(r"[A-Z][A-Z.]{0,5}-10-[KQ]-\d{4}-(?:FY|Q[1-4])-[a-z_]+-\d+")

SUMMARY_SYSTEM = """You answer questions about SEC filings using ONLY the excerpts provided.
- Every factual sentence ends with the id(s) of the excerpt(s) that support it in square brackets,
  exactly as given, e.g. [AAPL-10-K-2025-FY-risk_factors-2].
- Refer to periods by the fiscal label shown with each excerpt (e.g. "Q2 FY2025", "FY2024").
- Numbers in parentheses in tables are negative.
- Cover ONLY the company named below. The question may mention other companies or ask for a
  comparison: ignore that part (another step compares companies) and never mention other companies'
  missing excerpts.
- Do not use outside knowledge. If the excerpts do not answer the question for this company (or only
  partly), say so explicitly and state what is missing.
- Be concise: at most about 200 words, bullet points welcome."""

COMPARE_SYSTEM = """You write the final answer to a question about SEC filings from per-company summaries.
- Use ONLY facts in the summaries and keep their citations in square brackets exactly as given;
  do not invent new citations or facts.
- Answer the question directly, then compare the companies (similarities and differences).
- Refer to periods by fiscal label (e.g. "FY2025", "Q2 FY2025"); fiscal years differ between companies.
- End with a "Gaps" section listing every known gap given below and any company whose own summary says
  its excerpts did not (fully) answer the question for that company; write "Gaps: none" if there are none."""


def format_chunks(chunks):
    return "\n\n".join(
        f"[{c['id']}] ({c['meta']['company']}, {c['meta']['form']} {c['meta']['fiscal_label']}, "
        f"period ending {c['meta']['period_end']}, section {c['meta']['section']})\n{c['text']}"
        for c in chunks)


def summarize(question, ticker, chunks, company):
    if not chunks:
        return f"No relevant excerpts were found in {company}'s filings in the corpus."
    prompt = f"Question: {question}\n\nCompany: {company} ({ticker})\n\nExcerpts:\n\n{format_chunks(chunks)}\n\n" \
             f"Summarize what these excerpts say about the question for {company}, with citations."
    return llm(prompt, system=SUMMARY_SYSTEM)


def compare(question, summaries, gaps):
    body = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())
    gap_text = "\n".join(f"- {g}" for g in gaps) or "none"
    prompt = f"Question: {question}\n\nPer-company summaries:\n\n{body}\n\nKnown gaps:\n{gap_text}"
    return llm(prompt, system=COMPARE_SYSTEM)


def extract_citations(text):
    return list(dict.fromkeys(CHUNK_ID_RE.findall(text)))


def answer(question, plan=None):
    """Returns {"answer", "summaries", "citations": [chunk], "invalid_citations", "plan", "gaps", "notes"}."""
    companies = json.loads(config.COMPANIES.read_text())
    ret = retrieve(question, plan)
    results, gaps = ret["results"], list(ret["gaps"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {t: pool.submit(summarize, question, t, chunks, companies[t]["name"]) for t, chunks in results.items()}
        summaries = {t: f.result() for t, f in futures.items()}

    if ret["plan"]["needs_comparison"] and len(summaries) > 1:
        final = compare(question, summaries, gaps)
    elif summaries:
        final = "\n\n".join(summaries.values())
        if gaps:
            final += "\n\nGaps:\n" + "\n".join(f"- {g}" for g in gaps)
    else:  # nothing in the corpus matches the question
        final = "The corpus has no filings that can answer this question.\n\nGaps:\n" + \
                "\n".join(f"- {g}" for g in gaps or ["no matching companies"])

    retrieved = {c["id"]: c for chunks in results.values() for c in chunks}
    cited = extract_citations(final + "\n" + "\n".join(summaries.values()))
    return {
        "question": question,
        "answer": final,
        "summaries": summaries,
        "citations": [retrieved[i] for i in cited if i in retrieved],
        "invalid_citations": [i for i in cited if i not in retrieved],
        "retrieved": results,
        "plan": ret["plan"],
        "gaps": gaps,
        "notes": ret["notes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    args = ap.parse_args()
    out = answer(args.question)
    print(out["answer"])
    print("\n---\nPlan tickers:", out["plan"]["tickers"], "| gaps:", out["gaps"] or "none")
    print("Citations:")
    for c in out["citations"]:
        m = c["meta"]
        print(f"  [{c['id']}] {m['company']} {m['form']} {m['fiscal_label']} {m['section']}")
    if out["invalid_citations"]:
        print("Citations not in the retrieved set:", out["invalid_citations"])


if __name__ == "__main__":
    main()
