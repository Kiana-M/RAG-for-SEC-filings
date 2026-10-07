# RAG over SEC 10-K / 10-Q filings (proof of concept)

Cited answers over 246 EDGAR filings (54 companies) in `data/edgar_corpus/`.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

`.env` is git-ignored. Model names and providers are read from `.env` with defaults in `poc/config.py`:

- Embeddings: local `Alibaba-NLP/gte-modernbert-base` via sentence-transformers (downloaded on first run,
  runs on Apple MPS/CUDA/CPU; ~7 chunks/s on an M-series Mac, ~55 min for the full corpus).
  `EMBED_PROVIDER=gemini|openai` switches to an API (changing the model requires re-indexing: delete `poc/index/`).
- LLM (planner, answers, judge): `LLM_PROVIDER=gemini` (default, `gemini-flash-latest`, `GEMINI_API_KEY`) or
  `LLM_PROVIDER=anthropic` (default `claude-opus-5-5`, `ANTHROPIC_API_KEY`, depth via `CLAUDE_EFFORT`,
  server-side refusal fallback enabled) or `openai`. The judge has its own `JUDGE_PROVIDER` / `JUDGE_MODEL`,
  e.g. answer with Gemini and judge with Claude. All model calls go through `llm()` / `embed()` in `poc/llm.py`.

## Run

All commands run from `poc/`:

```bash
cd poc
../.venv/bin/python coverage.py               # coverage.csv + draft companies.json (hand edits are kept)
../.venv/bin/python ingest.py --sample --dry-run --show 5   # parse/chunk 5 files, print example chunks
../.venv/bin/python ingest.py --sample        # embed + index the 5-file sample
../.venv/bin/python ingest.py                 # index everything; logs chunks per file (idempotent: unchanged chunks are skipped)
../.venv/bin/python retrieve.py "How has NVIDIA's revenue and growth outlook changed over the last two years?"
../.venv/bin/python answer.py "What regulatory risks do the major pharmaceutical companies face?"
../.venv/bin/python eval.py --limit 1          # smoke test: one question per type
../.venv/bin/python eval.py                    # full evaluation -> results.csv, eval_summary.csv
../.venv/bin/streamlit run app.py              # demo at http://localhost:8501
../.venv/bin/python -m pytest -q tests
```

The index lives in `poc/index/` (Chroma + `bm25.pkl`) and is git-ignored.

## Data notes (verified against the corpus)

- Files are in `data/edgar_corpus/`; `manifest.json` lists filenames only (no per-file fields, no sector).
- 54 filings have no `Report Period` / `Quarter` header; the period end is taken from the URL filename
  (e.g. `xom-20251231.htm`) or the cover page.
- The `Quarter` header is the calendar quarter of the period end, not the fiscal quarter, so it is ignored.
  `fiscal_year` / `fiscal_quarter` are derived from the period end and each company's fiscal year end
  (`companies.json`), with the XBRL `dei` tag as a cross-check; the derived value wins on conflict.
  `fiscal_label` is the company's own naming (Home Depot and Target call the year ending Feb 2025 "fiscal 2024").
- Only 12 companies have fiscal Q1–Q3 10-Qs for every year 2023–2025:
  AAPL AMZN DIS GOOG JNJ KO MSFT NVDA PFE TSLA UNH XOM.
- `GE_10K_2015-02-27` (GE Capital, FY2014) is excluded from ingest; it stays in `coverage.csv` with a status.

## Section splitting

Item headings are matched anywhere in the text (they are often glued to the previous sentence) and
classified as real headings, table-of-contents rows (`Item 1A. | Risk Factors | 5`) or cross-references
(`see Part II, Item 7`). Filers without Item headings (banks, MCD, INTC) and stub items
("Item 7. Reference is made to ...") fall back to the section title itself; ExxonMobil's
"FINANCIAL SECTION" and statements placed after Item 16 are handled explicitly. Known gap: IBM's 10-K
incorporates its MD&A by reference, so no IBM chunk is labelled `mdna`.

## Retrieval

`retrieve.py`: one planner call (Gemini, JSON mode) maps the question to tickers (names, aliases, or
industry phrases like "major pharma" via `companies.json`), sections, period types and per-company
sub-queries; the plan is validated and retried once on error. "Last N years" is resolved in code from
each company's latest `period_end` in `coverage.csv`; with no time given, each company's latest fiscal
year is used. Each sub-query runs a Chroma search and a BM25 search under the same metadata filter
(top 20 each), fused with Reciprocal Rank Fusion (k=60); 5 chunks are kept per company (per company and
year for multi-year questions). Empty searches relax the section, then period filter; companies not in
the corpus and years without filings are returned as gaps. `RERANK=1` adds Cohere rerank.

Gemini free tier allows ~20 requests/day per model: `llm()` falls back through `LLM_FALLBACK_MODELS`
when a model is overloaded (503) or out of daily quota (429 with a long retry delay).

## Answering

`answer.py`: one LLM call per company summarizes only that company's retrieved chunks, citing chunk ids
in square brackets (or saying the excerpts do not answer the question); when the plan needs a comparison,
a final call answers the question from the summaries, keeps their citations and lists gaps. Citations are
checked against the retrieved set (`invalid_citations`). LLM responses are cached in `poc/cache/`
(`LLM_CACHE=0` disables), so repeated questions and eval re-runs cost no API calls.

## Evaluation

`eval.py` builds `eval_set.json` once (seeded; `--rebuild-set` to regenerate): the 3 example questions,
10 templated questions (multi-company comparisons, an industry phrase, temporal questions over the 12 fully
covered companies, single-section questions), 10 synthetic questions (random prose chunks, LLM-written
paraphrased question, chunk id kept as gold) and 3 unanswerable questions (revenue change for years a
single-10-K company's filing cannot cover). Metrics, per question in `results.csv` and by type in
`eval_summary.csv`:

- `recall_at_5` (synthetic): gold chunk among the company's top 5 retrieved chunks
- `company_coverage` (multi-company): expected companies cited in the answer or named as gaps
- `planner_accuracy` (template, example): share of checked plan fields correct (tickers, sections, years, comparison)
- `faithfulness` (LLM judge, `JUDGE_MODEL`): share of the answer's claims supported by its cited chunks
- `abstained` (unanswerable): the answer reports a gap or declines

A full run is ~110 LLM calls; on the Gemini free tier (20 requests/day/model) run it in parts
(`--types`, `--limit`) — cached calls are free on re-runs.
