"""Clean -> sections -> chunks -> Chroma (embeddings) + BM25 (pickle).

python ingest.py                 # all filings in coverage.csv with status ok
python ingest.py --sample        # 5-file sample
python ingest.py --dry-run ...   # parse and chunk only, no embeddings or index writes
python ingest.py --files X.txt   # specific files
"""
import argparse
import hashlib
import json
import pickle
import re
from collections import Counter

import tiktoken

import config
from coverage import load_coverage, parse_header

SAMPLE = [
    "AAPL_10K_2024Q3_2024-11-01_full.txt",
    "XOM_10K_2024Q4_2025-02-19_full.txt",
    "JPM_10Q_2025Q2_2025-08-05_full.txt",
    "PFE_10Q_2024Q2_2024-08-05_full.txt",
    "NVDA_10Q_2024Q2_2024-05-29_full.txt",
]

ENC = tiktoken.get_encoding("cl100k_base")  # tokenizer of text-embedding-3-*


def ntok(s):
    return len(ENC.encode(s, disallowed_special=()))


# ---------------------------------------------------------------- cleaning

SEC_START = re.compile(r"UNITED\s+STATES\s*SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION", re.I)
RUNNING_HEADERS = [
    r"MANAGEMENT[’']S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    r"NOTES TO (?:THE )?(?:CONDENSED )?CONSOLIDATED FINANCIAL STATEMENTS",
]
# optional all-caps company name ("ADOBE INC. "), the header, optional "(Continued)"
RUNNING_RE = re.compile(
    r"(?:[A-Z][A-Z&.,’'\- ]{1,40}?\s*)?(?:" + "|".join(RUNNING_HEADERS) + r")\s*(?:\((?:Continued|continued)\))?"
)
# Title-case page headers: "14NVIDIA Corporation and SubsidiariesNotes to Condensed Consolidated
# Financial Statements (Continued)(Unaudited)". The "(Continued)" marker makes these safe to drop.
CONTINUED_RE = re.compile(
    r"(?:[A-Z][A-Za-z.,&’' -]{2,60}?(?:and|AND) (?:Subsidiaries|SUBSIDIARIES)\s*)?"
    r"(?i:notes to (?:the )?(?:condensed )?consolidated financial statements|management[’']s discussion and analysis"
    r"(?: of financial condition and results of operations)?)\s*\((?i:continued)\)(?:\s*\((?i:unaudited)\))?"
)
ITEM_BEFORE = re.compile(r"(?i)item\s*\d{1,2}[a-c]?\s*[.:—–-]?\s*\Z")
PAGE_LINK_RE = re.compile(r"\s*\d{0,3}\s*Table of Contents")  # "53Table of Contents" page-break link
FOOTER_RE = re.compile(r"[A-Z][A-Za-z0-9.,&’' ]{1,40}\|\s*20\d\d Form 10-[KQ]\s*\|\s*\d{1,3}")  # "Apple Inc. | 2024 Form 10-K | 20"


def strip_running(text):
    """Drop page-top repeats of running headers; keep each header's first occurrence and any
    occurrence that is an Item title ("ITEM 7. MANAGEMENT'S DISCUSSION ...")."""
    seen = set()

    def repl(m):
        key = "MDA" if "DISCUSSION" in m.group(0) else "NOTES"
        is_item = "ITEM" in m.group(0).upper().split("MANAGEMENT")[0] or ITEM_BEFORE.search(text[max(0, m.start() - 20):m.start()])
        if is_item or key not in seen:
            seen.add(key)
            return m.group(0)
        return " "

    return RUNNING_RE.sub(repl, text)


def clean(body):
    """Drop the XBRL block and page furniture; keep line structure for tables."""
    m = SEC_START.search(body)
    text = body[m.start():] if m else body
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\u2009", " ")
    text = PAGE_LINK_RE.sub("\n", text)  # a page break is a line break
    text = FOOTER_RE.sub(" ", text)
    text = CONTINUED_RE.sub(" ", text)
    text = strip_running(text)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)  # bare page-number lines
    return text


def normalize_tables(text):
    """Collapse empty cells, glue '$' and '%' cells to their numbers. Single-cell rows become plain lines."""
    out = []
    for line in text.split("\n"):
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            merged = []
            for c in cells:
                if not c:
                    continue
                if merged and (merged[-1] in ("$", "(", "$(") or c in ("%", ")", ")%", "pts", "bps")):
                    merged[-1] = merged[-1] + c
                else:
                    merged.append(c)
            line = " | ".join(merged)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- sections

# "Item 1A." / "ITEM 7:" / "Item 2 —" followed by a capitalized title. No lookbehind: real headings are
# often glued to the previous text ("only.Item 1A", "| 17Item 2.", "PART IIItem 5", "INFORMATIONItem 1.").
ITEM_RE = re.compile(r"(?:ITEM|Item)\s*(\d{1,2}[A-C]?)(?![0-9(])\s*[.:—–-]?(?=[\s|]*[A-Z\[“\"])")
TOC_ROW_RE = re.compile(r"\|\s*(?:\d{1,3}|N/A)\s*$")
FIN_SECTION_RE = re.compile(r"FINANCIAL SECTION(?!\s*\|)")
AUDITOR_RE = re.compile(r"REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM")

K_LABELS = {"1": "business", "1A": "risk_factors", "3": "legal", "7": "mdna", "7A": "market_risk",
            "8": "financials", "9A": "controls", "15": "financials"}  # some 10-Ks (NVDA) put statements in Item 15
Q_PART1 = {"1": "financials", "2": "mdna", "3": "market_risk", "4": "controls"}
Q_PART2 = {"1": "legal", "1A": "risk_factors"}


def _caps_or_title(phrase):
    """Title Case, ALL CAPS or Sentence case (JPM: "Consolidated statements of income"); flexible spacing."""
    variants = (phrase, phrase.upper(), phrase[0] + phrase[1:].lower())
    return "(?:" + "|".join(v.replace(" ", r"\s*") for v in variants) + ")"


# Fallback for filers without "Item N" headings (MCD, INTC, MS...): the section title itself.
# Each label has titles in priority order; the first title with an acceptable match wins.
_AUDITOR = _caps_or_title("Report of Independent Registered Public Accounting Firm")
_FSSD = _caps_or_title("Financial Statements and Supplementary Data")
_INCOME = r"(?:(?:Condensed|CONDENSED)\s*)?" + _caps_or_title("Consolidated Statements? of (?:Income|Operations|Earnings)")
TITLE_RES = {
    "risk_factors": [_caps_or_title("Risk Factors")],
    "mdna": [_caps_or_title("Management(?:[’']s)? Discussion and Analysis(?: of Financial Conditions? and Results of Operations)?")],
    "market_risk": [_caps_or_title("Quantitative and Qualitative Disclosures? About Market Risk")],
    "legal": [_caps_or_title("Legal Proceedings")],
    "controls": [_caps_or_title("Controls and Procedures")],
    "financials": [_AUDITOR, _FSSD, _INCOME],  # 10-K: auditor's report precedes the statements
    "financials_10q": [_INCOME, _FSSD, _AUDITOR],  # 10-Q: review report (if any) follows them
}
# 10-Q items are labelled by their title when it is recognisable (BAC puts Item 2 before Item 1)
Q_TITLE_LABELS = [("Financial Statements", "financials"), ("Management", "mdna"), ("Quantitative", "market_risk"),
                  ("Controls", "controls"), ("Legal", "legal"), ("Risk Factors", "risk_factors")]
TITLE_RES = {k: [re.compile(v) for v in vs] for k, vs in TITLE_RES.items()}


def is_toc_row(text, end):
    """Heading text followed by '| <page>' at the end of its line, or, when the line is only a pipe
    (title wrapped: "Item 1. |\nFinancial Statements | 1"), at the end of the next line."""
    lines = text[end:end + 300].split("\n")
    if TOC_ROW_RE.search(lines[0]):
        return True
    return not re.sub(r"[\s|.]", "", lines[0]) and len(lines) > 1 and bool(TOC_ROW_RE.search(lines[1]))


def is_reference(text, start):
    """True for in-text cross-references ("see the Item 1A.", "Part II, Item 7", '"Risk Factors'),
    False for real headings, which are glued to the previous text ("InformationItem 1.") or follow a
    line/sentence end, a page number, a pipe, or an all-caps/stub title ("RESERVED ITEM 7.")."""
    before = text[max(0, start - 30):start]
    stripped = before.rstrip(" \t")
    if re.search(r"[\"“(,]\Z", stripped):
        return True  # quoted reference: 'see "Item 1A. Risk Factors'
    if stripped == before or not stripped:
        return False  # glued or at line start
    if re.search(r"[.;:|)\]\d\n”]\Z", stripped):
        return False
    return not re.search(r"(?:\b[A-Z]{2,}|PART\s+[IV]+|Reserved|None|Form 10-[KQ])\Z", stripped)


def _title_heading(text, regexes, skip_spans, head_positions):
    for rx in regexes:
        for m in rx.finditer(text, int(len(text) * 0.02)):
            if any(p - 50 <= m.start() < e for p, e in skip_spans):
                continue
            nxt = text[m.end():m.end() + 5].lstrip(" ")[:1]
            if not (nxt == "" or nxt.isupper() or nxt in "\n(|") or is_toc_row(text, m.end()) \
                    or is_reference(text, m.start()):
                continue
            if any(0 < p - m.start() < 1500 for p in head_positions):
                continue  # a stub ("Item 8. ... see page F-1") directly followed by the next Item
            return m.start()
    return None


def find_headings(text, form):
    """[(pos, label)] of real section starts. Skips TOC rows and in-text cross-references."""
    matches = list(ITEM_RE.finditer(text))
    # everything up to the last TOC row near the top is front matter (catches TOC rows without page numbers)
    toc_end = max((m.end() for m in matches if m.start() < len(text) * 0.1 and is_toc_row(text, m.end())), default=0)
    cands = [(m.start(), m.group(1).upper()) for m in matches
             if m.start() > toc_end and not is_toc_row(text, m.end()) and not is_reference(text, m.start())]

    def rank(item):
        return int(re.match(r"\d+", item).group()), item

    heads, part2 = [], False
    for pos, item in cands:
        if form == "10-K":
            label = K_LABELS.get(item, "other")
        else:  # 10-Q: label by title, else by number (numbering restarts in Part II)
            if item in ("5", "6") or (heads and rank(item) < rank(heads[-1][1])):
                part2 = True
            label = (Q_PART2 if part2 else Q_PART1).get(item, "other")
            title = text[pos:pos + 80].upper()
            if item in ("1", "1A", "2", "3", "4"):
                label = next((lab for t, lab in Q_TITLE_LABELS if t.upper() in title), label)
        heads.append((pos, item, label))

    tail16 = [p for p, item, _ in heads if item == "16"]
    if form == "10-K" and tail16 and len(text) - tail16[-1] > len(text) * 0.05 and not FIN_SECTION_RE.search(text):
        # NFLX and others: statements follow Item 16 ("INDEX TO CONSOLIDATED FINANCIAL STATEMENTS")
        rx = re.compile(_caps_or_title("Index to Consolidated Financial Statements") + "|" + _AUDITOR)
        m = next((m for m in rx.finditer(text, tail16[-1])
                  if not is_toc_row(text, m.end()) and not is_reference(text, m.start())), None)
        if m:
            heads.append((m.start(), "after16", "financials"))

    if form == "10-K":  # ExxonMobil: real MD&A and statements after Item 16 under "FINANCIAL SECTION"
        fs = [m.start() for m in FIN_SECTION_RE.finditer(text) if m.start() > len(text) * 0.05]
        if fs:
            heads.append((fs[0], "FS", "mdna"))
            aud = AUDITOR_RE.search(text, fs[0])
            if aud:
                heads.append((aud.start(), "FS-auditor", "financials"))

    # Title fallback for sections the Item headings missed, or found only as a stub
    # ("Item 7. ... Reference is made to ..."): first title-style heading outside the stub
    # that starts a real section (no other heading within 1500 chars).
    heads.sort()
    ends = [h[0] for h in heads[1:]] + [len(text)]
    for label in ("risk_factors", "mdna", "market_risk", "legal", "controls", "financials"):
        spans = [(p, e) for (p, _, lab), e in zip(heads, ends) if lab == label]
        if form == "10-Q" and label == "risk_factors" and not spans:
            continue  # 10-Q risk factors are optional; a title match would be a reference to the 10-K
        # stub threshold: MD&A, statements and 10-K risk factors are never shorter than 1% of the filing
        big = label in ("mdna", "financials") or (label == "risk_factors" and form == "10-K")
        # small sections (legal, controls...) only fall back when absent: their titles recur in the notes
        if spans and (not big or sum(e - p for p, e in spans) >= max(1500, len(text) * 0.01)):
            continue
        pos = _title_heading(text, TITLE_RES[label + ("_10q" if form == "10-Q" and label == "financials" else "")],
                             spans, [p for p, _, _ in heads])
        if pos is not None:
            heads.append((pos, "title", label))
    heads.sort()
    # Bank-style 10-Qs (JPM) open with an unheaded MD&A before the statements
    first = min((p for p, _, _ in heads if p > toc_end), default=len(text))
    mdna_size = sum(e - p for (p, _, lab), e in zip(heads, [h[0] for h in heads[1:]] + [len(text)]) if lab == "mdna")
    if form == "10-Q" and mdna_size < len(text) * 0.05 and first - toc_end > len(text) * 0.2:
        heads.append((toc_end, "lead", "mdna"))
    return sorted((p, lab) for p, _, lab in heads)


def split_sections(text, form):
    """[(label, text)] covering the whole document; text before the first heading is 'other'."""
    heads = [(0, "other")] + find_headings(text, form)
    out = []
    for (pos, label), nxt in zip(heads, heads[1:] + [(len(text), None)]):
        seg = text[pos:nxt[0]]
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1] + "\n" + seg)
        elif seg.strip():
            out.append((label, seg))
    return out


# ---------------------------------------------------------------- chunking

# sentence boundary: after .!? (also when glued: "results.The"), but not after "U.S." / "Inc." / "No."
SENT_RE = re.compile(r"(?<!U\.S\.)(?<!Inc\.)(?<!No\.)(?:(?<=[.!?”])\s+|(?<=[a-z0-9)][.!?])(?=[A-Z]))")


def units(text):
    """Table blocks (consecutive ' | ' rows) and single text lines."""
    out, table = [], []
    for line in text.split("\n"):
        if " | " in line:
            table.append(line)
            continue
        if table:
            out.append(("\n".join(table), True))
            table = []
        out.append((line, False))
    if table:
        out.append(("\n".join(table), True))
    return out


def pieces(text, is_table, limit):
    """Split one unit to pieces <= limit tokens: rows for tables, then sentences, words, raw tokens."""
    if ntok(text) <= limit:
        return [text]
    for pat in ([r"\n"] if is_table else []) + [SENT_RE.pattern, r"\s+"]:
        parts = [p for p in re.split(pat, text) if p and p.strip()]
        if len(parts) > 1:
            return [q for p in parts for q in pieces(p, False if pat != r"\n" else is_table, limit)]
    toks = ENC.encode(text, disallowed_special=())
    return [ENC.decode(toks[i:i + limit]) for i in range(0, len(toks), limit)]


def chunk_text(text, size=config.CHUNK_TOKENS, overlap=config.CHUNK_OVERLAP):
    """Greedy merge of pieces up to `size` tokens; each new chunk starts with <= `overlap` tokens of the previous.
    Pieces of one unit (sentences of a paragraph) are joined with spaces, units with newlines."""
    items = []  # (text, tokens, separator before it)
    for u, is_table in units(text):
        for i, p in enumerate(pieces(u, is_table, size)):
            items.append((p, ntok(p) + 1, "\n" if i == 0 or is_table else " "))  # +1 for the separator

    def join(xs):
        return "".join((sep if j else "") + x for j, (x, _, sep) in enumerate(xs))

    chunks, cur, cur_tok = [], [], 0
    for item in items:
        if cur and cur_tok + item[1] > size:
            chunks.append(join(cur))
            tail, t = [], 0
            for x in reversed(cur):
                if t + x[1] > overlap:
                    break
                tail.insert(0, x)
                t += x[1]
            cur, cur_tok = tail, t
        cur.append(item)
        cur_tok += item[1]
    if cur and (not chunks or cur_tok > overlap):
        chunks.append(join(cur))
    return chunks


# ---------------------------------------------------------------- filing -> chunks

def filing_chunks(row, companies):
    raw = (config.DATA_DIR / row["file"]).read_text(errors="replace")
    _, body = parse_header(raw)
    text = clean(body)
    t, form, fy, fq = row["ticker"], row["form"], row["fiscal_year"], row["fiscal_quarter"]
    comp = companies[t]
    counter, out = Counter(), []
    for label, seg in split_sections(text, form):
        for c in chunk_text(normalize_tables(seg)):
            if ntok(c) < 30:  # page furniture remnants
                continue
            n = counter[label]
            counter[label] += 1
            context = f"[{t} {comp['name']} | {form} | {row['fiscal_label']} | {label}]"
            out.append({
                "id": f"{t}-{form}-{fy}-{fq}-{label}-{n}",
                "text": c,
                "embed_text": context + "\n" + c,
                "meta": {
                    "ticker": t, "company": comp["name"], "sector": comp["sector"], "industry": comp["industry"],
                    "form": form, "fiscal_year": fy, "fiscal_quarter": fq, "fiscal_label": row["fiscal_label"],
                    "period_end": row["period_end"], "filing_date": row["filing_date"], "section": label,
                    "chunk": n, "file": row["file"], "hash": hashlib.md5(c.encode()).hexdigest()[:12],
                },
            })
    return out, counter


# ---------------------------------------------------------------- indexing

def tokenize(s):
    return re.findall(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", s.lower())


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def index_chunks(col, file, chunks):
    """Upsert new/changed chunks of one file, delete its stale ones. Returns (embedded, deleted)."""
    from llm import embed

    have = col.get(where={"file": file}, include=["metadatas"])
    old = {i: m["hash"] for i, m in zip(have["ids"], have["metadatas"])}
    new_ids = {c["id"] for c in chunks}
    stale = [i for i in old if i not in new_ids]
    if stale:
        col.delete(ids=stale)
    todo = [c for c in chunks if old.get(c["id"]) != c["meta"]["hash"]]
    for i in range(0, len(todo), config.EMBED_BATCH):
        batch = todo[i:i + config.EMBED_BATCH]
        col.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embed([c["embed_text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[c["meta"] for c in batch],
        )
    return len(todo), len(stale)


def build_bm25(col):
    from rank_bm25 import BM25Okapi

    got = col.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    corpus = [tokenize(f"{m['ticker']} {m['company']} {m['fiscal_label']} {m['section']} {d}") for d, m in zip(docs, metas)]
    config.BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"ids": ids, "metadatas": metas, "bm25": BM25Okapi(corpus)}, f)
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="5-file sample")
    ap.add_argument("--files", nargs="*", help="specific filenames")
    ap.add_argument("--dry-run", action="store_true", help="parse and chunk only")
    ap.add_argument("--show", type=int, default=0, help="print N example chunks (one per file)")
    args = ap.parse_args()

    companies = json.loads(config.COMPANIES.read_text())
    rows = [r for r in load_coverage() if r["status"] == "ok"]
    wanted = SAMPLE if args.sample else args.files
    if wanted:
        rows = [r for r in rows if r["file"] in wanted]

    col = None if args.dry_run else get_collection()
    total, examples = Counter(), []
    for r in rows:
        chunks, counts = filing_chunks(r, companies)
        msg = ""
        if col is not None:
            emb, dele = index_chunks(col, r["file"], chunks)
            msg = f" | embedded {emb}, deleted {dele}"
        print(f"{r['file']}: {len(chunks)} chunks {dict(counts)}{msg}", flush=True)
        total.update(counts)
        examples.append(chunks)

    print(f"\n{len(rows)} files, {sum(total.values())} chunks: {dict(total)}")
    if col is not None:
        print(f"Chroma: {col.count()} chunks; BM25: {build_bm25(col)} chunks -> {config.BM25_PATH}")

    for chunks in examples[:args.show]:
        # show a substantive chunk from a different section per file
        pref = ["risk_factors", "mdna", "financials", "legal", "mdna"]
        sec = pref[examples.index(chunks) % len(pref)]
        pick = [c for c in chunks if c["meta"]["section"] == sec] or chunks
        c = pick[len(pick) // 2]
        print("\n" + "=" * 100 + f"\n{c['id']}  ({ntok(c['text'])} tokens)\n{json.dumps(c['meta'])}\n" + "-" * 100)
        print(c["embed_text"][:2500] + (" ..." if len(c["embed_text"]) > 2500 else ""))


if __name__ == "__main__":
    main()
