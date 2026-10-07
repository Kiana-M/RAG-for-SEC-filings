"""Parse every filing header, derive fiscal periods, write coverage.csv and draft companies.json.

Header facts (verified against the corpus, see README):
- "Report Period" and "Quarter" are missing in 54 files; the period date is then taken from
  the URL filename (e.g. xom-20251231.htm) or the cover page ("For the fiscal year ended ...").
- "Quarter" is the calendar quarter of the period end, not the fiscal quarter, so it is ignored.
  fiscal_year / fiscal_quarter are derived from period_end and the company's fiscal year end;
  the XBRL dei tag is a cross-check and the derived value wins on conflict. fiscal_label is the
  name the company itself uses (HD/TGT call the year ending Feb 2025 "fiscal 2024").
"""
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import config

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
COVER_RE = re.compile(
    rf"for the (?:fiscal year|quarterly period|year) ended\s*({MONTHS})\s+(\d{{1,2}})\s*,\s*(\d{{4}})", re.I
)
EXCLUDED = {"GE_10K_2015-02-27_full.txt": "excluded: GE Capital FY2014, outdated"}

XBRL_PERIOD_RE = re.compile(r"(20\d\d)(FY|Q[1-4])|(FY|Q[1-4])(20\d\d)|(FY|Q[1-4])--\d\d-\d\d")


def parse_header(text):
    """Return (header dict, body after the '=====' line)."""
    head, _, body = text.partition("\n" + "=" * 10)
    body = body.split("\n", 1)[1] if "\n" in body else ""
    meta = {}
    for line in head.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def report_period(meta, body):
    """Period end date from header, else URL filename, else cover page."""
    if meta.get("Report Period"):
        return date.fromisoformat(meta["Report Period"]), "header"
    m = re.search(r"(20\d{6})", meta.get("URL", "").rsplit("/", 1)[-1])
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date(), "url"
    m = COVER_RE.search(body[:200_000])
    if m:
        return datetime.strptime(" ".join(m.groups()), "%B %d %Y").date(), "cover"
    return None, "missing"


def xbrl_period(body):
    """Fiscal period tag from the XBRL dei preamble, e.g. '2024Q3', 'FY'. Used only as a cross-check."""
    m = XBRL_PERIOD_RE.search(body[:400])
    if not m:
        return None
    year = m.group(1) or m.group(4) or ""
    per = m.group(2) or m.group(3) or m.group(5)
    return f"{year}{per}"


def fiscal_period(rp, fye_mmdd, form):
    """(fiscal_year, fiscal_quarter). Fiscal year = calendar year in which that fiscal year ends.
    A 10-day tolerance absorbs 52/53-week years (Apple, NVIDIA, Costco...)."""
    month, day = map(int, fye_mmdd.split("-"))
    fye = date(rp.year - 1, month, day)
    while fye < rp - timedelta(days=10):
        fye = date(fye.year + 1, month, day)
    if form == "10-K":
        return fye.year, "FY"
    prev = date(fye.year - 1, month, day)
    q = round((rp - prev).days / 91.3)
    return fye.year, f"Q{min(max(q, 1), 4)}"


def load_filings():
    rows = []
    for name in json.loads(config.MANIFEST.read_text())["files"]:
        text = (config.DATA_DIR / name).read_text(errors="replace")
        meta, body = parse_header(text)
        rp, rp_src = report_period(meta, body)
        rows.append({
            "file": name,
            "ticker": meta["Ticker"],
            "company": meta["Company"],
            "cik": meta.get("CIK", ""),
            "form": meta["Filing Type"].split()[0],
            "filing_date": meta["Filing Date"],
            "period_end": rp.isoformat() if rp else "",
            "period_end_src": rp_src,
            "header_quarter": meta.get("Quarter", ""),
            "xbrl_period": xbrl_period(body) or "",
        })
    return rows


def fiscal_year_ends(rows):
    """Per ticker: month-day of the latest 10-K period end; fall back to the XBRL '--MM-DD' tag, then 12-31."""
    fye = {}
    for r in sorted(rows, key=lambda r: r["period_end"]):
        if r["form"] == "10-K" and r["period_end"]:
            fye[r["ticker"]] = r["period_end"][5:]
    for r in rows:
        if r["ticker"] not in fye:
            text = (config.DATA_DIR / r["file"]).read_text(errors="replace")[:3000]
            m = re.search(r"--(\d\d-\d\d)", text)
            fye[r["ticker"]] = m.group(1) if m else "12-31"
    return fye


def draft_companies(rows, fye):
    """Write companies.json, keeping hand edits (sector, industry, aliases, fiscal_year_end) already in it."""
    existing = json.loads(config.COMPANIES.read_text()) if config.COMPANIES.exists() else {}
    out = {}
    for r in rows:
        t = r["ticker"]
        prev = existing.get(t, {})
        out[t] = {
            "name": r["company"],
            "cik": r["cik"],
            "sector": prev.get("sector"),  # manifest has no sector field: filled by hand
            "industry": prev.get("industry"),
            "aliases": prev.get("aliases", []),
            "fiscal_year_end": prev.get("fiscal_year_end") or fye[t],
        }
    config.COMPANIES.write_text(json.dumps(dict(sorted(out.items())), indent=2) + "\n")
    return out


def load_coverage():
    """coverage.csv rows with typed fields (used by ingest/retrieve/eval)."""
    with open(config.COVERAGE_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fiscal_year"] = int(r["fiscal_year"])
    return rows


def main():
    rows = load_filings()
    companies = draft_companies(rows, fiscal_year_ends(rows))
    for r in rows:
        fy, fq = fiscal_period(date.fromisoformat(r["period_end"]), companies[r["ticker"]]["fiscal_year_end"], r["form"])
        x = r["xbrl_period"]
        label_year = fy
        if len(x) > 3 and x.endswith(fq):
            label_year = int(x[:4])  # company's own naming; may differ from derived (HD, TGT)
        r["fiscal_year"], r["fiscal_quarter"] = fy, fq
        r["fiscal_label"] = f"FY{label_year}" if fq == "FY" else f"{fq} FY{label_year}"
        r["sector"] = companies[r["ticker"]]["sector"] or ""
        r["industry"] = companies[r["ticker"]]["industry"] or ""
        r["xbrl_mismatch"] = bool(x) and not (x.endswith(fq) and (len(x) <= 3 or x.startswith(str(fy))))
        r["status"] = EXCLUDED.get(r["file"], "ok")

    cols = ["ticker", "sector", "industry", "form", "fiscal_year", "fiscal_quarter", "fiscal_label", "period_end",
            "period_end_src", "filing_date", "header_quarter", "xbrl_period", "xbrl_mismatch", "status", "file"]
    rows.sort(key=lambda r: (r["ticker"], r["period_end"], r["form"]))
    with open(config.COVERAGE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    print(f"{len(rows)} filings, {len(companies)} companies -> {config.COVERAGE_CSV.name}, {config.COMPANIES.name}\n")
    print("Forms:", dict(Counter(r["form"] for r in rows)))
    print("Period end source:", dict(Counter(r["period_end_src"] for r in rows)))
    print("Status:", dict(Counter(r["status"] for r in rows)))

    by_ind = defaultdict(list)
    for t, c in companies.items():
        by_ind[(c["sector"] or "(unfilled)", c["industry"] or "")].append(t)
    print("\nCompanies by sector / industry:")
    for (s, i), ts in sorted(by_ind.items()):
        print(f"  {s} / {i}: {' '.join(ts)}")

    dupes = [k for k, n in Counter((r["ticker"], r["form"], r["fiscal_year"], r["fiscal_quarter"]) for r in rows).items() if n > 1]
    print("\nDuplicate (ticker, form, FY, quarter):", dupes or "none")
    mism = [r for r in rows if r["xbrl_mismatch"]]
    print(f"Derived period disagrees with XBRL dei tag (derived kept): {len(mism)}")
    for r in mism:
        print(f"  {r['file']}: derived FY{r['fiscal_year']} {r['fiscal_quarter']} vs xbrl {r['xbrl_period']}")
    relabeled = [r for r in rows if not r["fiscal_label"].endswith(str(r["fiscal_year"]))]
    for r in relabeled:
        print(f"  label differs: {r['file']}: fiscal_year {r['fiscal_year']}, company label {r['fiscal_label']}")

    per = defaultdict(list)
    for r in rows:
        per[r["ticker"]].append(r)
    fys = sorted({r["fiscal_year"] for r in rows if r["fiscal_year"] >= 2022})
    print(f"\nFilings per company (10-Q grid = fiscal Q1-Q3 for FY{fys[0]}-FY{fys[-1]}):")
    for t in sorted(per):
        rs = per[t]
        ks = sorted(r["fiscal_year"] for r in rs if r["form"] == "10-K")
        qs = {(r["fiscal_year"], r["fiscal_quarter"]) for r in rs if r["form"] == "10-Q"}
        grid = " ".join("".join(q[1] if (fy, q) in qs else "." for q in ("Q1", "Q2", "Q3")) for fy in fys)
        fye = companies[t]["fiscal_year_end"]
        print(f"  {t:6} n={len(rs):2}  FYE {fye}  10-K FY {','.join(map(str, ks)) or '-':24} 10-Q {grid}")
    full = fully_covered(rows)
    print(f"\nFull 10-Q coverage FY2023-2025 (Q1-Q3 each year): {len(full)}  {' '.join(full)}")


def fully_covered(rows):
    qs = defaultdict(set)
    for r in rows:
        if r["form"] == "10-Q":
            qs[r["ticker"]].add((int(r["fiscal_year"]), r["fiscal_quarter"]))
    need = {(fy, q) for fy in (2023, 2024, 2025) for q in ("Q1", "Q2", "Q3")}
    return sorted(t for t, s in qs.items() if need <= s)


if __name__ == "__main__":
    main()
