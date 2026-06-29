"""Expand the Mendelian mini corpus ~10x using live Semantic Scholar data.

Strategy (closes the gap where live search returns papers absent from the corpus):
  - For each curated claim/disease, pull papers from the live S2 API using both
    the relevance `paper/search` endpoint (mirrors what the pipeline retrieves)
    and the `paper/search/bulk` endpoint (volume), keeping only papers that carry
    an abstract -- an abstract is what makes a paper locally *accessible*.
  - Merge the harvested records with the existing curated source extracts.
  - Rewrite the source-extract jsonl files and the tracked manifest so every
    manifest corpusId has a local record (the build then needs zero remote shard
    streaming).

It does NOT build/index -- run materialize after this (see rebuild step).

Usage:
  python scripts/expand_mini_corpus.py --per-disease 500 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from app.config.settings import Config  # noqa: E402

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "corpusId,externalIds,url,title,abstract,year,venue,authors"

CLAIMS_BANK = ROOT / "semantic_scholar/datasets/mini/curation/mendelian_v1_claims.json"
EXTRACT_DIR = ROOT / "semantic_scholar/datasets/mini/source_extracts/mendelian_v1"
MANIFEST_PATH = ROOT / "semantic_scholar/manifests/mendelian_v1.json"
RELEASE_ID = "2026-05-26"
MINI_RELEASE_ID = "2026-05-26-mini-mendelian-v1"


# ----------------------------- API harvesting -----------------------------

class Harvester:
    def __init__(self, sleep: float = 1.1):
        key = (Config.SEMANTIC_SCHOLAR_API_KEY or "").strip()
        self.session = requests.Session()
        if key:
            self.session.headers.update({"x-api-key": key})
        self.sleep = sleep
        self.calls = 0

    def _get(self, url, params):
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                self.calls += 1
                if resp.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    print(f"    ! request failed ({exc})")
                    return {}
                time.sleep(1.5 + attempt)
            finally:
                time.sleep(self.sleep)
        return {}

    def relevance(self, query, limit=100):
        data = self._get(SEARCH_URL, {"query": query, "limit": limit, "fields": FIELDS})
        return data.get("data", []) or []

    def bulk(self, query, want, fields=FIELDS):
        """Pull up to `want` records via the bulk endpoint, paging with token."""
        out = []
        token = None
        while len(out) < want:
            params = {"query": query, "fields": fields, "limit": 1000}
            if token:
                params["token"] = token
            data = self._get(BULK_URL, params)
            rows = data.get("data", []) or []
            out.extend(rows)
            token = data.get("token")
            if not token or not rows:
                break
        return out


# ----------------------------- record synthesis -----------------------------

def to_paper_record(p):
    corpus = p.get("corpusId")
    ext = p.get("externalIds") or {}
    ext = {**ext, "CorpusId": str(corpus)} if corpus is not None else ext
    return {
        "corpusid": int(corpus),
        "externalids": ext,
        "url": p.get("url"),
        "title": p.get("title"),
        "authors": [{"authorId": a.get("authorId"), "name": a.get("name")}
                    for a in (p.get("authors") or [])],
        "year": p.get("year"),
        "venue": p.get("venue"),
    }


def to_abstract_record(p):
    return {"corpusid": int(p["corpusId"]), "abstract": p.get("abstract")}


def author_records(p):
    out = []
    for a in (p.get("authors") or []):
        aid = a.get("authorId")
        if aid:
            out.append({"authorid": str(aid), "name": a.get("name")})
    return out


# ----------------------------- extract IO -----------------------------

def read_jsonl(path):
    records = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            key = rec.get("corpusid", rec.get("authorid"))
            if key is not None:
                records[str(key)] = rec
    return records


def write_jsonl(path, records_by_key):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records_by_key.values():
            handle.write(json.dumps(rec, ensure_ascii=True) + "\n")


# ----------------------------- query planning -----------------------------

def disease_queries(claim):
    disease = (claim.get("disease_area") or "").strip()
    genes = claim.get("genes") or []
    terms = claim.get("query_terms") or []
    rel = []
    if disease and genes:
        rel.append(f"{disease} {' '.join(genes)}")
    if terms:
        rel.append(" ".join(terms[:5]))
    if disease:
        rel.append(f"{disease} pathogenic variants genotype phenotype")
        rel.append(f"{disease} diagnosis treatment review")
    for g in genes:
        rel.append(f"{disease} {g} mutation".strip())
    # bulk query: a broad boolean-ish phrase the bulk endpoint matches on
    bulk_q = disease
    if genes:
        bulk_q = f"{disease} {genes[0]}"
    # dedup relevance queries
    seen, rel_out = set(), []
    for q in rel:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            rel_out.append(q)
    return rel_out, bulk_q


# ----------------------------- main -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-disease", type=int, default=500,
                        help="Target unique papers (with abstract) to add per disease.")
    parser.add_argument("--sleep", type=float, default=1.1)
    parser.add_argument("--dry-run", action="store_true", help="Harvest + report only; write nothing.")
    args = parser.parse_args(argv)

    with open(CLAIMS_BANK, "r", encoding="utf-8") as handle:
        claims = json.load(handle).get("claims") or []

    harvester = Harvester(sleep=args.sleep)
    # corpusId -> api paper dict (only papers that have an abstract)
    harvested = {}
    per_disease_counts = {}

    for claim in claims:
        cid = claim.get("claim_id", "?")
        disease = claim.get("disease_area", "")
        rel_queries, bulk_q = disease_queries(claim)
        before = len(harvested)
        added_here = 0

        # 1) relevance hits (what the pipeline is most likely to retrieve)
        for q in rel_queries:
            for p in harvester.relevance(q, limit=100):
                c = p.get("corpusId")
                if c is None or not p.get("abstract"):
                    continue
                if str(c) not in harvested:
                    harvested[str(c)] = p
                    added_here += 1

        # 2) bulk breadth until we hit the per-disease target
        need = max(0, args.per_disease - added_here)
        if need > 0:
            # over-fetch since only ~45% carry abstracts
            for p in harvester.bulk(bulk_q, want=int(need / 0.4) + 1000):
                c = p.get("corpusId")
                if c is None or not p.get("abstract"):
                    continue
                if str(c) not in harvested:
                    harvested[str(c)] = p
                    added_here += 1
                if added_here >= args.per_disease:
                    break

        per_disease_counts[cid] = added_here
        print(f"[{cid}] {disease}: +{len(harvested) - before} new (target {args.per_disease}) "
              f"| total harvested {len(harvested)} | api calls {harvester.calls}")

    print(f"\nHarvested {len(harvested)} unique papers with abstracts ({harvester.calls} API calls).")

    # ---- merge with existing extracts ----
    existing_papers = read_jsonl(EXTRACT_DIR / "papers.jsonl")
    existing_abstracts = read_jsonl(EXTRACT_DIR / "abstracts.jsonl")
    existing_authors = read_jsonl(EXTRACT_DIR / "authors.jsonl")
    existing_tldrs = read_jsonl(EXTRACT_DIR / "tldrs.jsonl")
    existing_s2orc = read_jsonl(EXTRACT_DIR / "s2orc_v2.jsonl")
    print(f"Existing extracts: papers={len(existing_papers)} abstracts={len(existing_abstracts)} "
          f"authors={len(existing_authors)} tldrs={len(existing_tldrs)} s2orc_v2={len(existing_s2orc)}")

    papers = dict(existing_papers)
    abstracts = dict(existing_abstracts)
    # Authors are intentionally NOT expanded: the search API does not return
    # h-index/paper-count/citation-count, so synthesized author records carry no
    # bibliometric value and would bloat the index by tens of thousands of rows.
    # The pipeline gracefully skips authors it cannot find locally.
    authors = dict(existing_authors)
    for c, p in harvested.items():
        papers[c] = to_paper_record(p)
        abstracts[c] = to_abstract_record(p)

    print(f"Merged totals: papers={len(papers)} abstracts={len(abstracts)} "
          f"authors={len(authors)} (authors unchanged)")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    # ---- write extract files ----
    write_jsonl(EXTRACT_DIR / "papers.jsonl", papers)
    write_jsonl(EXTRACT_DIR / "abstracts.jsonl", abstracts)
    write_jsonl(EXTRACT_DIR / "authors.jsonl", authors)
    # tldrs / s2orc_v2 unchanged (the search API provides neither)

    # ---- write manifest (curated ID lists only) ----
    # The manifest carries only the curated corpus/author IDs. The build streams
    # those IDs from the Semantic Scholar dataset shards into the datasets volume,
    # exactly like a full-corpus download — just filtered to this ID set. We do
    # NOT emit machine-specific source_files paths: those bind the manifest to one
    # host and break portable (e.g. Docker) builds.
    def ids_in(records):
        return sorted(records.keys(), key=lambda v: int(v) if str(v).isdigit() else v)

    manifest = {
        "release_id": RELEASE_ID,
        "mini_release_id": MINI_RELEASE_ID,
        "topic_label": "mendelian_disease_v1",
        "datasets": {
            "papers": {"corpus_ids": ids_in(papers)},
            "abstracts": {"corpus_ids": ids_in(abstracts)},
            "tldrs": {"corpus_ids": ids_in(existing_tldrs)},
            "s2orc_v2": {"corpus_ids": ids_in(existing_s2orc)},
            "authors": {"author_ids": ids_in(authors)},
        },
        "notes": [
            "Expanded via scripts/expand_mini_corpus.py using live Semantic Scholar data.",
            "The build streams these IDs from Semantic Scholar dataset shards, like a full-corpus download.",
            "Content lookup prefers s2orc_v2, then abstracts and tldrs.",
        ],
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"  papers={len(papers)} abstracts={len(abstracts)} authors={len(authors)} "
          f"tldrs={len(existing_tldrs)} s2orc_v2={len(existing_s2orc)}")
    print("\nNext: rebuild + index via downloader.materialize_mini_corpus().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
