"""Verify the expanded mini corpus actually makes the test claims retrievable.

Mirrors the processor's accessibility check WITHOUT needing the LLM: for each
claim, run representative relevance queries against the live S2 API (as the
pipeline does), then call S2Searcher.get_paper_content() -- the exact function
processor.py uses -- against the local index for every returned corpusId.

A claim that previously scored -1 ("No relevant papers were found that support or
refute this claim") needs at least one accessible paper. This reports, per claim,
how many of the API-returned papers now resolve to local content.

Usage: python scripts/verify_mini_corpus_access.py [--per-query 5]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path

logging.disable(logging.INFO)  # silence the searcher's per-lookup INFO logging

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from app.config.settings import Config  # noqa: E402
from semantic_scholar.utils.searcher import S2Searcher  # noqa: E402

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "corpusId,title,abstract"
CLAIMS_BANK = ROOT / "semantic_scholar/datasets/mini/curation/mendelian_v1_claims.json"


def claim_queries(claim):
    disease = (claim.get("disease_area") or "").strip()
    genes = claim.get("genes") or []
    terms = claim.get("query_terms") or []
    queries = []
    if disease and genes:
        queries.append(f"{disease} {' '.join(genes)}")
    if terms:
        queries.append(" ".join(terms[:4]))
    if disease:
        queries.append(f"{disease} pathogenic variants genotype phenotype")
    seen, out = set(), []
    for q in queries:
        if q.strip() and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q.strip())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.1)
    args = parser.parse_args(argv)

    key = (Config.SEMANTIC_SCHOLAR_API_KEY or "").strip()
    session = requests.Session()
    if key:
        session.headers.update({"x-api-key": key})

    searcher = S2Searcher()
    print(f"Local release in use: {searcher.current_release}\n")

    with open(CLAIMS_BANK, "r", encoding="utf-8") as handle:
        claims = json.load(handle).get("claims") or []

    total_papers = 0
    total_accessible = 0
    claims_ok = 0

    for claim in claims:
        cid = claim.get("claim_id", "?")
        disease = claim.get("disease_area", "")
        union = {}
        for q in claim_queries(claim):
            try:
                resp = session.get(SEARCH_URL, params={"query": q, "limit": args.limit, "fields": FIELDS}, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("data", []) or []
            except Exception as exc:  # noqa: BLE001
                print(f"  [{cid}] query failed: {exc}")
                time.sleep(args.sleep)
                continue
            for p in data[: args.per_query]:
                c = p.get("corpusId")
                if c is not None:
                    union[str(c)] = p
            time.sleep(args.sleep)

        accessible = 0
        sources = {}
        for c in union:
            content = searcher.get_paper_content(c)
            if content and content.get("text"):
                accessible += 1
                src = content.get("source", "?")
                sources[src] = sources.get(src, 0) + 1

        total_papers += len(union)
        total_accessible += accessible
        ok = accessible > 0
        claims_ok += int(ok)
        flag = "OK " if ok else "!! "
        src_str = ", ".join(f"{k}:{v}" for k, v in sources.items()) or "none"
        print(f"{flag}[{cid}] {disease}: {accessible}/{len(union)} retrieved papers accessible "
              f"(sources: {src_str})")

    print(f"\n==== {claims_ok}/{len(claims)} claims now have >=1 accessible paper ====")
    pct = (100.0 * total_accessible / total_papers) if total_papers else 0.0
    print(f"Overall: {total_accessible}/{total_papers} retrieved papers accessible ({pct:.0f}%)")
    if claims_ok == len(claims):
        print("All test claims should now produce a real rating instead of -1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
