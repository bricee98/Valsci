"""Capture Semantic Scholar search provenance for a curated mini corpus."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from rich.console import Console
from urllib.parse import quote


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.config.settings import Config


console = Console()


def console_safe(value: Any) -> str:
    return str(value).encode("ascii", errors="replace").decode("ascii")

DEFAULT_QUERY_BANK_PATH = (
    project_root / "semantic_scholar/datasets/mini/curation/mendelian_v1_queries.json"
)
DEFAULT_CLAIM_BANK_PATH = (
    project_root / "semantic_scholar/datasets/mini/curation/mendelian_v1_claims.json"
)
DEFAULT_OUTPUT_DIR = project_root / "semantic_scholar/datasets/mini/curation"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REFERENCES_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"

SEARCH_FIELDS = [
    "paperId",
    "corpusId",
    "externalIds",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "publicationVenue",
    "url",
    "isOpenAccess",
    "openAccessPdf",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "citationCount",
    "influentialCitationCount",
    "referenceCount",
]

REFERENCE_FIELDS = [
    "contexts",
    "intents",
    "isInfluential",
    "citedPaper.paperId",
    "citedPaper.corpusId",
    "citedPaper.externalIds",
    "citedPaper.title",
    "citedPaper.abstract",
    "citedPaper.year",
    "citedPaper.authors",
    "citedPaper.venue",
    "citedPaper.publicationVenue",
    "citedPaper.url",
    "citedPaper.isOpenAccess",
    "citedPaper.openAccessPdf",
    "citedPaper.fieldsOfStudy",
    "citedPaper.s2FieldsOfStudy",
    "citedPaper.citationCount",
    "citedPaper.influentialCitationCount",
    "citedPaper.referenceCount",
]

GENERIC_REFERENCE_TERMS = [
    "mendelian",
    "monogenic",
    "pathogenic variant",
    "pathogenic variants",
    "genotype",
    "phenotype",
    "penetrance",
    "expressivity",
    "modifier",
    "modifiers",
    "variant classification",
    "newborn screening",
    "gene therapy",
    "enzyme replacement",
    "autosomal dominant",
    "autosomal recessive",
    "x-linked",
    "compound heterozygous",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


@dataclass(frozen=True)
class QuerySpec:
    query_set_id: str
    query_set_priority: int
    query_set_description: str
    query_id: str
    query: str
    intent: str
    linked_claim_ids: Tuple[str, ...]
    max_results: int

    def to_provenance(self) -> Dict[str, Any]:
        return {
            "query_set_id": self.query_set_id,
            "query_set_priority": self.query_set_priority,
            "query_set_description": self.query_set_description,
            "query_id": self.query_id,
            "query": self.query,
            "intent": self.intent,
            "linked_claim_ids": list(self.linked_claim_ids),
            "max_results": self.max_results,
        }


def flatten_query_bank(query_bank: Dict[str, Any], *, limit_override: Optional[int] = None) -> List[QuerySpec]:
    default_limit = safe_int(
        (query_bank.get("retrieval_policy") or {}).get("search_results_per_query"),
        25,
    )
    specs: List[QuerySpec] = []
    seen_ids = set()
    for query_set in query_bank.get("query_sets") or []:
        if not isinstance(query_set, dict):
            continue
        query_set_id = str(query_set.get("set_id") or "").strip()
        if not query_set_id:
            raise ValueError("Every query set must include set_id.")
        priority = safe_int(query_set.get("priority"), 999)
        description = str(query_set.get("description") or "")
        for query in query_set.get("queries") or []:
            if not isinstance(query, dict):
                continue
            query_id = str(query.get("query_id") or "").strip()
            text = str(query.get("query") or "").strip()
            if not query_id or not text:
                raise ValueError(f"Every query must include query_id and query text in set {query_set_id}.")
            if query_id in seen_ids:
                raise ValueError(f"Duplicate query_id: {query_id}")
            seen_ids.add(query_id)
            linked_claim_ids = tuple(
                str(value).strip()
                for value in (query.get("linked_claim_ids") or [])
                if str(value).strip()
            )
            max_results = safe_int(query.get("max_results"), default_limit)
            if limit_override is not None:
                max_results = limit_override
            specs.append(
                QuerySpec(
                    query_set_id=query_set_id,
                    query_set_priority=priority,
                    query_set_description=description,
                    query_id=query_id,
                    query=text,
                    intent=str(query.get("intent") or ""),
                    linked_claim_ids=linked_claim_ids,
                    max_results=max(1, min(max_results, 100)),
                )
            )
    specs.sort(key=lambda item: (item.query_set_priority, item.query_set_id, item.query_id))
    return specs


def load_claim_metadata(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {}
    claim_bank = load_json(path)
    metadata: Dict[str, Dict[str, Any]] = {}
    for claim in claim_bank.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            continue
        genes = [
            str(gene).strip()
            for gene in (claim.get("genes") or [])
            if str(gene).strip()
        ]
        metadata[claim_id] = {
            "claim_id": claim_id,
            "disease_area": str(claim.get("disease_area") or "").strip(),
            "genes": genes,
            "query_terms": [
                str(term).strip()
                for term in (claim.get("query_terms") or [])
                if str(term).strip()
            ],
            "expected_evidence_behavior": claim.get("expected_evidence_behavior"),
            "content_targets": claim.get("content_targets") or [],
        }
    return metadata


def enrich_query_provenance(
    query: Dict[str, Any],
    claim_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not claim_metadata:
        return query
    disease_areas: List[str] = []
    genes: List[str] = []
    expected_evidence_behaviors: List[str] = []
    content_targets: List[str] = []
    for claim_id in query.get("linked_claim_ids") or []:
        metadata = claim_metadata.get(str(claim_id)) or {}
        disease_area = str(metadata.get("disease_area") or "").strip()
        if disease_area and disease_area not in disease_areas:
            disease_areas.append(disease_area)
        for gene in metadata.get("genes") or []:
            if gene and gene not in genes:
                genes.append(gene)
        behavior = str(metadata.get("expected_evidence_behavior") or "").strip()
        if behavior and behavior not in expected_evidence_behaviors:
            expected_evidence_behaviors.append(behavior)
        for target in metadata.get("content_targets") or []:
            target = str(target).strip()
            if target and target not in content_targets:
                content_targets.append(target)
    query["linked_disease_areas"] = disease_areas
    query["linked_genes"] = genes
    query["linked_expected_evidence_behaviors"] = expected_evidence_behaviors
    query["linked_content_targets"] = content_targets
    return query


def paper_identity(paper: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    corpus_id = paper.get("corpusId")
    paper_id = paper.get("paperId")
    return (
        str(corpus_id) if corpus_id is not None and str(corpus_id).strip() else None,
        str(paper_id) if paper_id is not None and str(paper_id).strip() else None,
    )


def compact_authors(authors: Any, *, max_authors: int = 8) -> List[Dict[str, Any]]:
    if not isinstance(authors, list):
        return []
    compact = []
    for author in authors[:max_authors]:
        if not isinstance(author, dict):
            continue
        compact.append(
            {
                "authorId": author.get("authorId"),
                "name": author.get("name"),
            }
        )
    return compact


def normalize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    open_access_pdf = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else {}
    publication_venue = (
        paper.get("publicationVenue")
        if isinstance(paper.get("publicationVenue"), dict)
        else {}
    )
    return {
        "paperId": paper.get("paperId"),
        "corpusId": paper.get("corpusId"),
        "externalIds": paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {},
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "publicationVenue": {
            "id": publication_venue.get("id"),
            "name": publication_venue.get("name"),
            "type": publication_venue.get("type"),
            "url": publication_venue.get("url"),
        },
        "url": paper.get("url"),
        "isOpenAccess": bool(paper.get("isOpenAccess")),
        "openAccessPdf": {
            "url": open_access_pdf.get("url"),
            "status": open_access_pdf.get("status"),
        },
        "fieldsOfStudy": paper.get("fieldsOfStudy") or [],
        "s2FieldsOfStudy": paper.get("s2FieldsOfStudy") or [],
        "citationCount": safe_int(paper.get("citationCount")),
        "influentialCitationCount": safe_int(paper.get("influentialCitationCount")),
        "referenceCount": safe_int(paper.get("referenceCount")),
        "authors": compact_authors(paper.get("authors")),
    }


def build_result_record(
    *,
    run_id: str,
    topic_label: str,
    query_spec: QuerySpec,
    rank: int,
    paper: Dict[str, Any],
    captured_at: str,
    claim_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = normalize_paper(paper)
    query = enrich_query_provenance(query_spec.to_provenance(), claim_metadata)
    return {
        "run_id": run_id,
        "captured_at": captured_at,
        "topic_label": topic_label,
        "query": query,
        "rank": rank,
        "paper": normalized,
    }


def candidate_key(record: Dict[str, Any]) -> Optional[str]:
    paper = record.get("paper") or {}
    corpus_id = paper.get("corpusId")
    if corpus_id is not None and str(corpus_id).strip():
        return str(corpus_id)
    return None


def score_candidate(candidate: Dict[str, Any]) -> float:
    hits = candidate.get("query_hits") or []
    ranks = [safe_int(hit.get("rank"), 999) for hit in hits]
    query_hit_count = len(hits)
    claim_count = len(candidate.get("linked_claim_ids") or [])
    disease_area_count = len(candidate.get("disease_areas") or [])
    best_rank = min(ranks) if ranks else 999
    rank_points = sum(max(0, 31 - rank) for rank in ranks) / 3.0
    citation_count = safe_int(candidate.get("citationCount"))
    influential_count = safe_int(candidate.get("influentialCitationCount"))
    reference_count = safe_int(candidate.get("referenceCount"))
    score = (
        query_hit_count * 25
        + claim_count * 12
        + disease_area_count * 8
        + rank_points
        + max(0, 35 - best_rank)
        + min(math.log10(citation_count + 1) * 8, 32)
        + min(math.log10(influential_count + 1) * 6, 24)
        + min(math.log10(reference_count + 1) * 3, 12)
    )
    if candidate.get("isOpenAccess"):
        score += 8
    if candidate.get("openAccessPdf", {}).get("url"):
        score += 6
    if candidate.get("abstract_present"):
        score += 5
    if candidate.get("year") and safe_int(candidate.get("year")) >= 2010:
        score += 3
    return round(score, 3)


def build_seed_candidates(
    records: Iterable[Dict[str, Any]],
    claim_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    candidates: Dict[str, Dict[str, Any]] = {}
    skipped_without_corpus_id = 0
    for record in records:
        key = candidate_key(record)
        if not key:
            skipped_without_corpus_id += 1
            continue
        paper = record.get("paper") or {}
        query = enrich_query_provenance(dict(record.get("query") or {}), claim_metadata)
        candidate = candidates.setdefault(
            key,
            {
                "corpusId": str(paper.get("corpusId")),
                "paperId": paper.get("paperId"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "publicationVenue": paper.get("publicationVenue") or {},
                "url": paper.get("url"),
                "isOpenAccess": bool(paper.get("isOpenAccess")),
                "openAccessPdf": paper.get("openAccessPdf") or {},
                "abstract_present": bool(paper.get("abstract")),
                "abstract_snippet": str(paper.get("abstract") or "")[:600],
                "fieldsOfStudy": paper.get("fieldsOfStudy") or [],
                "s2FieldsOfStudy": paper.get("s2FieldsOfStudy") or [],
                "citationCount": safe_int(paper.get("citationCount")),
                "influentialCitationCount": safe_int(paper.get("influentialCitationCount")),
                "referenceCount": safe_int(paper.get("referenceCount")),
                "authors": paper.get("authors") or [],
                "query_hits": [],
                "query_ids": [],
                "query_set_ids": [],
                "linked_claim_ids": [],
                "disease_areas": [],
                "genes": [],
            },
        )

        # Keep strongest bibliometric/open-access values if repeated results differ.
        candidate["citationCount"] = max(
            safe_int(candidate.get("citationCount")),
            safe_int(paper.get("citationCount")),
        )
        candidate["influentialCitationCount"] = max(
            safe_int(candidate.get("influentialCitationCount")),
            safe_int(paper.get("influentialCitationCount")),
        )
        candidate["referenceCount"] = max(
            safe_int(candidate.get("referenceCount")),
            safe_int(paper.get("referenceCount")),
        )
        candidate["isOpenAccess"] = bool(candidate.get("isOpenAccess") or paper.get("isOpenAccess"))
        if not candidate.get("openAccessPdf", {}).get("url") and paper.get("openAccessPdf", {}).get("url"):
            candidate["openAccessPdf"] = paper.get("openAccessPdf") or {}
        if not candidate.get("abstract_present") and paper.get("abstract"):
            candidate["abstract_present"] = True
            candidate["abstract_snippet"] = str(paper.get("abstract") or "")[:600]

        query_id = query.get("query_id")
        query_set_id = query.get("query_set_id")
        linked_claim_ids = [str(value) for value in query.get("linked_claim_ids") or []]
        linked_disease_areas = [
            str(value)
            for value in query.get("linked_disease_areas") or []
            if str(value).strip()
        ]
        linked_genes = [
            str(value)
            for value in query.get("linked_genes") or []
            if str(value).strip()
        ]
        candidate["query_hits"].append(
            {
                "query_id": query_id,
                "query_set_id": query_set_id,
                "rank": record.get("rank"),
                "intent": query.get("intent"),
                "linked_claim_ids": linked_claim_ids,
                "linked_disease_areas": linked_disease_areas,
                "linked_genes": linked_genes,
            }
        )
        if query_id and query_id not in candidate["query_ids"]:
            candidate["query_ids"].append(query_id)
        if query_set_id and query_set_id not in candidate["query_set_ids"]:
            candidate["query_set_ids"].append(query_set_id)
        for claim_id in linked_claim_ids:
            if claim_id not in candidate["linked_claim_ids"]:
                candidate["linked_claim_ids"].append(claim_id)
        for disease_area in linked_disease_areas:
            if disease_area not in candidate["disease_areas"]:
                candidate["disease_areas"].append(disease_area)
        for gene in linked_genes:
            if gene not in candidate["genes"]:
                candidate["genes"].append(gene)

    for candidate in candidates.values():
        ranks = [safe_int(hit.get("rank"), 999) for hit in candidate["query_hits"]]
        candidate["query_hit_count"] = len(candidate["query_hits"])
        candidate["disease_area_count"] = len(candidate["disease_areas"])
        candidate["best_rank"] = min(ranks) if ranks else None
        candidate["average_rank"] = round(sum(ranks) / len(ranks), 2) if ranks else None
        candidate["score"] = score_candidate(candidate)

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            -safe_float(item.get("score")),
            safe_int(item.get("best_rank"), 999),
            -safe_int(item.get("citationCount")),
            str(item.get("title") or ""),
        ),
    )
    return {
        "candidate_count": len(sorted_candidates),
        "skipped_without_corpus_id": skipped_without_corpus_id,
        "candidates": sorted_candidates,
    }


def _candidate_id(candidate: Dict[str, Any]) -> Optional[str]:
    corpus_id = candidate.get("corpusId")
    return str(corpus_id) if corpus_id is not None and str(corpus_id).strip() else None


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, int, int, str]:
    return (
        -safe_float(candidate.get("score")),
        safe_int(candidate.get("best_rank"), 999),
        -safe_int(candidate.get("citationCount")),
        str(candidate.get("title") or ""),
    )


def _append_selection(
    selected: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Any],
    reason: str,
) -> None:
    corpus_id = _candidate_id(candidate)
    if not corpus_id:
        return
    entry = selected.setdefault(corpus_id, dict(candidate))
    reasons = entry.setdefault("selection_reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def select_seed_candidates(
    candidates: Iterable[Dict[str, Any]],
    *,
    claim_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    max_seeds: int = 120,
    top_global: int = 40,
    per_disease: int = 8,
    per_claim: int = 5,
) -> List[Dict[str, Any]]:
    sorted_candidates = sorted(
        [candidate for candidate in candidates if _candidate_id(candidate)],
        key=_candidate_sort_key,
    )
    selected: Dict[str, Dict[str, Any]] = {}

    for candidate in sorted_candidates[:max(0, top_global)]:
        _append_selection(selected, candidate, "top_global_score")

    disease_areas = sorted(
        {
            str(metadata.get("disease_area") or "").strip()
            for metadata in (claim_metadata or {}).values()
            if str(metadata.get("disease_area") or "").strip()
        }
    )
    for disease_area in disease_areas:
        matches = [
            candidate
            for candidate in sorted_candidates
            if disease_area in (candidate.get("disease_areas") or [])
        ]
        for candidate in matches[:max(0, per_disease)]:
            _append_selection(selected, candidate, f"top_for_disease:{disease_area}")

    claim_ids = sorted((claim_metadata or {}).keys())
    for claim_id in claim_ids:
        matches = [
            candidate
            for candidate in sorted_candidates
            if claim_id in (candidate.get("linked_claim_ids") or [])
        ]
        for candidate in matches[:max(0, per_claim)]:
            _append_selection(selected, candidate, f"top_for_claim:{claim_id}")

    return sorted(selected.values(), key=_candidate_sort_key)[:max(0, max_seeds)]


def reference_text_blob(paper: Dict[str, Any]) -> str:
    fields = paper.get("fieldsOfStudy") or []
    s2_fields = paper.get("s2FieldsOfStudy") or []
    s2_categories = []
    if isinstance(s2_fields, list):
        for item in s2_fields:
            if isinstance(item, dict):
                s2_categories.append(str(item.get("category") or ""))
            else:
                s2_categories.append(str(item))
    return " ".join(
        str(value or "")
        for value in [
            paper.get("title"),
            paper.get("abstract"),
            paper.get("venue"),
            " ".join(str(field) for field in fields),
            " ".join(s2_categories),
        ]
    ).lower()


def _term_hits(text: str, terms: Iterable[str], *, weight: int) -> Tuple[int, List[str]]:
    score = 0
    hits: List[str] = []
    for raw_term in terms:
        term = str(raw_term or "").strip()
        if not term:
            continue
        normalized = term.lower()
        if normalized in text:
            score += weight
            hits.append(term)
    return score, hits


def score_reference_relevance(
    paper: Dict[str, Any],
    seed: Dict[str, Any],
    *,
    claim_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    text = reference_text_blob(paper)
    disease_terms: List[str] = []
    gene_terms: List[str] = []
    query_terms: List[str] = []
    for disease_area in seed.get("disease_areas") or []:
        if disease_area:
            disease_terms.append(str(disease_area))
    for gene in seed.get("genes") or []:
        if gene:
            gene_terms.append(str(gene))
    for claim_id in seed.get("linked_claim_ids") or []:
        metadata = (claim_metadata or {}).get(str(claim_id)) or {}
        if metadata.get("disease_area"):
            disease_terms.append(str(metadata["disease_area"]))
        gene_terms.extend(str(gene) for gene in metadata.get("genes") or [])
        query_terms.extend(str(term) for term in metadata.get("query_terms") or [])

    score = 0
    matched_terms: List[str] = []
    for terms, weight in [
        (disease_terms, 5),
        (gene_terms, 4),
        (query_terms, 3),
        (GENERIC_REFERENCE_TERMS, 2),
    ]:
        term_score, hits = _term_hits(text, terms, weight=weight)
        score += term_score
        for hit in hits:
            if hit not in matched_terms:
                matched_terms.append(hit)

    # Keep a little room for classic broad papers with sparse abstracts.
    citation_count = safe_int(paper.get("citationCount"))
    if citation_count >= 1000 and any(term in text for term in ["genetic", "gene", "variant", "disease"]):
        score += 3
        matched_terms.append("high_citation_genetics_context")

    return {
        "score": score,
        "matched_terms": matched_terms,
        "citation_bonus": citation_count >= 1000,
    }


def build_reference_record(
    *,
    run_id: str,
    seed: Dict[str, Any],
    rank: int,
    reference: Dict[str, Any],
    relevance: Dict[str, Any],
    captured_at: str,
) -> Optional[Dict[str, Any]]:
    cited_paper = reference.get("citedPaper") if isinstance(reference.get("citedPaper"), dict) else {}
    normalized = normalize_paper(cited_paper)
    if normalized.get("corpusId") is None and normalized.get("paperId") is None:
        return None
    return {
        "run_id": run_id,
        "captured_at": captured_at,
        "seed": {
            "corpusId": seed.get("corpusId"),
            "paperId": seed.get("paperId"),
            "title": seed.get("title"),
            "score": seed.get("score"),
            "linked_claim_ids": seed.get("linked_claim_ids") or [],
            "disease_areas": seed.get("disease_areas") or [],
        },
        "rank": rank,
        "reference_metadata": {
            "contexts": reference.get("contexts") or [],
            "intents": reference.get("intents") or [],
            "isInfluential": bool(reference.get("isInfluential")),
        },
        "relevance": relevance,
        "paper": normalized,
    }


def build_reference_candidate_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: Dict[str, Dict[str, Any]] = {}
    skipped_without_corpus_id = 0
    for record in records:
        paper = record.get("paper") or {}
        corpus_id = paper.get("corpusId")
        if corpus_id is None or not str(corpus_id).strip():
            skipped_without_corpus_id += 1
            continue
        key = str(corpus_id)
        seed = record.get("seed") or {}
        candidate = candidates.setdefault(
            key,
            {
                "corpusId": key,
                "paperId": paper.get("paperId"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "url": paper.get("url"),
                "isOpenAccess": bool(paper.get("isOpenAccess")),
                "openAccessPdf": paper.get("openAccessPdf") or {},
                "abstract_present": bool(paper.get("abstract")),
                "abstract_snippet": str(paper.get("abstract") or "")[:600],
                "fieldsOfStudy": paper.get("fieldsOfStudy") or [],
                "s2FieldsOfStudy": paper.get("s2FieldsOfStudy") or [],
                "citationCount": safe_int(paper.get("citationCount")),
                "influentialCitationCount": safe_int(paper.get("influentialCitationCount")),
                "referenceCount": safe_int(paper.get("referenceCount")),
                "authors": paper.get("authors") or [],
                "seed_hits": [],
                "seed_corpus_ids": [],
                "linked_claim_ids": [],
                "disease_areas": [],
                "matched_terms": [],
                "best_reference_rank": None,
                "max_relevance_score": 0,
            },
        )
        candidate["citationCount"] = max(candidate["citationCount"], safe_int(paper.get("citationCount")))
        candidate["influentialCitationCount"] = max(
            candidate["influentialCitationCount"],
            safe_int(paper.get("influentialCitationCount")),
        )
        candidate["referenceCount"] = max(candidate["referenceCount"], safe_int(paper.get("referenceCount")))
        candidate["isOpenAccess"] = bool(candidate.get("isOpenAccess") or paper.get("isOpenAccess"))
        if not candidate.get("abstract_present") and paper.get("abstract"):
            candidate["abstract_present"] = True
            candidate["abstract_snippet"] = str(paper.get("abstract") or "")[:600]
        rank = safe_int(record.get("rank"), 999)
        if candidate["best_reference_rank"] is None or rank < candidate["best_reference_rank"]:
            candidate["best_reference_rank"] = rank
        relevance = record.get("relevance") or {}
        candidate["max_relevance_score"] = max(
            safe_int(candidate.get("max_relevance_score")),
            safe_int(relevance.get("score")),
        )
        seed_corpus_id = seed.get("corpusId")
        if seed_corpus_id and seed_corpus_id not in candidate["seed_corpus_ids"]:
            candidate["seed_corpus_ids"].append(seed_corpus_id)
        candidate["seed_hits"].append(
            {
                "seed_corpusId": seed_corpus_id,
                "seed_title": seed.get("title"),
                "rank": record.get("rank"),
                "relevance_score": relevance.get("score"),
            }
        )
        for claim_id in seed.get("linked_claim_ids") or []:
            if claim_id not in candidate["linked_claim_ids"]:
                candidate["linked_claim_ids"].append(claim_id)
        for disease_area in seed.get("disease_areas") or []:
            if disease_area not in candidate["disease_areas"]:
                candidate["disease_areas"].append(disease_area)
        for term in relevance.get("matched_terms") or []:
            if term not in candidate["matched_terms"]:
                candidate["matched_terms"].append(term)

    for candidate in candidates.values():
        candidate["seed_hit_count"] = len(candidate["seed_hits"])
        candidate["reference_score"] = round(
            safe_int(candidate.get("max_relevance_score")) * 8
            + candidate["seed_hit_count"] * 20
            + max(0, 51 - safe_int(candidate.get("best_reference_rank"), 999))
            + min(math.log10(safe_int(candidate.get("citationCount")) + 1) * 8, 32)
            + (8 if candidate.get("isOpenAccess") else 0)
            + (5 if candidate.get("abstract_present") else 0),
            3,
        )

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            -safe_float(item.get("reference_score")),
            safe_int(item.get("best_reference_rank"), 999),
            -safe_int(item.get("citationCount")),
            str(item.get("title") or ""),
        ),
    )
    return {
        "candidate_count": len(sorted_candidates),
        "skipped_without_corpus_id": skipped_without_corpus_id,
        "candidates": sorted_candidates,
    }


def paper_author_ids(paper: Dict[str, Any]) -> List[str]:
    authors = paper.get("authors") if isinstance(paper.get("authors"), list) else []
    if not authors:
        return []
    key_authors = [authors[0]]
    if len(authors) > 1:
        key_authors.append(authors[-1])
    ids: List[str] = []
    for author in key_authors:
        if not isinstance(author, dict):
            continue
        author_id = author.get("authorId") or author.get("authorid")
        if author_id is not None and str(author_id).strip() and str(author_id) not in ids:
            ids.append(str(author_id))
    return ids


def build_final_selection(
    *,
    selected_seeds: List[Dict[str, Any]],
    reference_candidates: List[Dict[str, Any]],
    max_corpus_ids: int,
    release_id: str,
    topic_label: str,
) -> Dict[str, Any]:
    selected: Dict[str, Dict[str, Any]] = {}
    author_ids: List[str] = []

    def add_paper(paper: Dict[str, Any], source: str) -> None:
        corpus_id = _candidate_id(paper)
        if not corpus_id or corpus_id in selected or len(selected) >= max_corpus_ids:
            return
        selected[corpus_id] = {
            "corpusId": corpus_id,
            "paperId": paper.get("paperId"),
            "title": paper.get("title"),
            "source": source,
            "score": paper.get("score") or paper.get("reference_score"),
            "linked_claim_ids": paper.get("linked_claim_ids") or [],
            "disease_areas": paper.get("disease_areas") or [],
        }
        for author_id in paper_author_ids(paper):
            if author_id not in author_ids:
                author_ids.append(author_id)

    for seed in selected_seeds:
        add_paper(seed, "seed")
    for reference in reference_candidates:
        add_paper(reference, "reference")

    return {
        "release_id": release_id,
        "topic_label": topic_label,
        "created_at": utc_now(),
        "selection_policy": {
            "seed_policy": "top global + top per disease area + top per claim",
            "reference_policy": "bounded outbound references filtered by claim/gene/disease relevance",
            "max_corpus_ids": max_corpus_ids,
            "author_policy": "first and last authors from selected papers where author IDs are available",
        },
        "corpus_id_count": len(selected),
        "author_id_count": len(author_ids),
        "corpus_ids": list(selected.keys()),
        "author_ids": author_ids,
        "papers": list(selected.values()),
    }


class SemanticScholarSearchClient:
    def __init__(self, *, api_key: Optional[str] = None, timeout: int = 30, max_retries: int = 4):
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def search(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(SEARCH_FIELDS),
        }
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(SEARCH_URL, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    wait_seconds = min(30, 2 ** attempt + 1)
                    console.print(f"[yellow]Rate limited. Waiting {wait_seconds}s...[/yellow]")
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or []
                return data if isinstance(data, list) else []
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                wait_seconds = min(30, 2 ** attempt + 1)
                console.print(f"[yellow]Search request failed. Retrying in {wait_seconds}s...[/yellow]")
                time.sleep(wait_seconds)
        return []

    def references(self, paper_id: str, *, limit: int) -> List[Dict[str, Any]]:
        url = REFERENCES_URL_TEMPLATE.format(paper_id=quote(str(paper_id), safe=":"))
        params = {
            "limit": max(1, min(limit, 1000)),
            "fields": ",".join(REFERENCE_FIELDS),
        }
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    wait_seconds = min(30, 2 ** attempt + 1)
                    console.print(f"[yellow]Rate limited. Waiting {wait_seconds}s...[/yellow]")
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or []
                return data if isinstance(data, list) else []
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                wait_seconds = min(30, 2 ** attempt + 1)
                console.print(f"[yellow]Reference request failed. Retrying in {wait_seconds}s...[/yellow]")
                time.sleep(wait_seconds)
        return []


def output_paths(output_dir: Path, prefix: str) -> Dict[str, Path]:
    return {
        "search_results": output_dir / f"{prefix}_search_results.jsonl",
        "query_outcomes": output_dir / f"{prefix}_query_outcomes.jsonl",
        "seed_candidates": output_dir / f"{prefix}_seed_candidates.json",
        "seed_selection": output_dir / f"{prefix}_seed_selection.json",
        "reference_candidates": output_dir / f"{prefix}_reference_candidates.jsonl",
        "reference_outcomes": output_dir / f"{prefix}_reference_outcomes.jsonl",
        "reference_summary": output_dir / f"{prefix}_reference_summary.json",
        "final_selection": output_dir / f"{prefix}_final_selection.json",
        "run_summary": output_dir / f"{prefix}_run_summary.json",
    }


def ensure_outputs_available(paths: Dict[str, Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing curation outputs. Pass --overwrite. Existing: "
            + ", ".join(existing)
        )


def load_existing_search_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def load_successful_query_outcomes(path: Path) -> List[Dict[str, Any]]:
    outcomes = []
    if not path.exists():
        return outcomes
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                outcome = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(outcome, dict) and outcome.get("status") == "success":
                outcomes.append(outcome)
    return outcomes


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_seed_candidate_outputs(
    *,
    paths: Dict[str, Path],
    run_id: str,
    topic_label: str,
    query_bank_path: Path,
    claim_bank_path: Optional[Path],
    claim_metadata: Optional[Dict[str, Dict[str, Any]]],
    records: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = build_seed_candidates(records, claim_metadata=claim_metadata)
    seed_payload = {
        "run_id": run_id,
        "topic_label": topic_label,
        "created_at": utc_now(),
        "query_bank_path": str(query_bank_path),
        "claim_bank_path": str(claim_bank_path) if claim_bank_path else None,
        "search_results_path": str(paths["search_results"]),
        "scoring_notes": {
            "score_inputs": [
                "query hit count",
                "linked claim coverage",
                "linked disease-area diversity",
                "rank across queries",
                "citation and influential citation counts",
                "reference count",
                "open access flags",
                "abstract presence",
                "recency bonus for papers from 2010 onward",
            ],
            "review_instruction": "Use score for triage only; final corpus IDs should be selected by human review and live pipeline retrieval behavior.",
        },
        **candidates,
    }
    write_json(paths["seed_candidates"], seed_payload)
    return candidates


def capture_search_results(
    *,
    query_bank_path: Path = DEFAULT_QUERY_BANK_PATH,
    claim_bank_path: Optional[Path] = DEFAULT_CLAIM_BANK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_prefix: str = "mendelian_v1",
    limit_override: Optional[int] = None,
    max_queries: Optional[int] = None,
    sleep_seconds: float = 1.0,
    timeout: int = 30,
    max_retries: int = 4,
    overwrite: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    rebuild_candidates: bool = False,
    api_key: Optional[str] = None,
    client: Optional[SemanticScholarSearchClient] = None,
) -> Dict[str, Any]:
    started_at = utc_now()
    run_id = uuid.uuid4().hex[:12]
    query_bank_path = Path(query_bank_path).expanduser().resolve()
    claim_bank_path = Path(claim_bank_path).expanduser().resolve() if claim_bank_path else None
    output_dir = Path(output_dir).expanduser().resolve()
    paths = output_paths(output_dir, output_prefix)
    if resume and overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if not resume and not rebuild_candidates:
        ensure_outputs_available(paths, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_bank = load_json(query_bank_path)
    claim_metadata = load_claim_metadata(claim_bank_path)
    topic_label = str(query_bank.get("topic_label") or output_prefix)
    specs = flatten_query_bank(query_bank, limit_override=limit_override)
    if max_queries is not None:
        specs = specs[:max(0, max_queries)]

    existing_records = (
        load_existing_search_records(paths["search_results"])
        if (resume or rebuild_candidates)
        else []
    )
    existing_success_outcomes = (
        load_successful_query_outcomes(paths["query_outcomes"])
        if (resume or rebuild_candidates)
        else []
    )
    completed_query_ids = {
        str((record.get("query") or {}).get("query_id"))
        for record in existing_records
        if (record.get("query") or {}).get("query_id")
    }
    completed_query_ids.update(
        str(outcome.get("query_id"))
        for outcome in existing_success_outcomes
        if outcome.get("query_id")
    )
    pending_specs = [spec for spec in specs if spec.query_id not in completed_query_ids]

    api_key = api_key if api_key is not None else Config.SEMANTIC_SCHOLAR_API_KEY
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "topic_label": topic_label,
        "query_bank_path": str(query_bank_path),
        "claim_bank_path": str(claim_bank_path) if claim_bank_path else None,
        "claim_metadata_count": len(claim_metadata),
        "output_dir": str(output_dir),
        "outputs": {name: str(path) for name, path in paths.items()},
        "started_at": started_at,
        "completed_at": None,
        "dry_run": dry_run,
        "rebuild_candidates": rebuild_candidates,
        "api_key_present": bool(api_key and str(api_key).strip()),
        "query_count": len(specs),
        "completed_query_count_before_run": len(completed_query_ids),
        "pending_query_count": len(pending_specs),
        "result_row_count": 0,
        "candidate_count": 0,
        "skipped_without_corpus_id": 0,
        "errors": [],
        "queries": [spec.to_provenance() for spec in specs],
    }

    if dry_run:
        summary["completed_at"] = utc_now()
        write_json(paths["run_summary"], summary)
        return summary

    if rebuild_candidates:
        candidates = write_seed_candidate_outputs(
            paths=paths,
            run_id=run_id,
            topic_label=topic_label,
            query_bank_path=query_bank_path,
            claim_bank_path=claim_bank_path,
            claim_metadata=claim_metadata,
            records=existing_records,
        )
        summary["completed_at"] = utc_now()
        summary["result_row_count"] = len(existing_records)
        summary["candidate_count"] = candidates["candidate_count"]
        summary["skipped_without_corpus_id"] = candidates["skipped_without_corpus_id"]
        write_json(paths["run_summary"], summary)
        return summary

    search_client = client or SemanticScholarSearchClient(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )

    records: List[Dict[str, Any]] = list(existing_records)
    file_mode = "a" if resume else "w"
    with (
        open(paths["search_results"], file_mode, encoding="utf-8") as result_handle,
        open(paths["query_outcomes"], file_mode, encoding="utf-8") as outcome_handle,
    ):
        for index, spec in enumerate(pending_specs, start=1):
            console.print(
                f"[cyan]({index}/{len(pending_specs)}) Searching {spec.query_id}: {console_safe(spec.query)}[/cyan]"
            )
            captured_at = utc_now()
            outcome = {
                "run_id": run_id,
                "captured_at": captured_at,
                "topic_label": topic_label,
                "query_set_id": spec.query_set_id,
                "query_id": spec.query_id,
                "query": spec.query,
                "status": "success",
                "result_count": 0,
            }
            try:
                papers = search_client.search(spec.query, limit=spec.max_results)
                outcome["result_count"] = len(papers)
                console.print(f"[green]Found {len(papers)} results for {spec.query_id}[/green]")
            except Exception as exc:
                outcome["status"] = "error"
                outcome["error"] = str(exc)
                error = {
                    "query_id": spec.query_id,
                    "query": spec.query,
                    "error": str(exc),
                    "captured_at": captured_at,
                }
                summary["errors"].append(error)
                console.print(f"[red]Search failed for {spec.query_id}: {console_safe(exc)}[/red]")
                papers = []

            for rank, paper in enumerate(papers, start=1):
                if not isinstance(paper, dict):
                    continue
                record = build_result_record(
                    run_id=run_id,
                    topic_label=topic_label,
                    query_spec=spec,
                    rank=rank,
                    paper=paper,
                    captured_at=captured_at,
                    claim_metadata=claim_metadata,
                )
                records.append(record)
                result_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            outcome_handle.write(json.dumps(outcome, ensure_ascii=True) + "\n")
            result_handle.flush()
            outcome_handle.flush()

            if sleep_seconds > 0 and index < len(pending_specs):
                time.sleep(sleep_seconds)

    candidates = write_seed_candidate_outputs(
        paths=paths,
        run_id=run_id,
        topic_label=topic_label,
        query_bank_path=query_bank_path,
        claim_bank_path=claim_bank_path,
        claim_metadata=claim_metadata,
        records=records,
    )

    summary["completed_at"] = utc_now()
    summary["result_row_count"] = len(records)
    summary["candidate_count"] = candidates["candidate_count"]
    summary["skipped_without_corpus_id"] = candidates["skipped_without_corpus_id"]
    write_json(paths["run_summary"], summary)
    return summary


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def seed_paper_identifier(seed: Dict[str, Any]) -> Optional[str]:
    paper_id = seed.get("paperId")
    if paper_id is not None and str(paper_id).strip():
        return str(paper_id)
    corpus_id = seed.get("corpusId")
    if corpus_id is not None and str(corpus_id).strip():
        return f"CorpusId:{corpus_id}"
    return None


def expand_references(
    *,
    seed_candidates_path: Path = DEFAULT_OUTPUT_DIR / "mendelian_v1_seed_candidates.json",
    claim_bank_path: Optional[Path] = DEFAULT_CLAIM_BANK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_prefix: str = "mendelian_v1",
    release_id: str = "2026-05-26",
    max_seeds: int = 120,
    top_global: int = 40,
    per_disease: int = 8,
    per_claim: int = 5,
    references_per_seed: int = 50,
    min_reference_score: int = 4,
    max_final_corpus_ids: int = 500,
    sleep_seconds: float = 1.0,
    timeout: int = 30,
    max_retries: int = 4,
    resume: bool = False,
    rebuild_only: bool = False,
    api_key: Optional[str] = None,
    client: Optional[SemanticScholarSearchClient] = None,
) -> Dict[str, Any]:
    started_at = utc_now()
    run_id = uuid.uuid4().hex[:12]
    seed_candidates_path = Path(seed_candidates_path).expanduser().resolve()
    claim_bank_path = Path(claim_bank_path).expanduser().resolve() if claim_bank_path else None
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, output_prefix)

    seed_payload = load_json(seed_candidates_path)
    claim_metadata = load_claim_metadata(claim_bank_path)
    topic_label = str(seed_payload.get("topic_label") or output_prefix)
    candidates = seed_payload.get("candidates") or []
    selected_seeds = select_seed_candidates(
        candidates,
        claim_metadata=claim_metadata,
        max_seeds=max_seeds,
        top_global=top_global,
        per_disease=per_disease,
        per_claim=per_claim,
    )
    seed_selection = {
        "run_id": run_id,
        "created_at": utc_now(),
        "topic_label": topic_label,
        "seed_candidates_path": str(seed_candidates_path),
        "claim_bank_path": str(claim_bank_path) if claim_bank_path else None,
        "policy": {
            "max_seeds": max_seeds,
            "top_global": top_global,
            "per_disease": per_disease,
            "per_claim": per_claim,
        },
        "selected_seed_count": len(selected_seeds),
        "selected_seeds": selected_seeds,
    }
    write_json(paths["seed_selection"], seed_selection)

    records = load_jsonl(paths["reference_candidates"]) if (resume or rebuild_only) else []
    reference_outcomes = load_jsonl(paths["reference_outcomes"]) if (resume or rebuild_only) else []
    completed_seed_ids = {
        str((record.get("seed") or {}).get("corpusId"))
        for record in records
        if (record.get("seed") or {}).get("corpusId")
    }
    completed_seed_ids.update(
        str(outcome.get("seed_corpusId"))
        for outcome in reference_outcomes
        if outcome.get("status") == "success" and outcome.get("seed_corpusId")
    )

    api_key = api_key if api_key is not None else Config.SEMANTIC_SCHOLAR_API_KEY
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "topic_label": topic_label,
        "release_id": release_id,
        "started_at": started_at,
        "completed_at": None,
        "api_key_present": bool(api_key and str(api_key).strip()),
        "seed_candidates_path": str(seed_candidates_path),
        "claim_bank_path": str(claim_bank_path) if claim_bank_path else None,
        "selected_seed_count": len(selected_seeds),
        "completed_seed_count_before_run": len(completed_seed_ids),
        "pending_seed_count": 0,
        "reference_records_written": len(records),
        "reference_outcome_count": len(reference_outcomes),
        "reference_candidate_count": 0,
        "final_corpus_id_count": 0,
        "final_author_id_count": 0,
        "references_per_seed": references_per_seed,
        "min_reference_score": min_reference_score,
        "errors": [],
        "outputs": {name: str(path) for name, path in paths.items()},
    }

    if not rebuild_only:
        search_client = client or SemanticScholarSearchClient(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        pending_seeds = [
            seed
            for seed in selected_seeds
            if str(seed.get("corpusId")) not in completed_seed_ids
        ]
        summary["pending_seed_count"] = len(pending_seeds)
        file_mode = "a" if resume else "w"
        if not resume:
            records = []
        with (
            open(paths["reference_candidates"], file_mode, encoding="utf-8") as output,
            open(paths["reference_outcomes"], file_mode, encoding="utf-8") as outcome_output,
        ):
            for index, seed in enumerate(pending_seeds, start=1):
                identifier = seed_paper_identifier(seed)
                seed_title = console_safe(seed.get("title"))
                outcome = {
                    "run_id": run_id,
                    "captured_at": utc_now(),
                    "seed_corpusId": seed.get("corpusId"),
                    "seed_paperId": seed.get("paperId"),
                    "seed_title": seed.get("title"),
                    "status": "success",
                    "result_count": 0,
                    "kept_count": 0,
                }
                if not identifier:
                    outcome["status"] = "error"
                    outcome["error"] = "seed has no paperId or corpusId"
                    summary["errors"].append(
                        {
                            "seed_corpusId": seed.get("corpusId"),
                            "title": seed.get("title"),
                            "error": "seed has no paperId or corpusId",
                            "captured_at": utc_now(),
                        }
                    )
                    outcome_output.write(json.dumps(outcome, ensure_ascii=True) + "\n")
                    outcome_output.flush()
                    continue
                console.print(
                    f"[cyan]({index}/{len(pending_seeds)}) Fetching references for {seed.get('corpusId')}: {seed_title}[/cyan]"
                )
                captured_at = utc_now()
                outcome["captured_at"] = captured_at
                try:
                    references = search_client.references(identifier, limit=references_per_seed)
                    outcome["result_count"] = len(references)
                    console.print(f"[green]Found {len(references)} references for {seed.get('corpusId')}[/green]")
                except Exception as exc:
                    outcome["status"] = "error"
                    outcome["error"] = str(exc)
                    summary["errors"].append(
                        {
                            "seed_corpusId": seed.get("corpusId"),
                            "title": seed.get("title"),
                            "error": str(exc),
                            "captured_at": captured_at,
                        }
                    )
                    console.print(f"[red]Reference expansion failed for {seed.get('corpusId')}: {console_safe(exc)}[/red]")
                    references = []
                kept = 0
                for rank, reference in enumerate(references, start=1):
                    if not isinstance(reference, dict):
                        continue
                    cited_paper = reference.get("citedPaper") if isinstance(reference.get("citedPaper"), dict) else {}
                    relevance = score_reference_relevance(
                        cited_paper,
                        seed,
                        claim_metadata=claim_metadata,
                    )
                    if safe_int(relevance.get("score")) < min_reference_score:
                        continue
                    record = build_reference_record(
                        run_id=run_id,
                        seed=seed,
                        rank=rank,
                        reference=reference,
                        relevance=relevance,
                        captured_at=captured_at,
                    )
                    if not record:
                        continue
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=True) + "\n")
                    kept += 1
                outcome["kept_count"] = kept
                outcome_output.write(json.dumps(outcome, ensure_ascii=True) + "\n")
                output.flush()
                outcome_output.flush()
                console.print(f"[green]Kept {kept} relevant references for {seed.get('corpusId')}[/green]")
                if sleep_seconds > 0 and index < len(pending_seeds):
                    time.sleep(sleep_seconds)

    reference_summary = build_reference_candidate_summary(records)
    reference_payload = {
        "run_id": run_id,
        "topic_label": topic_label,
        "created_at": utc_now(),
        "reference_candidates_path": str(paths["reference_candidates"]),
        "selection_policy": {
            "references_per_seed": references_per_seed,
            "min_reference_score": min_reference_score,
        },
        **reference_summary,
    }
    write_json(paths["reference_summary"], reference_payload)

    final_selection = build_final_selection(
        selected_seeds=selected_seeds,
        reference_candidates=reference_summary["candidates"],
        max_corpus_ids=max_final_corpus_ids,
        release_id=release_id,
        topic_label=topic_label,
    )
    write_json(paths["final_selection"], final_selection)

    summary["completed_at"] = utc_now()
    summary["reference_records_written"] = len(records)
    summary["reference_candidate_count"] = reference_summary["candidate_count"]
    summary["final_corpus_id_count"] = final_selection["corpus_id_count"]
    summary["final_author_id_count"] = final_selection["author_id_count"]
    write_json(paths["reference_summary"], {**reference_payload, "run_summary": summary})
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Semantic Scholar search provenance for a mini corpus.")
    parser.add_argument("--query-bank", default=str(DEFAULT_QUERY_BANK_PATH), help="Path to the query bank JSON.")
    parser.add_argument("--claims-bank", default=str(DEFAULT_CLAIM_BANK_PATH), help="Path to the claim bank JSON used for disease/gene provenance.")
    parser.add_argument("--seed-candidates", default=None, help="Path to seed candidates JSON for --expand-references.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for curation outputs.")
    parser.add_argument("--output-prefix", default="mendelian_v1", help="Filename prefix for output artifacts.")
    parser.add_argument("--release-id", default="2026-05-26", help="Semantic Scholar release ID to record in final selection.")
    parser.add_argument("--limit", type=int, default=None, help="Override max results per query.")
    parser.add_argument("--max-queries", type=int, default=None, help="Run only the first N queries.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between search requests.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout per request in seconds.")
    parser.add_argument("--max-retries", type=int, default=4, help="HTTP retries per search query.")
    parser.add_argument("--api-key", default=None, help="Semantic Scholar API key. Defaults to SEMANTIC_SCHOLAR_API_KEY from env_vars.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--resume", action="store_true", help="Append to existing search results and skip query IDs already captured.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write only the run summary.")
    parser.add_argument("--rebuild-candidates", action="store_true", help="Rebuild seed candidates from existing search results without making API calls.")
    parser.add_argument("--expand-references", action="store_true", help="Run deterministic seed selection and bounded outbound reference expansion.")
    parser.add_argument("--rebuild-reference-summary", action="store_true", help="Rebuild reference summary/final selection from existing reference candidates without API calls.")
    parser.add_argument("--max-seeds", type=int, default=120, help="Maximum selected seed papers for reference expansion.")
    parser.add_argument("--top-global-seeds", type=int, default=40, help="Top global seed candidates to include.")
    parser.add_argument("--per-disease-seeds", type=int, default=8, help="Top seed candidates per disease area to include.")
    parser.add_argument("--per-claim-seeds", type=int, default=5, help="Top seed candidates per claim to include.")
    parser.add_argument("--references-per-seed", type=int, default=50, help="Maximum outbound references fetched per selected seed.")
    parser.add_argument("--min-reference-score", type=int, default=4, help="Minimum relevance score for keeping an outbound reference.")
    parser.add_argument("--max-final-corpus-ids", type=int, default=500, help="Maximum corpus IDs in final selection.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.expand_references or args.rebuild_reference_summary:
            seed_candidates_path = (
                Path(args.seed_candidates)
                if args.seed_candidates
                else Path(args.output_dir) / f"{args.output_prefix}_seed_candidates.json"
            )
            summary = expand_references(
                seed_candidates_path=seed_candidates_path,
                claim_bank_path=Path(args.claims_bank) if args.claims_bank else None,
                output_dir=Path(args.output_dir),
                output_prefix=args.output_prefix,
                release_id=args.release_id,
                max_seeds=args.max_seeds,
                top_global=args.top_global_seeds,
                per_disease=args.per_disease_seeds,
                per_claim=args.per_claim_seeds,
                references_per_seed=args.references_per_seed,
                min_reference_score=args.min_reference_score,
                max_final_corpus_ids=args.max_final_corpus_ids,
                sleep_seconds=args.sleep,
                timeout=args.timeout,
                max_retries=args.max_retries,
                resume=args.resume,
                rebuild_only=args.rebuild_reference_summary,
                api_key=args.api_key,
            )
            console.print(
                "[green]Reference expansion complete:[/green] "
                f"{summary['selected_seed_count']} seeds, "
                f"{summary['reference_records_written']} reference rows, "
                f"{summary['reference_candidate_count']} reference candidates, "
                f"{summary['final_corpus_id_count']} final corpus IDs"
            )
            console.print(f"[cyan]Outputs:[/cyan] {summary['outputs']}")
            return 0

        summary = capture_search_results(
            query_bank_path=Path(args.query_bank),
            claim_bank_path=Path(args.claims_bank) if args.claims_bank else None,
            output_dir=Path(args.output_dir),
            output_prefix=args.output_prefix,
            limit_override=args.limit,
            max_queries=args.max_queries,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
            resume=args.resume,
            dry_run=args.dry_run,
            rebuild_candidates=args.rebuild_candidates,
            api_key=args.api_key,
        )
    except Exception as exc:
        console.print(f"[red]Curator failed: {console_safe(exc)}[/red]")
        return 1

    console.print(
        "[green]Curator complete:[/green] "
        f"{summary['query_count']} queries, "
        f"{summary['result_row_count']} search rows, "
        f"{summary['candidate_count']} seed candidates"
    )
    console.print(f"[cyan]Outputs:[/cyan] {summary['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
