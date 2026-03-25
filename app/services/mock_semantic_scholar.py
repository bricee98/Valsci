"""Deterministic Semantic Scholar fixtures for local UI and integration testing."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm.types import empty_usage
from app.services.llm.trace_store import TraceStore


def _author(name: str, author_id: str, h_index: int) -> Dict[str, Any]:
    return {
        "name": name,
        "authorId": author_id,
        "hIndex": h_index,
    }


def _paper(
    corpus_id: int,
    *,
    title: str,
    abstract: str,
    venue: str,
    year: int,
    citation_count: int,
    authors: List[Dict[str, Any]],
    url: Optional[str] = None,
    content: Optional[str] = None,
    content_source: str = "mock-fulltext",
    accessible: bool = True,
    reason_code: str = "mock_inaccessible",
    reason: str = "Mock fixture marks this paper as inaccessible.",
    delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    return {
        "paperId": f"mock-{corpus_id}",
        "corpusId": corpus_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": authors,
        "venue": venue,
        "url": url or f"https://example.org/papers/{corpus_id}",
        "isOpenAccess": accessible,
        "fieldsOfStudy": ["Medicine", "Computer Science"],
        "citationCount": citation_count,
        "mock_content": content,
        "mock_content_source": content_source,
        "mock_accessible": accessible,
        "mock_reason_code": reason_code,
        "mock_reason": reason,
        "mock_delay_seconds": delay_seconds,
    }


LONG_EVIDENCE_TEXT = (
    "This mock paper intentionally contains a very long body to exercise truncation, "
    "context budgeting, and side-by-side review. "
    + "Evidence paragraph. " * 1200
)


MOCK_FIXTURE_PACKS: Dict[str, Dict[str, Any]] = {
    "happy_path": {
        "label": "Plausible Claims Demo",
        "description": "Two claims with retrieval success and clear supporting evidence.",
        "claims": [
            "Creatine supplementation improves short-term memory in healthy adults.",
            "Magnesium improves subjective sleep quality in adults with poor sleep.",
        ],
    },
    "edge_cases": {
        "label": "False Claims Demo",
        "description": "Covers no results, inaccessible content, mixed evidence, slow retrieval, and truncation.",
        "claims": [
            "Quantum socks cure migraines overnight.",
            "Blue light blocking glasses improve sleep latency in adults.",
            "Omega-3 supplements eliminate depression in all adults.",
        ],
    },
}


CLAIM_FIXTURES: Dict[str, Dict[str, Any]] = {
    "Creatine supplementation improves short-term memory in healthy adults.": {
        "queries": [
            "creatine supplementation short term memory healthy adults randomized trial",
            "creatine cognitive performance healthy adults placebo controlled",
            "creatine memory meta analysis healthy adults",
        ],
        "papers": [
            _paper(
                1001,
                title="Creatine supplementation and working memory performance in healthy adults",
                abstract="A randomized trial showing modest improvement in short-term memory after creatine supplementation.",
                venue="Journal of Cognitive Nutrition",
                year=2022,
                citation_count=84,
                authors=[_author("Mina Torres", "a-1001", 29), _author("Paul Choi", "a-1002", 24)],
                content=(
                    "Methods: participants received creatine or placebo for six weeks. "
                    "Results: the creatine group improved on digit span and delayed recall tests. "
                    "Mechanistically, phosphocreatine buffering may support neuronal energy use during demanding tasks."
                ),
            ),
            _paper(
                1002,
                title="Meta-analysis of creatine and cognitive performance",
                abstract="A synthesis showing benefits strongest in memory tasks under cognitive load.",
                venue="Nutritional Neuroscience Review",
                year=2023,
                citation_count=112,
                authors=[_author("Aria Patel", "a-1003", 34), _author("Jon Meyer", "a-1004", 31)],
                content=(
                    "Across controlled studies, creatine showed the clearest gains for short-term memory and working memory. "
                    "Effect sizes were moderate and heterogeneous but directionally positive."
                ),
            ),
        ],
    },
    "Magnesium improves subjective sleep quality in adults with poor sleep.": {
        "queries": [
            "magnesium poor sleep adults sleep quality trial",
            "magnesium supplementation subjective sleep quality adults",
            "magnesium insomnia adults randomized placebo",
        ],
        "papers": [
            _paper(
                1101,
                title="Magnesium supplementation and sleep quality in adults with poor sleep",
                abstract="Participants reported modest improvements in sleep quality and reduced awakenings.",
                venue="Sleep Health Reports",
                year=2021,
                citation_count=61,
                authors=[_author("Sara Kim", "a-1101", 22), _author("Leah Grant", "a-1102", 19)],
                content=(
                    "Adults with poor sleep receiving magnesium reported better subjective sleep quality and fewer nighttime awakenings. "
                    "Objective measures were mixed, but questionnaire-based improvement was consistent."
                ),
            ),
        ],
    },
    "Quantum socks cure migraines overnight.": {
        "queries": [
            "quantum socks migraine overnight cure",
            "quantum textile migraine treatment trial",
            "quantum socks headache placebo",
        ],
        "papers": [],
    },
    "Blue light blocking glasses improve sleep latency in adults.": {
        "queries": [
            "blue light blocking glasses sleep latency adults randomized",
            "amber lenses sleep onset latency adults",
            "blue blocking glasses mixed evidence sleep adults",
        ],
        "papers": [
            _paper(
                1201,
                title="Evening blue-light blocking glasses and sleep onset latency",
                abstract="A small study showing faster sleep onset when participants wore amber lenses before bed.",
                venue="Chronobiology Practice",
                year=2020,
                citation_count=45,
                authors=[_author("Nadia Brooks", "a-1201", 17), _author("Evan Holt", "a-1202", 21)],
                content=(
                    "Participants wearing blue-light blocking glasses before bedtime fell asleep modestly faster than controls. "
                    "Benefits were strongest in participants with heavy evening screen exposure."
                ),
                delay_seconds=1.2,
            ),
            _paper(
                1202,
                title="Blue-light blocking lenses and sleep outcomes: null findings in a crossover study",
                abstract="A controlled crossover study that did not find a significant change in sleep latency.",
                venue="Behavioral Sleep Journal",
                year=2021,
                citation_count=38,
                authors=[_author("Rita Owens", "a-1203", 16), _author("Cam Ortiz", "a-1204", 18)],
                content=(
                    "The crossover trial did not observe a statistically significant reduction in sleep latency. "
                    "Authors noted substantial participant-level variability."
                ),
            ),
            _paper(
                1203,
                title="Observational note on evening eyewear habits and sleep",
                abstract="The paper metadata is available but content is intentionally inaccessible in mock mode.",
                venue="Digital Health Notes",
                year=2019,
                citation_count=9,
                authors=[_author("Jules Park", "a-1205", 11)],
                accessible=False,
                content=None,
                reason_code="mock_pdf_missing",
                reason="Mock fixture simulates inaccessible paper content.",
            ),
        ],
    },
    "Omega-3 supplements eliminate depression in all adults.": {
        "queries": [
            "omega 3 depression adults broad effect trial",
            "omega 3 supplements depression meta analysis adults",
            "omega 3 depression severe truncation mock",
        ],
        "papers": [
            _paper(
                1301,
                title="Omega-3 supplementation and depressive symptoms in adults",
                abstract="A mixed-evidence trial showing improvement in some subgroups rather than universal benefit.",
                venue="Affective Science Review",
                year=2024,
                citation_count=53,
                authors=[_author("Iris Nguyen", "a-1301", 27), _author("Malik Stone", "a-1302", 25)],
                content=LONG_EVIDENCE_TEXT,
            ),
        ],
    },
}


def available_mock_claim_sets() -> List[Dict[str, Any]]:
    return [
        {
            "pack_id": pack_id,
            "label": pack["label"],
            "description": pack["description"],
            "claims": list(pack["claims"]),
        }
        for pack_id, pack in MOCK_FIXTURE_PACKS.items()
    ]


class MockSemanticScholarSearcher:
    def __init__(self, *, fixture_pack: str = "happy_path", delay_seconds: float = 0.0, trace_root: str = "saved_jobs"):
        self.fixture_pack = fixture_pack if fixture_pack in MOCK_FIXTURE_PACKS else "happy_path"
        self.delay_seconds = max(float(delay_seconds or 0.0), 0.0)
        self._query_cache: Dict[Tuple[str, ...], str] = {}
        self._trace_store = TraceStore(root_dir=trace_root, enabled=True)

    def _claim_fixture(self, claim_text: str) -> Dict[str, Any]:
        fixture = CLAIM_FIXTURES.get(claim_text)
        if fixture:
            return fixture
        digest = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()[:8]
        return {
            "queries": [
                f"{claim_text} evidence",
                f"{claim_text} randomized trial",
                f"{claim_text} mechanism {digest}",
            ],
            "papers": [],
        }

    async def generate_search_queries(
        self,
        claim_text: str,
        num_queries: int = 5,
        ai_service=None,
        batch_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        await asyncio.sleep(self.delay_seconds)
        queries = list(self._claim_fixture(claim_text)["queries"])[: max(int(num_queries or 1), 1)]
        if len(queries) < num_queries:
            queries.extend(
                f"{claim_text} mock query {index}"
                for index in range(len(queries) + 1, num_queries + 1)
            )
        self._query_cache[tuple(queries)] = claim_text
        if batch_id and claim_id:
            await self._trace_store.append(batch_id, claim_id, {
                "trace_id": uuid.uuid4().hex[:12],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": "query_generation",
                "task": "generate_search_queries",
                "status": "mock",
                "mock_mode": True,
                "fixture_pack": self.fixture_pack,
                "model": "mock (no LLM call)",
                "input_summary": f"Claim: {claim_text[:100]}",
                "output_summary": f"{len(queries)} queries generated from mock fixtures",
                "queries": queries,
                "latency_ms": int(self.delay_seconds * 1000),
                "tokens": {"input": 0, "output": 0, "total": 0},
                "cost_usd": 0.0,
            })
        return queries, empty_usage()

    async def search_papers_for_claim(
        self,
        queries: List[str],
        results_per_query: int = 5,
        batch_id: Optional[str] = None,
        claim_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        await asyncio.sleep(self.delay_seconds)
        claim_text = self._query_cache.get(tuple(queries))
        if not claim_text:
            claim_text = next(
                (
                    text
                    for text, fixture in CLAIM_FIXTURES.items()
                    if any(query in fixture["queries"] for query in queries)
                ),
                "",
            )
        fixture = self._claim_fixture(claim_text)
        papers = [deepcopy(paper) for paper in fixture.get("papers", [])]
        result = papers[: max(int(results_per_query or 1) * max(len(queries), 1), 1)]
        if batch_id and claim_id:
            await self._trace_store.append(batch_id, claim_id, {
                "trace_id": uuid.uuid4().hex[:12],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": "paper_search",
                "task": "search_papers_for_claim",
                "status": "mock",
                "mock_mode": True,
                "fixture_pack": self.fixture_pack,
                "model": "mock (Semantic Scholar bypassed)",
                "input_summary": f"{len(queries)} queries, {results_per_query} results/query",
                "output_summary": f"{len(result)} papers returned from mock fixtures",
                "latency_ms": int(self.delay_seconds * 1000),
                "tokens": {"input": 0, "output": 0, "total": 0},
                "cost_usd": 0.0,
            })
        return result

    def get_paper_content(self, corpus_id: str) -> Dict[str, Any]:
        for fixture in CLAIM_FIXTURES.values():
            for paper in fixture.get("papers", []):
                if str(paper.get("corpusId")) != str(corpus_id):
                    continue
                if paper.get("mock_delay_seconds"):
                    time.sleep(float(paper["mock_delay_seconds"]))
                if not paper.get("mock_accessible", True):
                    return {
                        "text": None,
                        "source": None,
                        "pdf_hash": None,
                        "status": "inaccessible",
                        "reason_code": paper.get("mock_reason_code", "mock_inaccessible"),
                        "reason": paper.get("mock_reason", "Mock fixture marked this paper inaccessible."),
                        "lookup_details": {
                            "corpus_id": str(corpus_id),
                            "fixture_pack": self.fixture_pack,
                            "attempts": [{"dataset": "mock", "status": "inaccessible"}],
                        },
                    }
                content = str(paper.get("mock_content") or paper.get("abstract") or "")
                return {
                    "text": content,
                    "source": paper.get("mock_content_source", "mock"),
                    "pdf_hash": None,
                    "status": "ok",
                    "lookup_details": {
                        "corpus_id": str(corpus_id),
                        "fixture_pack": self.fixture_pack,
                        "attempts": [{"dataset": "mock", "status": "found_text"}],
                    },
                }
        return {
            "text": None,
            "source": None,
            "pdf_hash": None,
            "status": "inaccessible",
            "reason_code": "mock_missing_corpus",
            "reason": "Mock fixture does not contain this corpus ID.",
            "lookup_details": {
                "corpus_id": str(corpus_id),
                "fixture_pack": self.fixture_pack,
                "attempts": [{"dataset": "mock", "status": "missing_record"}],
            },
        }
