import asyncio
import sys
import types


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault(
    "openai",
    types.SimpleNamespace(
        OpenAI=object,
        AsyncOpenAI=object,
        AsyncAzureOpenAI=object,
    ),
)

if "aiofiles" not in sys.modules:
    aiofiles_module = types.ModuleType("aiofiles")
    aiofiles_module.__path__ = []

    async def _unsupported_open(*args, **kwargs):  # pragma: no cover - import stub only
        raise RuntimeError("aiofiles stub should not be used in this test")

    aiofiles_module.open = _unsupported_open
    sys.modules["aiofiles"] = aiofiles_module

if "aiofiles.os" not in sys.modules:
    aiofiles_os_module = types.ModuleType("aiofiles.os")
    aiofiles_os_module.path = types.SimpleNamespace(exists=lambda *args, **kwargs: False)

    async def _noop(*args, **kwargs):  # pragma: no cover - import stub only
        return None

    aiofiles_os_module.remove = _noop
    sys.modules["aiofiles.os"] = aiofiles_os_module
    sys.modules["aiofiles"].os = aiofiles_os_module

from app.services.llm.gateway import LLMTask
from app.services.llm.types import empty_usage
import processor as processor_module


class DummyAIService:
    default_model = "test-model"

    def __init__(self):
        self.issues = []

    async def add_issue(self, batch_id, claim_id, severity, stage, message, details):
        self.issues.append(
            {
                "batch_id": batch_id,
                "claim_id": claim_id,
                "severity": severity,
                "stage": stage,
                "message": message,
                "details": details,
            }
        )

    async def get_claim_issues(self, batch_id, claim_id):
        return [issue for issue in self.issues if issue["batch_id"] == batch_id and issue["claim_id"] == claim_id]

    async def build_debug_trace(self, batch_id, claim_id):
        return {"summary": {"llm_calls": 0, "models_used": [self.default_model], "retries": 0}}


def build_processor(max_stage_retries=1):
    ai_service = DummyAIService()
    saved_claims = []

    processor = processor_module.ValsciProcessor.__new__(processor_module.ValsciProcessor)
    processor.s2_searcher = types.SimpleNamespace()
    processor.paper_analyzer = types.SimpleNamespace()
    processor.evidence_scorer = types.SimpleNamespace()
    processor.claim_processor = types.SimpleNamespace(
        _format_non_relevant_papers=lambda papers: papers,
        _format_inaccessible_papers=lambda papers: papers,
    )
    processor.claim_store = types.SimpleNamespace()
    processor.gateway_factory = types.SimpleNamespace(get_gateway=lambda snapshot: ai_service)
    processor.ai_service = ai_service
    processor.email_service = types.SimpleNamespace()
    processor.claims_in_memory = {}
    processor.claims_query_generation_in_progress = set()
    processor.claims_searching_in_progress = set()
    processor.papers_analyzing_in_progress = set()
    processor.papers_scoring_in_progress = set()
    processor.claims_final_reporting_in_progress = set()
    processor.claim_token_usage = {}
    processor.max_tokens_per_claim = 1_000_000
    processor.request_token_estimates = []
    processor.max_tokens_per_window = 1_000_000
    processor.max_requests_per_window = 100
    processor.window_size_seconds = 60
    processor.last_token_update_time = 0
    processor.model = "test-model"
    processor._active_locks = set()
    processor._claim_stage_retries = {}
    processor._paper_stage_retries = {}
    processor.max_stage_retries = max_stage_retries

    async def fake_save_processed_claim(claim_data, batch_id, claim_id):
        saved_claims.append(
            {
                "batch_id": batch_id,
                "claim_id": claim_id,
                "claim_data": claim_data,
            }
        )

    processor._save_processed_claim = fake_save_processed_claim
    return processor, ai_service, saved_claims


def test_search_papers_applies_retry_cap_and_terminal_report():
    processor, ai_service, saved_claims = build_processor(max_stage_retries=1)
    claim_data = {
        "text": "Creatine improves memory.",
        "status": "ready_for_search",
        "semantic_scholar_queries": ["creatine memory"],
        "search_config": {"results_per_query": 5},
        "usage": empty_usage(),
        "usage_by_stage": {},
        "provider_snapshot": {},
    }
    processor.claims_in_memory[("batch-a", "claim-a")] = claim_data
    processor.claims_searching_in_progress.add("claim-a")

    async def failing_search(*args, **kwargs):
        raise RuntimeError("semantic scholar unavailable")

    processor.s2_searcher.search_papers_for_claim = failing_search

    asyncio.run(processor.search_papers(claim_data, "batch-a", "claim-a"))
    assert claim_data["status"] == "ready_for_search"
    assert saved_claims == []
    assert len(ai_service.issues) == 1

    processor.claims_searching_in_progress.add("claim-a")
    asyncio.run(processor.search_papers(claim_data, "batch-a", "claim-a"))

    assert claim_data["status"] == "processed"
    assert len(saved_claims) == 1
    assert "after all retries" in saved_claims[0]["claim_data"]["report"]["explanation"]
    assert ("batch-a", "claim-a", "paper_search") not in processor._claim_stage_retries


def test_failed_analysis_paper_is_terminal_for_analyze_claim(monkeypatch):
    processor, ai_service, _ = build_processor(max_stage_retries=1)
    raw_paper = {
        "corpusId": 101,
        "title": "Paper 101",
        "content": "Full text",
        "content_type": "s2orc",
    }
    claim_data = {
        "text": "Claim text",
        "status": "ready_for_analysis",
        "raw_papers": [raw_paper],
        "processed_papers": [],
        "non_relevant_papers": [],
        "inaccessible_papers": [],
        "failed_papers": [],
        "semantic_scholar_queries": ["query"],
        "usage": empty_usage(),
        "usage_by_stage": {},
        "provider_snapshot": {},
    }
    processor.claims_in_memory[("batch-b", "claim-b")] = claim_data

    async def failing_analysis(*args, **kwargs):
        raise RuntimeError("analysis failed")

    processor.paper_analyzer.analyze_relevance_and_extract = failing_analysis

    asyncio.run(processor.analyze_single_paper(raw_paper, claim_data["text"], "batch-b", "claim-b"))
    assert claim_data["failed_papers"] == []

    asyncio.run(processor.analyze_single_paper(raw_paper, claim_data["text"], "batch-b", "claim-b"))
    assert len(claim_data["failed_papers"]) == 1
    assert claim_data["failed_papers"][0]["stage"] == LLMTask.PAPER_ANALYSIS

    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return types.SimpleNamespace()

    monkeypatch.setattr(processor_module.asyncio, "create_task", fake_create_task)
    processor.s2_searcher.get_paper_content = lambda corpus_id: (_ for _ in ()).throw(AssertionError("failed paper was rescheduled"))

    asyncio.run(processor.analyze_claim(claim_data, "batch-b", "claim-b"))

    assert len(scheduled) == 1
    assert "claim-b" in processor.claims_final_reporting_in_progress
    assert len(ai_service.issues) == 2


def test_failed_scoring_is_terminal_and_allows_final_report(monkeypatch):
    processor, ai_service, _ = build_processor(max_stage_retries=1)
    processed_paper = {
        "paper": {"corpusId": 202, "title": "Paper 202"},
        "relevance": 0.9,
        "excerpts": ["Excerpt"],
        "explanations": ["Explanation"],
        "score": -1,
        "score_status": "pending",
        "content_type": "s2orc",
    }
    claim_data = {
        "text": "Claim text",
        "status": "ready_for_analysis",
        "raw_papers": [{"corpusId": 202, "title": "Paper 202"}],
        "processed_papers": [processed_paper],
        "non_relevant_papers": [],
        "inaccessible_papers": [],
        "failed_papers": [],
        "semantic_scholar_queries": ["query"],
        "bibliometric_config": {"use_bibliometrics": True},
        "usage": empty_usage(),
        "usage_by_stage": {},
        "provider_snapshot": {},
    }
    processor.claims_in_memory[("batch-c", "claim-c")] = claim_data
    processor.claim_token_usage["claim-c"] = 0

    async def failing_score(*args, **kwargs):
        raise RuntimeError("scoring failed")

    processor.evidence_scorer.calculate_paper_weight = failing_score

    asyncio.run(processor.score_paper(processed_paper, "batch-c", "claim-c"))
    assert processed_paper["score"] == -1

    asyncio.run(processor.score_paper(processed_paper, "batch-c", "claim-c"))
    assert processed_paper["score"] == 0.0
    assert processed_paper["score_status"] == "failed"
    assert len(claim_data["failed_papers"]) == 1
    assert claim_data["failed_papers"][0]["stage"] == LLMTask.VENUE_SCORING

    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return types.SimpleNamespace()

    monkeypatch.setattr(processor_module.asyncio, "create_task", fake_create_task)
    processor.s2_searcher.get_paper_content = lambda corpus_id: (_ for _ in ()).throw(AssertionError("processed paper was fetched again"))

    asyncio.run(processor.analyze_claim(claim_data, "batch-c", "claim-c"))

    assert len(scheduled) == 1
    assert "claim-c" in processor.claims_final_reporting_in_progress
    assert len(ai_service.issues) == 2
