import asyncio
import json
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
from app.services.claim_store import ClaimStore
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


def test_generate_search_queries_stops_at_query_generation_checkpoint():
    processor, _, saved_claims = build_processor(max_stage_retries=1)
    claim_data = {
        "text": "Creatine improves memory.",
        "status": "queued",
        "stop_after": "query_generation",
        "search_config": {"num_queries": 2},
        "usage": empty_usage(),
        "usage_by_stage": {},
        "provider_snapshot": {},
    }
    processor.claims_in_memory[("batch-q", "claim-q")] = claim_data
    processor.claims_query_generation_in_progress.add("claim-q")

    async def fake_generate(*args, **kwargs):
        return ["creatine memory", "memory trial"], empty_usage()

    processor.s2_searcher.generate_search_queries = fake_generate

    asyncio.run(processor.generate_search_queries(claim_data, "batch-q", "claim-q"))

    assert len(saved_claims) == 1
    assert claim_data["status"] == "processed"
    assert claim_data["completed_stage"] == LLMTask.QUERY_GENERATION
    assert claim_data["is_stage_checkpoint"] is True


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


def test_analyze_claim_stops_at_paper_analysis_checkpoint(monkeypatch):
    processor, _, saved_claims = build_processor(max_stage_retries=1)
    claim_data = {
        "text": "Claim text",
        "status": "ready_for_analysis",
        "stop_after": "paper_analysis",
        "raw_papers": [{"corpusId": 301, "title": "Paper 301"}],
        "processed_papers": [
            {
                "paper": {"corpusId": 301, "title": "Paper 301"},
                "relevance": 0.8,
                "excerpts": ["Excerpt"],
                "explanations": ["Explanation"],
                "score": -1,
                "score_status": "pending",
            }
        ],
        "non_relevant_papers": [],
        "inaccessible_papers": [],
        "failed_papers": [],
        "semantic_scholar_queries": ["query"],
        "usage": empty_usage(),
        "usage_by_stage": {},
        "provider_snapshot": {},
    }
    processor.claims_in_memory[("batch-pa", "claim-pa")] = claim_data

    def fail_create_task(coro):
        coro.close()
        raise AssertionError("venue scoring should not be scheduled for a paper-analysis checkpoint")

    monkeypatch.setattr(processor_module.asyncio, "create_task", fail_create_task)
    processor.s2_searcher.get_paper_content = lambda corpus_id: None

    asyncio.run(processor.analyze_claim(claim_data, "batch-pa", "claim-pa"))

    assert len(saved_claims) == 1
    assert claim_data["completed_stage"] == LLMTask.PAPER_ANALYSIS
    assert claim_data["is_stage_checkpoint"] is True


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


def test_analyze_claim_stops_at_venue_scoring_checkpoint(monkeypatch):
    processor, _, saved_claims = build_processor(max_stage_retries=1)
    claim_data = {
        "text": "Claim text",
        "status": "ready_for_analysis",
        "stop_after": "venue_scoring",
        "raw_papers": [{"corpusId": 401, "title": "Paper 401"}],
        "processed_papers": [
            {
                "paper": {"corpusId": 401, "title": "Paper 401"},
                "relevance": 0.9,
                "excerpts": ["Excerpt"],
                "explanations": ["Explanation"],
                "score": 0.7,
                "score_status": "completed",
            }
        ],
        "non_relevant_papers": [],
        "inaccessible_papers": [],
        "failed_papers": [],
        "semantic_scholar_queries": ["query"],
        "usage": empty_usage(),
        "usage_by_stage": {},
        "provider_snapshot": {},
    }
    processor.claims_in_memory[("batch-vs", "claim-vs")] = claim_data

    def fail_create_task(coro):
        coro.close()
        raise AssertionError("final report should not be scheduled for a venue-scoring checkpoint")

    monkeypatch.setattr(processor_module.asyncio, "create_task", fail_create_task)
    processor.s2_searcher.get_paper_content = lambda corpus_id: None

    asyncio.run(processor.analyze_claim(claim_data, "batch-vs", "claim-vs"))

    assert len(saved_claims) == 1
    assert claim_data["completed_stage"] == LLMTask.VENUE_SCORING
    assert claim_data["is_stage_checkpoint"] is True


def test_recover_waiting_reuse_runs_materializes_orphans_from_saved_baseline(tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    state_dir = tmp_path / "state"
    arena_id = "arena-recovery"
    claim_text = "Creatine improves memory."

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    claim_record, _ = store.get_or_create_claim(claim_text, batch_tags=[arena_id])
    baseline_run = store.create_run(
        claim_record=claim_record,
        batch_tags=[arena_id],
        arena_id=arena_id,
        execution_mode="reuse_retrieval",
        provider_snapshot={"provider_id": "baseline"},
        cost_confirmation={"accepted": True},
        transport_batch_id=arena_id,
        review_type="regular",
        status="queued",
        source="arena",
    )
    waiting_run = store.create_run(
        claim_record=claim_record,
        batch_tags=[arena_id],
        arena_id=arena_id,
        execution_mode="reuse_retrieval",
        provider_snapshot={"provider_id": "candidate-1"},
        cost_confirmation={"accepted": True},
        transport_batch_id=arena_id,
        review_type="regular",
        status="waiting_for_baseline",
        source="arena",
    )
    waiting_run["reuse_from_run_id"] = baseline_run["run_id"]
    store.save_run(waiting_run)

    baseline_claim_data = {
        "text": claim_text,
        "status": "processed",
        "batch_id": arena_id,
        "claim_id": baseline_run["run_id"],
        "run_id": baseline_run["run_id"],
        "claim_key": claim_record["claim_key"],
        "arena_id": arena_id,
        "review_type": "regular",
        "execution_mode": "reuse_retrieval",
        "stop_after": "final_report",
        "completed_stage": "final_report",
        "is_stage_checkpoint": False,
        "provider_snapshot": {"provider_id": "baseline"},
        "model_overrides": {},
        "search_config": {"num_queries": 5, "results_per_query": 5},
        "bibliometric_config": {"use_bibliometrics": True},
        "semantic_scholar_queries": ["creatine memory"],
        "raw_papers": [{"corpusId": 101, "title": "Paper 101"}],
        "processed_papers": [
            {
                "paper": {"corpusId": 101, "title": "Paper 101"},
                "relevance": 0.8,
                "excerpts": ["Excerpt"],
                "explanations": ["Explanation"],
                "score": 0.7,
                "score_status": "completed",
            }
        ],
        "non_relevant_papers": [{"paper": {"corpusId": 202, "title": "Paper 202"}}],
        "usage": empty_usage(),
        "usage_by_stage": {},
        "report": {"claimRating": 4, "explanation": "Baseline report"},
    }
    baseline_path = saved_jobs_dir / arena_id / f"{baseline_run['run_id']}.txt"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline_claim_data, indent=2), encoding="utf-8")

    processor, _, _ = build_processor(max_stage_retries=1)
    processor.claim_store = store

    recovered = processor.recover_waiting_reuse_runs()

    assert recovered == 1
    recovered_run = store.get_run(waiting_run["run_id"])
    assert recovered_run["status"] == "ready_for_analysis"

    queued_path = queued_jobs_dir / arena_id / f"{waiting_run['run_id']}.txt"
    assert queued_path.exists()
    queued_payload = json.loads(queued_path.read_text(encoding="utf-8"))
    assert queued_payload["status"] == "ready_for_analysis"
    assert queued_payload["raw_papers"] == baseline_claim_data["raw_papers"]
    assert queued_payload["processed_papers"] == []
    assert queued_payload["non_relevant_papers"] == []
    assert queued_payload["report"] is None
