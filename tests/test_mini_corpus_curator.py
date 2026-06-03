import json
from pathlib import Path

from semantic_scholar.utils.mini_corpus_curator import (
    build_seed_candidates,
    capture_search_results,
    expand_references,
    flatten_query_bank,
    select_seed_candidates,
)


def _query_bank():
    return {
        "topic_label": "mendelian_disease_v1",
        "retrieval_policy": {"search_results_per_query": 5},
        "query_sets": [
            {
                "set_id": "set_b",
                "priority": 2,
                "description": "Later set",
                "queries": [
                    {
                        "query_id": "Q-B",
                        "query": "Wilson disease ATP7B",
                        "intent": "Carrier nuance",
                        "linked_claim_ids": ["MENDEL-012"],
                    }
                ],
            },
            {
                "set_id": "set_a",
                "priority": 1,
                "description": "Earlier set",
                "queries": [
                    {
                        "query_id": "Q-A",
                        "query": "cystic fibrosis CFTR modifier genes",
                        "intent": "Modifier evidence",
                        "linked_claim_ids": ["MENDEL-002"],
                        "max_results": 3,
                    }
                ],
            },
        ],
    }


class FakeSearchClient:
    def __init__(self):
        self.calls = []

    def search(self, query, *, limit):
        self.calls.append((query, limit))
        if "cystic fibrosis" in query:
            return [
                {
                    "paperId": "paper-cf-1",
                    "corpusId": 101,
                    "title": "CFTR modifier genes and lung disease",
                    "abstract": "Modifier gene evidence.",
                    "year": 2015,
                    "authors": [{"authorId": "a1", "name": "Ada Author"}],
                    "venue": "Genetics",
                    "url": "https://example.org/cf",
                    "isOpenAccess": True,
                    "openAccessPdf": {"url": "https://example.org/cf.pdf", "status": "GREEN"},
                    "fieldsOfStudy": ["Medicine"],
                    "s2FieldsOfStudy": [{"category": "Medicine"}],
                    "citationCount": 120,
                    "influentialCitationCount": 12,
                    "referenceCount": 30,
                },
                {
                    "paperId": "paper-no-corpus",
                    "title": "Missing corpus id",
                    "citationCount": 1,
                },
            ]
        return [
            {
                "paperId": "paper-cf-1",
                "corpusId": 101,
                "title": "CFTR modifier genes and lung disease",
                "year": 2015,
                "authors": [],
                "venue": "Genetics",
                "isOpenAccess": False,
                "citationCount": 90,
                "influentialCitationCount": 10,
                "referenceCount": 28,
            },
            {
                "paperId": "paper-wilson-1",
                "corpusId": 202,
                "title": "ATP7B carrier status in Wilson disease",
                "abstract": "Carrier status is not diagnostic.",
                "year": 2008,
                "authors": [{"authorId": "a2", "name": "Ben Author"}],
                "venue": "Hepatology",
                "isOpenAccess": False,
                "citationCount": 40,
                "influentialCitationCount": 4,
                "referenceCount": 18,
            },
        ]

    def references(self, paper_id, *, limit):
        self.calls.append((f"references:{paper_id}", limit))
        return [
            {
                "contexts": ["CFTR modifier background"],
                "intents": ["background"],
                "isInfluential": True,
                "citedPaper": {
                    "paperId": "paper-ref-1",
                    "corpusId": 303,
                    "title": "Cystic fibrosis CFTR modifier gene review",
                    "abstract": "CFTR genotype phenotype modifier evidence.",
                    "year": 2018,
                    "authors": [
                        {"authorId": "a3", "name": "Cora Author"},
                        {"authorId": "a4", "name": "Drew Author"},
                    ],
                    "venue": "Genetics",
                    "isOpenAccess": True,
                    "citationCount": 200,
                    "influentialCitationCount": 20,
                    "referenceCount": 40,
                },
            },
            {
                "contexts": ["Unrelated"],
                "intents": [],
                "isInfluential": False,
                "citedPaper": {
                    "paperId": "paper-ref-2",
                    "corpusId": 404,
                    "title": "Unrelated physics paper",
                    "abstract": "A paper about optics.",
                    "citationCount": 10,
                },
            },
        ]


def test_flatten_query_bank_sorts_by_priority_and_applies_limit_override():
    specs = flatten_query_bank(_query_bank(), limit_override=7)

    assert [spec.query_id for spec in specs] == ["Q-A", "Q-B"]
    assert [spec.max_results for spec in specs] == [7, 7]
    assert specs[0].linked_claim_ids == ("MENDEL-002",)


def test_build_seed_candidates_dedupes_by_corpus_id_and_scores():
    records = [
        {
            "rank": 1,
            "query": {"query_id": "Q-A", "query_set_id": "set_a", "linked_claim_ids": ["MENDEL-002"]},
            "paper": {
                "corpusId": 101,
                "paperId": "paper-101",
                "title": "Paper 101",
                "abstract": "Text",
                "isOpenAccess": True,
                "openAccessPdf": {"url": "https://example.org/paper.pdf"},
                "citationCount": 100,
                "influentialCitationCount": 10,
                "referenceCount": 20,
                "year": 2020,
            },
        },
        {
            "rank": 2,
            "query": {"query_id": "Q-B", "query_set_id": "set_b", "linked_claim_ids": ["MENDEL-012"]},
            "paper": {
                "corpusId": 101,
                "paperId": "paper-101",
                "title": "Paper 101",
                "citationCount": 80,
                "influentialCitationCount": 8,
                "referenceCount": 18,
                "year": 2020,
            },
        },
        {
            "rank": 1,
            "query": {"query_id": "Q-MISSING", "query_set_id": "set_b"},
            "paper": {"paperId": "paper-missing"},
        },
    ]

    result = build_seed_candidates(records)

    assert result["candidate_count"] == 1
    assert result["skipped_without_corpus_id"] == 1
    candidate = result["candidates"][0]
    assert candidate["corpusId"] == "101"
    assert candidate["query_hit_count"] == 2
    assert candidate["linked_claim_ids"] == ["MENDEL-002", "MENDEL-012"]
    assert candidate["score"] > 0


def test_build_seed_candidates_enriches_disease_diversity_from_claim_metadata():
    records = [
        {
            "rank": 1,
            "query": {"query_id": "Q-A", "query_set_id": "set_a", "linked_claim_ids": ["MENDEL-002"]},
            "paper": {"corpusId": 101, "paperId": "paper-101", "title": "Paper 101", "citationCount": 10},
        },
        {
            "rank": 2,
            "query": {"query_id": "Q-B", "query_set_id": "set_b", "linked_claim_ids": ["MENDEL-012"]},
            "paper": {"corpusId": 101, "paperId": "paper-101", "title": "Paper 101", "citationCount": 10},
        },
    ]
    claim_metadata = {
        "MENDEL-002": {"disease_area": "cystic fibrosis", "genes": ["CFTR"]},
        "MENDEL-012": {"disease_area": "Wilson disease", "genes": ["ATP7B"]},
    }

    result = build_seed_candidates(records, claim_metadata=claim_metadata)

    candidate = result["candidates"][0]
    assert candidate["disease_areas"] == ["cystic fibrosis", "Wilson disease"]
    assert candidate["genes"] == ["CFTR", "ATP7B"]
    assert candidate["disease_area_count"] == 2
    assert candidate["query_hits"][0]["linked_disease_areas"] == ["cystic fibrosis"]


def test_select_seed_candidates_is_deterministic_and_balances_claims():
    candidates = [
        {"corpusId": 1, "title": "A", "score": 100, "best_rank": 1, "citationCount": 10, "linked_claim_ids": ["MENDEL-001"], "disease_areas": ["phenylketonuria"]},
        {"corpusId": 2, "title": "B", "score": 90, "best_rank": 2, "citationCount": 10, "linked_claim_ids": ["MENDEL-002"], "disease_areas": ["cystic fibrosis"]},
        {"corpusId": 3, "title": "C", "score": 80, "best_rank": 3, "citationCount": 10, "linked_claim_ids": ["MENDEL-002"], "disease_areas": ["cystic fibrosis"]},
    ]
    claim_metadata = {
        "MENDEL-001": {"disease_area": "phenylketonuria"},
        "MENDEL-002": {"disease_area": "cystic fibrosis"},
    }

    selected = select_seed_candidates(
        candidates,
        claim_metadata=claim_metadata,
        max_seeds=3,
        top_global=1,
        per_disease=1,
        per_claim=1,
    )

    assert [candidate["corpusId"] for candidate in selected] == [1, 2]
    assert "top_global_score" in selected[0]["selection_reasons"]
    assert "top_for_disease:cystic fibrosis" in selected[1]["selection_reasons"]


def test_capture_search_results_writes_provenance_outputs(tmp_path):
    query_bank_path = tmp_path / "queries.json"
    query_bank_path.write_text(json.dumps(_query_bank()), encoding="utf-8")
    output_dir = tmp_path / "outputs"
    client = FakeSearchClient()

    summary = capture_search_results(
        query_bank_path=query_bank_path,
        output_dir=output_dir,
        output_prefix="test_mendelian",
        overwrite=True,
        sleep_seconds=0,
        client=client,
        api_key="fake-key",
    )

    assert summary["query_count"] == 2
    assert summary["result_row_count"] == 4
    assert summary["candidate_count"] == 2
    assert client.calls == [
        ("cystic fibrosis CFTR modifier genes", 3),
        ("Wilson disease ATP7B", 5),
    ]

    results_path = output_dir / "test_mendelian_search_results.jsonl"
    outcomes_path = output_dir / "test_mendelian_query_outcomes.jsonl"
    candidate_path = output_dir / "test_mendelian_seed_candidates.json"
    summary_path = output_dir / "test_mendelian_run_summary.json"
    assert results_path.exists()
    assert outcomes_path.exists()
    assert candidate_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["query"]["query_id"] == "Q-A"
    assert rows[0]["rank"] == 1
    assert rows[0]["paper"]["corpusId"] == 101
    outcomes = [json.loads(line) for line in outcomes_path.read_text(encoding="utf-8").splitlines()]
    assert [outcome["query_id"] for outcome in outcomes] == ["Q-A", "Q-B"]
    assert [outcome["status"] for outcome in outcomes] == ["success", "success"]
    assert [outcome["result_count"] for outcome in outcomes] == [2, 2]

    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidates["candidate_count"] == 2
    assert candidates["candidates"][0]["score"] >= candidates["candidates"][1]["score"]


def test_capture_search_results_dry_run_writes_summary_only(tmp_path):
    query_bank_path = tmp_path / "queries.json"
    query_bank_path.write_text(json.dumps(_query_bank()), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    summary = capture_search_results(
        query_bank_path=query_bank_path,
        output_dir=output_dir,
        output_prefix="dry",
        overwrite=True,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["query_count"] == 2
    assert (output_dir / "dry_run_summary.json").exists()
    assert not (output_dir / "dry_search_results.jsonl").exists()
    assert not (output_dir / "dry_query_outcomes.jsonl").exists()


def test_capture_search_results_resume_skips_existing_query_rows(tmp_path):
    query_bank_path = tmp_path / "queries.json"
    query_bank_path.write_text(json.dumps(_query_bank()), encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing_record = {
        "run_id": "old",
        "captured_at": "2026-06-02T00:00:00+00:00",
        "topic_label": "mendelian_disease_v1",
        "query": {
            "query_set_id": "set_a",
            "query_id": "Q-A",
            "linked_claim_ids": ["MENDEL-002"],
            "intent": "Modifier evidence",
        },
        "rank": 1,
        "paper": {
            "corpusId": 101,
            "paperId": "paper-cf-1",
            "title": "Existing CF paper",
            "citationCount": 12,
            "influentialCitationCount": 1,
            "referenceCount": 4,
            "abstract": "Existing abstract",
        },
    }
    (output_dir / "resume_search_results.jsonl").write_text(
        json.dumps(existing_record) + "\n",
        encoding="utf-8",
    )
    client = FakeSearchClient()

    summary = capture_search_results(
        query_bank_path=query_bank_path,
        output_dir=output_dir,
        output_prefix="resume",
        resume=True,
        sleep_seconds=0,
        client=client,
        api_key="fake-key",
    )

    assert summary["completed_query_count_before_run"] == 1
    assert summary["pending_query_count"] == 1
    assert client.calls == [("Wilson disease ATP7B", 5)]
    assert summary["result_row_count"] == 3
    candidates = json.loads((output_dir / "resume_seed_candidates.json").read_text(encoding="utf-8"))
    assert candidates["candidate_count"] == 2


def test_capture_search_results_resume_skips_successful_zero_result_outcomes(tmp_path):
    query_bank_path = tmp_path / "queries.json"
    query_bank_path.write_text(json.dumps(_query_bank()), encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing_outcome = {
        "run_id": "old",
        "captured_at": "2026-06-02T00:00:00+00:00",
        "topic_label": "mendelian_disease_v1",
        "query_set_id": "set_a",
        "query_id": "Q-A",
        "query": "cystic fibrosis CFTR modifier genes",
        "status": "success",
        "result_count": 0,
    }
    (output_dir / "resume_zero_query_outcomes.jsonl").write_text(
        json.dumps(existing_outcome) + "\n",
        encoding="utf-8",
    )
    client = FakeSearchClient()

    summary = capture_search_results(
        query_bank_path=query_bank_path,
        output_dir=output_dir,
        output_prefix="resume_zero",
        resume=True,
        sleep_seconds=0,
        client=client,
        api_key="fake-key",
    )

    assert summary["completed_query_count_before_run"] == 1
    assert summary["pending_query_count"] == 1
    assert client.calls == [("Wilson disease ATP7B", 5)]
    assert summary["result_row_count"] == 2


def test_rebuild_candidates_uses_existing_rows_without_searching(tmp_path):
    query_bank_path = tmp_path / "queries.json"
    query_bank_path.write_text(json.dumps(_query_bank()), encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing_record = {
        "run_id": "old",
        "captured_at": "2026-06-02T00:00:00+00:00",
        "topic_label": "mendelian_disease_v1",
        "query": {
            "query_set_id": "set_a",
            "query_id": "Q-A",
            "linked_claim_ids": ["MENDEL-002"],
            "intent": "Modifier evidence",
        },
        "rank": 1,
        "paper": {
            "corpusId": 101,
            "paperId": "paper-cf-1",
            "title": "Existing CF paper",
            "citationCount": 12,
            "influentialCitationCount": 1,
            "referenceCount": 4,
            "abstract": "Existing abstract",
        },
    }
    (output_dir / "rebuild_search_results.jsonl").write_text(
        json.dumps(existing_record) + "\n",
        encoding="utf-8",
    )
    client = FakeSearchClient()

    summary = capture_search_results(
        query_bank_path=query_bank_path,
        output_dir=output_dir,
        output_prefix="rebuild",
        rebuild_candidates=True,
        client=client,
        api_key="fake-key",
    )

    assert summary["rebuild_candidates"] is True
    assert summary["result_row_count"] == 1
    assert summary["candidate_count"] == 1
    assert client.calls == []
    candidates = json.loads((output_dir / "rebuild_seed_candidates.json").read_text(encoding="utf-8"))
    assert candidates["candidates"][0]["corpusId"] == "101"


def test_expand_references_writes_reference_and_final_selection_outputs(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    seed_candidates_path = tmp_path / "seed_candidates.json"
    seed_candidates_path.write_text(
        json.dumps(
            {
                "topic_label": "mendelian_disease_v1",
                "candidates": [
                    {
                        "corpusId": 101,
                        "paperId": "paper-cf-1",
                        "title": "CFTR modifier genes and lung disease",
                        "score": 100,
                        "best_rank": 1,
                        "citationCount": 120,
                        "linked_claim_ids": ["MENDEL-002"],
                        "disease_areas": ["cystic fibrosis"],
                        "genes": ["CFTR"],
                        "authors": [
                            {"authorId": "a1", "name": "Ada Author"},
                            {"authorId": "a2", "name": "Ben Author"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "MENDEL-002",
                        "disease_area": "cystic fibrosis",
                        "genes": ["CFTR"],
                        "query_terms": ["modifier genes"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    client = FakeSearchClient()

    summary = expand_references(
        seed_candidates_path=seed_candidates_path,
        claim_bank_path=claims_path,
        output_dir=output_dir,
        output_prefix="expand",
        max_seeds=1,
        top_global=1,
        per_disease=0,
        per_claim=0,
        references_per_seed=2,
        min_reference_score=4,
        max_final_corpus_ids=3,
        sleep_seconds=0,
        client=client,
        api_key="fake-key",
    )

    assert summary["selected_seed_count"] == 1
    assert summary["reference_records_written"] == 1
    assert summary["reference_candidate_count"] == 1
    assert summary["final_corpus_id_count"] == 2
    rows = [
        json.loads(line)
        for line in (output_dir / "expand_reference_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["paper"]["corpusId"] == 303
    outcomes = [
        json.loads(line)
        for line in (output_dir / "expand_reference_outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert outcomes[0]["status"] == "success"
    assert outcomes[0]["result_count"] == 2
    assert outcomes[0]["kept_count"] == 1
    final_selection = json.loads((output_dir / "expand_final_selection.json").read_text(encoding="utf-8"))
    assert final_selection["corpus_ids"] == ["101", "303"]
    assert final_selection["author_ids"] == ["a1", "a2", "a3", "a4"]

