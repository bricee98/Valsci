#!/usr/bin/env python
"""
Seed a local demo arena so the arena results UI can be previewed without
running Semantic Scholar or the processor.

Usage:
  python scripts/seed_demo_arena.py
  python scripts/seed_demo_arena.py --stage venue_scoring
  python scripts/seed_demo_arena.py --stage final_report --url-base http://localhost:3000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.config.settings import Config  # noqa: E402
from app.services.claim_store import ClaimStore  # noqa: E402
from app.services.llm.types import empty_usage  # noqa: E402
from app.services.provider_catalog import ProviderCatalog  # noqa: E402


CLAIMS = [
    "Creatine improves short-term memory in adults.",
    "Vitamin D reduces fall risk in older adults.",
]


def _build_usage(cost_usd: float) -> Dict[str, object]:
    usage = empty_usage()
    usage["input_tokens"] = 1200
    usage["output_tokens"] = 320
    usage["total_tokens"] = 1520
    usage["cost_usd"] = round(cost_usd, 4)
    usage["is_estimated"] = False
    return usage


def _build_queries(claim_text: str, variant: str) -> List[str]:
    if "Creatine" in claim_text:
        base = [
            "creatine memory adults randomized trial",
            "creatine supplementation cognition healthy adults",
            "creatine working memory placebo controlled",
        ]
    else:
        base = [
            "vitamin d fall risk older adults randomized trial",
            "vitamin d supplementation falls elderly meta analysis",
            "vitamin d balance fracture prevention aged adults",
        ]
    if variant == "Alt":
        return [f"{query} systematic review" for query in base]
    return base


def _paper_title(claim_text: str, variant: str, index: int) -> str:
    prefix = "Creatine" if "Creatine" in claim_text else "Vitamin D"
    return f"{prefix} Study {index + 1} ({variant})"


def _processed_papers(claim_text: str, variant: str, stage: str) -> List[Dict[str, object]]:
    rows = []
    score_base = 0.74 if variant == "Baseline" else 0.61
    for index in range(3):
        score = round(score_base - (index * 0.08), 2)
        rows.append(
            {
                "paper": {
                    "corpusId": 1000 + index,
                    "title": _paper_title(claim_text, variant, index),
                },
                "relevance": round(0.91 - (index * 0.07), 2),
                "excerpts": [f"Representative excerpt {index + 1} for {variant.lower()}."],
                "explanations": [f"{variant} interpreted this paper as moderately relevant."],
                "score": score if stage in {"venue_scoring", "final_report"} else None,
                "score_status": "completed" if stage in {"venue_scoring", "final_report"} else "pending",
            }
        )
    return rows


def _build_claim_data(
    *,
    claim_text: str,
    claim_key: str,
    run_id: str,
    arena_id: str,
    stage: str,
    provider_snapshot: Dict[str, object],
    variant: str,
    claim_index: int,
) -> Dict[str, object]:
    usage_cost = 0.013 + (claim_index * 0.002) + (0.001 if variant == "Alt" else 0.0)
    claim_data: Dict[str, object] = {
        "text": claim_text,
        "status": "processed",
        "batch_id": arena_id,
        "claim_id": run_id,
        "run_id": run_id,
        "claim_key": claim_key,
        "arena_id": arena_id,
        "review_type": "regular",
        "execution_mode": "full_pipeline",
        "stop_after": stage,
        "completed_stage": stage,
        "is_stage_checkpoint": stage != "final_report",
        "provider_snapshot": provider_snapshot,
        "model_overrides": {},
        "search_config": {"num_queries": 3, "results_per_query": 5},
        "bibliometric_config": {"use_bibliometrics": True},
        "usage": _build_usage(usage_cost),
        "usage_by_stage": {
            "query_generation": _build_usage(0.0025),
            "paper_analysis": _build_usage(0.0075 if variant == "Baseline" else 0.0085),
        },
        "semantic_scholar_queries": _build_queries(claim_text, variant),
    }

    if stage in {"paper_analysis", "venue_scoring", "final_report"}:
        claim_data["raw_papers"] = [
            {"corpusId": 1000 + idx, "title": _paper_title(claim_text, variant, idx)}
            for idx in range(3)
        ]
        claim_data["processed_papers"] = _processed_papers(claim_text, variant, stage)
        claim_data["non_relevant_papers"] = [{"corpusId": 2000, "title": f"Off-topic screen for {variant}"}]
        claim_data["inaccessible_papers"] = [{"corpusId": 3000, "title": f"Unavailable PDF for {variant}"}]
        claim_data["failed_papers"] = []

    if stage == "final_report":
        claim_data["report"] = {
            "claimRating": 4 if variant == "Baseline" else 3,
            "explanation": (
                "The evidence trends supportive overall, but effect sizes are modest."
                if variant == "Baseline"
                else "The evidence is mixed and depends heavily on study quality."
            ),
            "usage_summary": claim_data["usage"],
        }

    return claim_data


def seed_demo_arena(stage: str, url_base: str) -> str:
    store = ClaimStore()
    catalog = ProviderCatalog()
    base_snapshot = catalog.build_snapshot("default")

    arena = store.create_arena(
        title=f"Demo Arena ({stage.replace('_', ' ').title()})",
        batch_tags=["demo-arena"],
        execution_mode="full_pipeline",
        current_stage=stage,
        candidates=[
            {"candidate_id": "candidate-0", "provider_id": "default", "label": "Baseline"},
            {"candidate_id": "candidate-1", "provider_id": "default", "label": "Alt"},
        ],
        source="demo_seed",
    )

    run_ids: List[str] = []
    claim_keys: List[str] = []
    stage_cost = {
        "query_generation": 0.0042,
        "paper_analysis": 0.0134,
        "venue_scoring": 0.0151,
        "final_report": 0.0186,
    }[stage]

    for claim_index, claim_text in enumerate(CLAIMS):
        claim_record, _ = store.get_or_create_claim(claim_text, batch_tags=[arena["arena_id"], "demo-arena"])
        claim_keys.append(claim_record["claim_key"])
        for variant in ["Baseline", "Alt"]:
            provider_snapshot = dict(base_snapshot)
            provider_snapshot["label"] = variant
            run_record = store.create_run(
                claim_record=claim_record,
                batch_tags=[arena["arena_id"], "demo-arena"],
                arena_id=arena["arena_id"],
                execution_mode="full_pipeline",
                stop_after=stage,
                provider_snapshot=provider_snapshot,
                cost_estimate={"expected": {"cost_usd": stage_cost + (0.001 if variant == "Alt" else 0.0)}},
                cost_confirmation={"accepted": True},
                transport_batch_id=arena["arena_id"],
                review_type="regular",
                status="processed",
                source="demo_seed",
                completed_stage=stage,
                is_stage_checkpoint=(stage != "final_report"),
            )

            claim_data = _build_claim_data(
                claim_text=claim_text,
                claim_key=claim_record["claim_key"],
                run_id=run_record["run_id"],
                arena_id=arena["arena_id"],
                stage=stage,
                provider_snapshot=provider_snapshot,
                variant=variant,
                claim_index=claim_index,
            )

            claim_path = Path(Config.SAVED_JOBS_DIR) / arena["arena_id"] / f"{run_record['run_id']}.txt"
            claim_path.parent.mkdir(parents=True, exist_ok=True)
            claim_path.write_text(json.dumps(claim_data, indent=2), encoding="utf-8")

            store.ingest_transport_artifact(arena["arena_id"], run_record["run_id"])
            run_ids.append(run_record["run_id"])

    arena["claim_keys"] = claim_keys
    store.append_arena_stage_history(
        arena,
        stage=stage,
        run_ids=run_ids,
        source="demo_seed",
    )

    return f"{url_base.rstrip('/')}/arena_results?arena_id={arena['arena_id']}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a demo arena into local state.")
    parser.add_argument(
        "--stage",
        default="paper_analysis",
        choices=["query_generation", "paper_analysis", "venue_scoring", "final_report"],
        help="Arena stage to render in the seeded results page.",
    )
    parser.add_argument(
        "--url-base",
        default="http://localhost:3000",
        help="Base URL for the local Flask app.",
    )
    args = parser.parse_args()

    url = seed_demo_arena(args.stage, args.url_base)
    print("Demo arena seeded successfully.")
    print(url)
    if not Config.SEMANTIC_SCHOLAR_API_KEY:
        print("Note: the app still needs a non-empty SEMANTIC_SCHOLAR_API_KEY to pass startup validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
