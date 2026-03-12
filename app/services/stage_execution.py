"""Shared helpers for stage-gated execution and arena continuation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set


TASK_NAMES = ["query_generation", "paper_analysis", "venue_scoring", "final_report"]
VALID_EXECUTION_MODES = {"full_pipeline", "reuse_retrieval"}
STAGE_REVIEW_FIELDS = {
    "query_generation": "semantic_scholar_queries",
    "paper_analysis": "processed_papers",
    "venue_scoring": "processed_papers",
    "final_report": "report",
}
START_STATUS_BY_STAGE = {
    "query_generation": "queued",
    "paper_analysis": "ready_for_search",
    "venue_scoring": "ready_for_analysis",
    "final_report": "ready_for_analysis",
}


def normalize_execution_mode(value: Optional[str]) -> str:
    mode = str(value or "full_pipeline").strip().lower()
    if mode not in VALID_EXECUTION_MODES:
        raise ValueError("execution_mode must be either 'full_pipeline' or 'reuse_retrieval'.")
    return mode


def normalize_stop_after(value: Optional[str]) -> str:
    stage = str(value or "final_report").strip().lower()
    if stage not in TASK_NAMES:
        raise ValueError(
            "stop_after must be one of 'query_generation', 'paper_analysis', 'venue_scoring', or 'final_report'."
        )
    return stage


def normalize_execution_settings(
    execution_mode: Optional[str],
    stop_after: Optional[str],
) -> tuple[str, str]:
    mode = normalize_execution_mode(execution_mode)
    stage = normalize_stop_after(stop_after)
    if stage == "query_generation" and mode == "reuse_retrieval":
        mode = "full_pipeline"
    return mode, stage


def next_stage(stage: Optional[str]) -> Optional[str]:
    if stage not in TASK_NAMES:
        return None
    index = TASK_NAMES.index(stage)
    if index >= len(TASK_NAMES) - 1:
        return None
    return TASK_NAMES[index + 1]


def stages_after(stage: str) -> List[str]:
    return TASK_NAMES[TASK_NAMES.index(stage) + 1 :]


def stages_before(stage: str) -> List[str]:
    return TASK_NAMES[: TASK_NAMES.index(stage)]


def skip_stages_for_execution(
    *,
    stop_after: str,
    start_stage: Optional[str] = None,
    reuse_query_generation: bool = False,
) -> Set[str]:
    skipped: Set[str] = set(stages_after(stop_after))
    if start_stage:
        skipped.update(stages_before(start_stage))
    if reuse_query_generation:
        skipped.add("query_generation")
    return skipped


def start_status_for_stage(stage: str) -> str:
    return START_STATUS_BY_STAGE[stage]


def is_stage_checkpoint(stop_after: Optional[str]) -> bool:
    return bool(stop_after) and normalize_stop_after(stop_after) != "final_report"


def continue_stage_for_run(run_record: Dict[str, object]) -> Optional[str]:
    completed_stage = run_record.get("completed_stage")
    if isinstance(completed_stage, str):
        return next_stage(completed_stage)
    return None


def checkpoint_complete(run_record: Dict[str, object]) -> bool:
    return bool(run_record.get("report_available")) or bool(run_record.get("is_stage_checkpoint"))


def stage_label(stage: Optional[str]) -> str:
    labels = {
        "query_generation": "Query Generation",
        "paper_analysis": "Paper Analysis",
        "venue_scoring": "Venue Scoring",
        "final_report": "Final Report",
    }
    return labels.get(stage or "", "Unknown")


def review_value_for_stage(claim_data: Dict[str, object], stage: Optional[str]):
    if not isinstance(claim_data, dict) or not stage:
        return None
    field = STAGE_REVIEW_FIELDS.get(stage)
    if not field:
        return None
    return claim_data.get(field)


def chosen_run_ids(decisions: Iterable[Dict[str, object]]) -> List[str]:
    run_ids: List[str] = []
    for item in decisions:
        run_id = item.get("selected_run_id") if isinstance(item, dict) else None
        if isinstance(run_id, str) and run_id.strip():
            run_ids.append(run_id.strip())
    return run_ids
