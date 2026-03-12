from flask import Blueprint, request, jsonify, send_file, render_template, current_app, Response, has_app_context
from flask import session, redirect, url_for, flash
import os
import uuid
import shutil
import threading
import json
import gzip
from app.services.claim_processor import ClaimProcessor
from app.models.claim import Claim
from app.models.batch_job import BatchJob
from app.models.paper import Paper
from datetime import datetime, timezone
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import math
from app.services.email_service import EmailService
import logging
import traceback
from functools import wraps
from pathlib import Path
from app.services.batch_export import build_export_document
from app.services.batch_state import build_batch_state, list_batch_ids
from app.config.settings import Config
from app.services.claim_store import ClaimStore
from app.services.ollama_discovery import discover_ollama_models
from app.services.provider_catalog import ProviderCatalog
from app.services.submission_service import SubmissionService

api = Blueprint('api', __name__)

QUEUED_JOBS_DIR = Config.QUEUED_JOBS_DIR
SAVED_JOBS_DIR = Config.SAVED_JOBS_DIR

logger = logging.getLogger(__name__)

# Initialize EmailService at module level
email_service = EmailService()


def _configured_path(config_key: str, fallback: str) -> Path:
    if has_app_context():
        configured_value = current_app.config.get(config_key)
        if configured_value:
            return Path(configured_value)
    return Path(fallback)


def _saved_jobs_dir() -> Path:
    return _configured_path("SAVED_JOBS_DIR", SAVED_JOBS_DIR)


def _queued_jobs_dir() -> Path:
    return _configured_path("QUEUED_JOBS_DIR", QUEUED_JOBS_DIR)


def _trace_root_dir() -> Path:
    return _configured_path("TRACE_DIR", SAVED_JOBS_DIR)


def _state_dir() -> Path:
    return _configured_path("STATE_DIR", Config.STATE_DIR)


def _provider_catalog_path() -> Path:
    return _configured_path("PROVIDER_CATALOG_PATH", Config.PROVIDER_CATALOG_PATH)


def _claim_store() -> ClaimStore:
    return ClaimStore(
        state_dir=str(_state_dir()),
        saved_jobs_dir=str(_saved_jobs_dir()),
        queued_jobs_dir=str(_queued_jobs_dir()),
        trace_dir=str(_trace_root_dir()),
    )


def _provider_catalog() -> ProviderCatalog:
    return ProviderCatalog(path=str(_provider_catalog_path()))


def _submission_service() -> SubmissionService:
    return SubmissionService(_claim_store(), _provider_catalog())


def _public_providers() -> List[Dict[str, Any]]:
    public_fields = {
        "provider_id",
        "label",
        "provider_type",
        "default_model",
        "local_backend",
        "task_defaults",
        "http_referer",
        "site_name",
        "azure_openai_endpoint",
        "azure_openai_api_version",
        "azure_ai_inference_endpoint",
        "models",
    }
    providers = []
    for provider in _provider_catalog().list_providers():
        providers.append({key: provider.get(key) for key in public_fields})
    return providers


def _find_claim_artifact(batch_id: str, artifact_dir: str, claim_id: str, extension: str):
    """Resolve trace/issue artifact path with TRACE_DIR support and saved_jobs fallback."""
    candidates = [
        _trace_root_dir() / batch_id / artifact_dir / f"{claim_id}.{extension}",
        _trace_root_dir() / batch_id / artifact_dir / f"{claim_id}.{extension}.gz",
    ]
    # Backward compatibility fallback
    candidates.extend(
        [
            _saved_jobs_dir() / batch_id / artifact_dir / f"{claim_id}.{extension}",
            _saved_jobs_dir() / batch_id / artifact_dir / f"{claim_id}.{extension}.gz",
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return None


def _read_jsonl_artifact(path: str) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid_lines = 0
    artifact_path = Path(path)
    open_fn = gzip.open if artifact_path.suffix == ".gz" else open
    with open_fn(artifact_path, "rt", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                if isinstance(record, dict):
                    records.append(record)
                else:
                    invalid_lines += 1
            except json.JSONDecodeError:
                invalid_lines += 1
    return records, invalid_lines


def _claim_file_candidates(batch_id: str, claim_id: str) -> List[Tuple[str, str]]:
    return [
        ("queued_jobs", str(_queued_jobs_dir() / batch_id / f"{claim_id}.txt")),
        ("saved_jobs", str(_saved_jobs_dir() / batch_id / f"{claim_id}.txt")),
    ]


def _find_claim_file(batch_id: str, claim_id: str) -> Tuple[Optional[str], Optional[str]]:
    for location, path in _claim_file_candidates(batch_id, claim_id):
        if os.path.exists(path):
            return location, path
    return None, None


def _load_claim_data(batch_id: str, claim_id: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    location, path = _find_claim_file(batch_id, claim_id)
    if not path:
        return None, None, None
    with open(path, 'r', encoding='utf-8') as f:
        return location, path, json.load(f)


def _infer_resume_stage(claim_data: Dict[str, Any]) -> Tuple[str, str]:
    queries = claim_data.get("semantic_scholar_queries")
    raw_papers = claim_data.get("raw_papers")
    if not isinstance(queries, list) or not queries:
        return "queued", "No saved search queries were found."
    if not isinstance(raw_papers, list) or not raw_papers:
        return "ready_for_search", "Saved search queries exist, but no fetched paper set was found."
    return "ready_for_analysis", "Saved paper search output exists; resume from analysis/report generation."


def _build_claim_trace_metadata(batch_id: str, claim_id: str) -> Dict[str, Any]:
    location, path = _find_claim_file(batch_id, claim_id)
    metadata = {
        "claim_status": None,
        "claim_location": location,
        "resume_available": False,
        "resume_stage": None,
        "resume_reason": None,
    }
    if not path:
        metadata["resume_reason"] = "Claim file was not found."
        return metadata

    try:
        with open(path, 'r', encoding='utf-8') as f:
            claim_data = json.load(f)
    except Exception as exc:
        metadata["resume_reason"] = f"Claim file could not be read: {exc}"
        return metadata

    status = claim_data.get("status")
    metadata["claim_status"] = status

    if location == "queued_jobs":
        metadata["resume_reason"] = "Claim is already queued or in progress."
        return metadata

    if status != "processed":
        metadata["resume_reason"] = f"Claim is stored in saved_jobs with status '{status}'."
        return metadata

    resume_stage, resume_reason = _infer_resume_stage(claim_data)
    metadata.update(
        {
            "resume_available": True,
            "resume_stage": resume_stage,
            "resume_reason": resume_reason,
        }
    )
    return metadata


def _build_batch_state_view(batch_id: str) -> Optional[Dict[str, Any]]:
    store_state = _claim_store().build_batch_state(batch_id)
    if store_state is not None:
        return store_state
    return build_batch_state(
        batch_id=batch_id,
        saved_jobs_root=_saved_jobs_dir(),
        queued_jobs_root=_queued_jobs_dir(),
    )


def _claim_results_template(claim_data: Dict[str, Any]) -> str:
    try:
        additional_info = json.loads(claim_data.get('additional_info', '{}'))
        if 'overall_rating' in additional_info and 'plausibility_level' in additional_info:
            return 'llm_screen_results.html'
    except json.JSONDecodeError:
        pass
    return 'results.html'


def _serialize_batch_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id"),
        "claim_key": claim.get("claim_key"),
        "text": claim.get("text", ""),
        "status": claim.get("status", "unknown"),
        "report": claim.get("report", {}),
        "review_type": claim.get("review_type", "regular"),
        "claim_location": claim.get("claim_location") or claim.get("location"),
        "report_available": bool(claim.get("report_available", False)),
        "is_active": bool(claim.get("is_active", False)),
        "rating": claim.get("rating", claim.get("claimRating")),
    }


def _default_search_config() -> Dict[str, Any]:
    return {"num_queries": 5, "results_per_query": 5}


def _default_bibliometric_config() -> Dict[str, Any]:
    return {
        "use_bibliometrics": True,
        "author_impact_weight": 0.4,
        "citation_impact_weight": 0.4,
        "venue_impact_weight": 0.2,
    }


def _claims_from_payload(payload: Dict[str, Any]) -> List[str]:
    claims = payload.get("claims", [])
    cleaned: List[str] = []
    for value in claims if isinstance(claims, list) else []:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            cleaned.append(text)
    return cleaned


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.split(",")
    else:
        items = []

    cleaned: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned

def save_claim_to_file(claim, batch_id, claim_id):
    claim_dir = _queued_jobs_dir() / batch_id
    claim_dir.mkdir(parents=True, exist_ok=True)
    claim_file = claim_dir / f"{claim_id}.txt"
    
    # Ensure claim.text is a string, not a list
    claim_text = claim.text[0] if isinstance(claim.text, list) else claim.text
    
    with claim_file.open('w', encoding='utf-8') as f:
        json.dump({
            "text": claim_text,
            "status": "queued",
            "batch_id": batch_id,
            "claim_id": claim_id,
            "search_config": claim.search_config,
            "bibliometric_config": claim.bibliometric_config,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "is_estimated": False
            },
            "usage_by_stage": {},
            "additional_info": ""
        }, f, indent=2)

def verify_password(password):
    if current_app.config['REQUIRE_PASSWORD']:
        if not password:
            return False
        return password == current_app.config['ACCESS_PASSWORD']
    return True

def is_authenticated():
    """Check if the user is authenticated"""
    if not current_app.config['REQUIRE_PASSWORD']:
        return True
    return session.get('authenticated', False)

def auth_required(f):
    """Decorator to require authentication for a route"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_app.config['REQUIRE_PASSWORD']:
            return f(*args, **kwargs)
        
        if not is_authenticated():
            if request.path.startswith('/api/v1/'):
                # Return JSON error for API routes
                return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401
            else:
                # Redirect to login page for UI routes
                return redirect(url_for('api.login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function

@api.route('/api/v1/claims/<batch_id>/<claim_id>', methods=['GET'])
@auth_required
def get_claim_status(batch_id, claim_id):
    # Check queued jobs first
    claim_file = _queued_jobs_dir() / batch_id / f"{claim_id}.txt"
    if claim_file.exists():
        with claim_file.open('r', encoding='utf-8') as f:
            claim_data = json.load(f)
            return jsonify({
                "claim_id": claim_id,
                "text": claim_data.get('text', ''),
                "status": claim_data.get('status', 'Unknown'),
                "additional_info": claim_data.get('additional_info', {}),
                "review_type": claim_data.get('review_type', 'regular')
            }), 200
    
    # Then check saved jobs
    claim_file = _saved_jobs_dir() / batch_id / f"{claim_id}.txt"
    if claim_file.exists():
        with claim_file.open('r', encoding='utf-8') as f:
            claim_data = json.load(f)
            return jsonify({
                "claim_id": claim_id,
                "text": claim_data.get('text', ''),
                "status": claim_data.get('status', 'Unknown'),
                "additional_info": claim_data.get('additional_info', {}),
                "review_type": claim_data.get('review_type', 'regular')
            }), 200
            
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/claims/<batch_id>/<claim_id>/report', methods=['GET'])
@auth_required
def get_claim_report(batch_id, claim_id):
    location, _, claim_data = _load_claim_data(batch_id, claim_id)
    if claim_data is None:
        return jsonify({"error": "Claim not found"}), 404
    if location == "queued_jobs":
        return jsonify({
            "error": "Claim is still processing",
            "claim_id": claim_id,
            "status": claim_data.get("status", "unknown"),
            "claim_location": location,
            "code": "CLAIM_PROCESSING",
        }), 409
    if location == "saved_jobs":
        return jsonify({
            "claim_id": claim_id,
            "text": claim_data.get('text', ''),
            "status": claim_data.get('status', ''),
            "report": claim_data.get('report', {})
        }), 200
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/claims/<batch_id>/<claim_id>/trace', methods=['GET'])
@auth_required
def download_claim_trace(batch_id, claim_id):
    trace_file = _find_claim_artifact(batch_id, "traces", claim_id, "jsonl")
    if not trace_file or not trace_file.exists():
        return jsonify({"error": "Trace not found"}), 404
    download_name = f"{claim_id}_trace.jsonl.gz" if trace_file.suffix == ".gz" else f"{claim_id}_trace.jsonl"
    return send_file(trace_file, as_attachment=True, download_name=download_name)


@api.route('/api/v1/claims/<batch_id>/<claim_id>/trace_records', methods=['GET'])
@auth_required
def get_claim_trace_records(batch_id, claim_id):
    trace_file = _find_claim_artifact(batch_id, "traces", claim_id, "jsonl")
    if not trace_file or not trace_file.exists():
        return jsonify({"error": "Trace not found"}), 404

    focus_trace_id = (request.args.get("focus_trace_id") or "").strip() or None
    try:
        records, invalid_lines = _read_jsonl_artifact(trace_file)
    except Exception as exc:
        logger.error(f"Error loading trace file for claim {claim_id}: {str(exc)}")
        return jsonify({
            "error": "Failed to load trace records",
            "details": str(exc),
        }), 500

    error_like_indices = []
    focused_index = None
    for idx, record in enumerate(records):
        status = str(record.get("status", "")).lower()
        has_error = bool(
            record.get("parse_error")
            or record.get("error_message")
            or record.get("error_type")
            or status in {"error", "retrying"}
        )
        if has_error:
            error_like_indices.append(idx)
        if focus_trace_id and record.get("trace_id") == focus_trace_id:
            focused_index = idx

    if focused_index is not None:
        highlighted_indices = [focused_index]
    elif error_like_indices:
        highlighted_indices = [error_like_indices[-1]]
    else:
        highlighted_indices = []

    claim_metadata = _build_claim_trace_metadata(batch_id, claim_id)

    return jsonify({
        "batch_id": batch_id,
        "claim_id": claim_id,
        "focus_trace_id": focus_trace_id,
        "focused_index": focused_index,
        "highlighted_indices": highlighted_indices,
        "error_like_indices": error_like_indices,
        "invalid_lines": invalid_lines,
        "compressed": trace_file.suffix == ".gz",
        "trace_file": f"traces/{claim_id}.jsonl" + (".gz" if trace_file.suffix == ".gz" else ""),
        "records": records,
        **claim_metadata,
    }), 200

@api.route('/api/v1/claims/<batch_id>/<claim_id>/issues', methods=['GET'])
@auth_required
def download_claim_issues(batch_id, claim_id):
    issue_file = _find_claim_artifact(batch_id, "issues", claim_id, "jsonl")
    if not issue_file or not issue_file.exists():
        return jsonify({"error": "Issues not found"}), 404
    download_name = f"{claim_id}_issues.jsonl.gz" if issue_file.suffix == ".gz" else f"{claim_id}_issues.jsonl"
    return send_file(issue_file, as_attachment=True, download_name=download_name)


@api.route('/api/v1/migration/status', methods=['GET'])
@auth_required
def migration_status():
    return jsonify(_claim_store().migration_status()), 200


@api.route('/api/v1/migration/run', methods=['POST'])
@auth_required
def migration_run():
    payload = request.get_json(silent=True) or {}
    apply_changes = bool(payload.get("apply", False))
    return jsonify(_claim_store().migrate_legacy(apply_changes=apply_changes)), 200


@api.route('/api/v1/providers', methods=['GET'])
@auth_required
def list_providers():
    return jsonify({"providers": _provider_catalog().list_providers()}), 200


@api.route('/api/v1/providers', methods=['POST'])
@auth_required
def create_provider():
    payload = request.get_json(silent=True) or {}
    provider = _provider_catalog().upsert_provider(payload)
    return jsonify(provider), 201


@api.route('/api/v1/providers/<provider_id>', methods=['PUT'])
@auth_required
def update_provider(provider_id):
    payload = request.get_json(silent=True) or {}
    payload["provider_id"] = provider_id
    provider = _provider_catalog().upsert_provider(payload)
    return jsonify(provider), 200


@api.route('/api/v1/providers/<provider_id>', methods=['DELETE'])
@auth_required
def delete_provider(provider_id):
    deleted = _provider_catalog().delete_provider(provider_id)
    if not deleted:
        return jsonify({"error": "Provider not found"}), 404
    return jsonify({"deleted": True, "provider_id": provider_id}), 200


@api.route('/api/v1/providers/ollama/discover', methods=['POST'])
@auth_required
def discover_ollama_provider_models():
    payload = request.get_json(silent=True) or {}
    provider_id = str(payload.get("provider_id", "")).strip()
    if provider_id:
        provider = _provider_catalog().get_provider(provider_id)
        if not provider:
            return jsonify({"error": "Provider not found"}), 404
        is_ollama_provider = provider.get("provider_type") == "ollama" or provider.get("local_backend") == "ollama"
        if not is_ollama_provider:
            return jsonify({"error": "Provider is not Ollama-backed"}), 400
    try:
        models = discover_ollama_models()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"models": models, "count": len(models)}), 200


@api.route('/api/v1/claims/preflight', methods=['POST'])
@auth_required
def claims_preflight():
    payload = request.get_json(silent=True) or {}
    claims = _claims_from_payload(payload)
    if not claims:
        return jsonify({"error": "At least one claim is required"}), 400

    search_config = dict(payload.get("search_config") or _default_search_config())
    try:
        candidates = _submission_service().resolve_candidates(payload.get("candidates"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    preflight = _submission_service().preflight(
        claims=claims,
        candidates=candidates,
        search_config=search_config,
        duplicate_strategy=str(payload.get("duplicate_strategy", "rerun")),
        execution_mode=str(payload.get("execution_mode", "full_pipeline")),
        stop_after=str(payload.get("stop_after", "final_report")),
    )
    return jsonify(preflight), 200


@api.route('/api/v1/runs', methods=['POST'])
@auth_required
def create_runs():
    payload = request.get_json(silent=True) or {}
    claims = _claims_from_payload(payload)
    if not claims:
        return jsonify({"error": "At least one claim is required"}), 400

    search_config = dict(payload.get("search_config") or _default_search_config())
    bibliometric_config = dict(payload.get("bibliometric_config") or _default_bibliometric_config())
    cost_confirmation = dict(payload.get("cost_confirmation") or {})
    duplicate_strategy = str(payload.get("duplicate_strategy", "rerun"))
    batch_tags = _string_list(payload.get("batch_tags"))
    execution_mode = str(payload.get("execution_mode", "full_pipeline"))

    try:
        candidates = _submission_service().resolve_candidates(payload.get("candidates"))
        result = _submission_service().submit(
            claims=claims,
            candidates=candidates,
            search_config=search_config,
            bibliometric_config=bibliometric_config,
            batch_tags=batch_tags,
            execution_mode=execution_mode,
            stop_after=str(payload.get("stop_after", "final_report")),
            cost_confirmation=cost_confirmation,
            duplicate_strategy=duplicate_strategy,
            create_arena=False,
            review_type=str(payload.get("review_type", "regular")),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result), 202


@api.route('/api/v1/arenas', methods=['POST'])
@auth_required
def create_arena():
    payload = request.get_json(silent=True) or {}
    claims = _claims_from_payload(payload)
    if not claims:
        return jsonify({"error": "At least one claim is required"}), 400

    try:
        candidates = _submission_service().resolve_candidates(payload.get("candidates"))
        result = _submission_service().submit(
            claims=claims,
            candidates=candidates,
            search_config=dict(payload.get("search_config") or _default_search_config()),
            bibliometric_config=dict(payload.get("bibliometric_config") or _default_bibliometric_config()),
            batch_tags=_string_list(payload.get("batch_tags")),
            execution_mode=str(payload.get("execution_mode", "full_pipeline")),
            stop_after=str(payload.get("stop_after", "final_report")),
            cost_confirmation=dict(payload.get("cost_confirmation") or {}),
            duplicate_strategy=str(payload.get("duplicate_strategy", "rerun")),
            create_arena=True,
            arena_title=payload.get("title"),
            review_type="regular",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result), 202


@api.route('/api/v1/arenas/<arena_id>/continue/preflight', methods=['POST'])
@auth_required
def continue_arena_preflight(arena_id):
    payload = request.get_json(silent=True) or {}
    try:
        result = _submission_service().continue_arena_preflight(
            arena_id=arena_id,
            decisions=payload.get("decisions"),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result), 200


@api.route('/api/v1/arenas/<arena_id>/continue', methods=['POST'])
@auth_required
def continue_arena(arena_id):
    payload = request.get_json(silent=True) or {}
    try:
        result = _submission_service().continue_arena(
            arena_id=arena_id,
            decisions=payload.get("decisions"),
            cost_confirmation=dict(payload.get("cost_confirmation") or {}),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result), 202


@api.route('/api/v1/runs/<run_id>', methods=['GET'])
@auth_required
def get_run(run_id):
    run_record = _claim_store().get_run(run_id)
    if not run_record:
        return jsonify({"error": "Run not found"}), 404
    claim_data = _claim_store().load_claim_data_for_run(run_record)
    return jsonify(
        {
            "run": _claim_store().build_run_summary(run_record),
            "claim_data": claim_data or run_record.get("claim_data") or {},
        }
    ), 200


@api.route('/api/v1/claims/<claim_key>', methods=['GET'])
@auth_required
def get_claim_detail_api(claim_key):
    detail = _claim_store().build_claim_detail(claim_key)
    if not detail:
        return jsonify({"error": "Claim not found"}), 404
    return jsonify(detail), 200


@api.route('/api/v1/claims', methods=['GET'])
@auth_required
def list_claims_api():
    search_term = (request.args.get("search") or "").strip().lower()
    claims = []
    store = _claim_store()
    for claim in store.list_claims():
        text = str(claim.get("text", ""))
        if search_term and search_term not in text.lower() and search_term not in claim.get("claim_key", "").lower():
            continue
        latest_run = store.get_run(claim.get("latest_run_id")) if claim.get("latest_run_id") else None
        claims.append(
            {
                "claim_key": claim.get("claim_key"),
                "text": text,
                "batch_tags": claim.get("batch_tags", []),
                "latest_run": store.build_run_summary(latest_run) if latest_run else None,
                "run_count": len(claim.get("run_ids", [])),
            }
        )
    claims.sort(key=lambda item: item.get("text", ""))
    return jsonify({"claims": claims}), 200


@api.route('/api/v1/arenas/<arena_id>', methods=['GET'])
@auth_required
def get_arena_api(arena_id):
    arena = _claim_store().get_arena(arena_id)
    if not arena:
        return jsonify({"error": "Arena not found"}), 404
    return jsonify(arena), 200

@api.route('/api/v1/batch', methods=['POST'])
@auth_required
def start_batch_job():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    # No need to check password here as @auth_required does that for us already
    
    # Get search configuration
    num_queries = int(request.form.get('numQueries', 5))
    results_per_query = int(request.form.get('resultsPerQuery', 5))
    
    # Get bibliometric configuration
    bibliometric_config = {
        'use_bibliometrics': request.form.get('useBibliometrics', 'true').lower() == 'true',
        'author_impact_weight': float(request.form.get('authorImpactWeight', 0.4)),
        'citation_impact_weight': float(request.form.get('citationImpactWeight', 0.4)),
        'venue_impact_weight': float(request.form.get('venueImpactWeight', 0.2))
    }

    model_overrides = {
        "query_generation": request.form.get("model_query_generation", "").strip() or None,
        "paper_analysis": request.form.get("model_paper_analysis", "").strip() or None,
        "venue_scoring": request.form.get("model_venue_scoring", "").strip() or None,
        "final_report": request.form.get("model_final_report", "").strip() or None,
    }
    model_overrides = {k: v for k, v in model_overrides.items() if v}
    
    # Get email notification settings
    notification_email = request.form.get('email')
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.txt'):
        temp_batch_id = str(uuid.uuid4())[:8]
        batch_dir = _queued_jobs_dir() / temp_batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        file_path = batch_dir / 'claims.txt'
        file.save(file_path)
        with file_path.open('r', encoding='utf-8') as f:
            claims = [line.strip() for line in f if line.strip()]

        provider_id = request.form.get("providerId", "").strip() or "default"
        cost_confirmation = {
            "accepted": request.form.get("costConfirmationAccepted", "").strip().lower() == "true",
        }
        try:
            candidates = _submission_service().resolve_candidates(
                [
                    {
                        "provider_id": provider_id,
                        "model_overrides": model_overrides,
                    }
                ]
            )
            result = _submission_service().submit(
                claims=claims,
                candidates=candidates,
                search_config={
                    'num_queries': num_queries,
                    'results_per_query': results_per_query
                },
                bibliometric_config=bibliometric_config,
                batch_tags=[temp_batch_id],
                execution_mode="full_pipeline",
                cost_confirmation=cost_confirmation,
                duplicate_strategy=request.form.get("duplicateStrategy", "rerun"),
                create_arena=False,
                review_type="regular",
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        batch_id = result["batch_id"]
        actual_batch_dir = _queued_jobs_dir() / batch_id
        actual_batch_dir.mkdir(parents=True, exist_ok=True)
        try:
            if batch_dir != actual_batch_dir and file_path.exists():
                shutil.copy(file_path, actual_batch_dir / "claims.txt")
        except Exception:
            pass
        
        # Save notification settings
        if notification_email:
            notification_file = actual_batch_dir / 'notification.json'
            with notification_file.open('w', encoding='utf-8') as f:
                json.dump({
                    'email': notification_email,
                    'num_claims': len(claims)
                }, f)
        
        # Send start notification if email provided
        email_service.send_batch_start_notification(
            notification_email,
            batch_id,
            len(claims),
            'regular'
        )
        
        return jsonify({
            "batch_id": batch_id,
            "status": "processing",
            "created_runs": result["created_runs"],
            "reused_existing": result["reused_existing"],
        }), 202

@api.route('/api/v1/batch/<batch_id>/download', methods=['GET'])
@auth_required
def download_batch_reports(batch_id):
    batch_dir = _saved_jobs_dir() / batch_id
    if not batch_dir.exists():
        return jsonify({"error": "Batch not found"}), 404
    
    # Create a zip file of the batch directory
    zip_path = f"{batch_dir}.zip"
    shutil.make_archive(str(batch_dir), 'zip', str(batch_dir))
    return send_file(zip_path, as_attachment=True)

@api.route('/', methods=['GET'])
@auth_required
def index():
    saved_jobs_path = _saved_jobs_dir()
    saved_jobs_exist = saved_jobs_path.is_dir()
    return render_template('index.html', 
                         saved_jobs_exist=saved_jobs_exist,
                         config=current_app.config,
                         migration_status=_claim_store().migration_status(),
                         providers=_public_providers())

@api.route('/results', methods=['GET'])
@auth_required
def results():
    return render_template('results.html')


@api.route('/arena', methods=['GET'])
@auth_required
def arena_submit():
    return render_template(
        'arena.html',
        config=current_app.config,
        providers=_public_providers(),
        migration_status=_claim_store().migration_status(),
    )


@api.route('/arena_results', methods=['GET'])
@auth_required
def arena_results():
    arena_id = request.args.get('arena_id')
    if not arena_id:
        return "Arena not found", 404
    return render_template('arena_results.html', arena_id=arena_id, config=current_app.config)


@api.route('/providers', methods=['GET'])
@auth_required
def providers_page():
    return render_template('providers.html', config=current_app.config)


@api.route('/claims/<claim_key>', methods=['GET'])
@auth_required
def claim_detail_page(claim_key):
    return render_template(
        'claim_detail.html',
        claim_key=claim_key,
        config=current_app.config,
        providers=_public_providers(),
    )


@api.route('/claims/<batch_id>/<claim_id>/trace', methods=['GET'])
@auth_required
def trace_view(batch_id, claim_id):
    focus_trace_id = (request.args.get("focus_trace_id") or "").strip()
    return render_template(
        'trace_view.html',
        batch_id=batch_id,
        claim_id=claim_id,
        focus_trace_id=focus_trace_id,
    )

@api.route('/progress', methods=['GET'])
@auth_required
def progress():
    claim_id = request.args.get('claim_id')
    batch_id = request.args.get('batch_id')
    
    # If we have a claim_id, check its review type
    if claim_id:
        if batch_id:
            location, _, claim_data = _load_claim_data(batch_id, claim_id)
            if location == "saved_jobs" and claim_data:
                return render_template(_claim_results_template(claim_data), claim_id=claim_id)
        else:
            for root, dirs, files in os.walk(_saved_jobs_dir()):
                if f"{claim_id}.txt" in files:
                    with open(os.path.join(root, f"{claim_id}.txt"), 'r', encoding='utf-8') as f:
                        claim_data = json.load(f)
                        return render_template(_claim_results_template(claim_data), claim_id=claim_id)
    
    # Default to progress template for batches or not-found claims
    return render_template('progress.html', claim_id=claim_id, batch_id=batch_id)

@api.route('/api/v1/batch/<batch_id>', methods=['GET'])
@auth_required
def get_batch_status(batch_id):
    batch_state = _build_batch_state_view(batch_id)
    if batch_state is None or batch_state["total_claims"] == 0:
        return jsonify({"error": "Batch not found"}), 404

    return jsonify({
        "batch_id": batch_id,
        "status": batch_state["status"],
        "claims": [_serialize_batch_claim(claim) for claim in batch_state["claims"]],
        "all_runs": [_serialize_batch_claim(run) for run in batch_state.get("all_runs", [])],
        "review_type": batch_state["claims"][0].get("review_type", "regular"),
        "total_claims": batch_state["total_claims"],
        "processed_claims": batch_state["processed_claims"],
        "counts_by_status": batch_state["counts_by_status"],
        "counts_by_location": batch_state["counts_by_location"],
        "has_active_claims": batch_state["has_active_claims"],
        "has_partial_resume": batch_state["has_partial_resume"],
        "current_claim_id": batch_state["current_claim_id"],
        "errors": batch_state["errors"],
    }), 200

@api.route('/api/v1/batch/<batch_id>/progress', methods=['GET'])
@auth_required
def get_batch_progress(batch_id):
    """Get overall batch progress and detailed status breakdown."""
    batch_state = _build_batch_state_view(batch_id)
    detailed_counts = {
        "queued": 0,
        "ready_for_search": 0,
        "ready_for_analysis": 0,
        "processed": 0,
        "error": 0,
        "unknown": 0,
    }
    if batch_state is not None:
        for status_name, count in batch_state["counts_by_status"].items():
            if status_name in detailed_counts:
                detailed_counts[status_name] += count
            else:
                detailed_counts[status_name] = count

    return jsonify({
        "status": batch_state["status"] if batch_state is not None else "initializing",
        "total_claims": batch_state["total_claims"] if batch_state is not None else 0,
        "processed_claims": batch_state["processed_claims"] if batch_state is not None else 0,
        "current_claim_id": batch_state["current_claim_id"] if batch_state is not None else None,
        "detailed_counts": detailed_counts
    })

@api.route('/batch_results', methods=['GET'])
@auth_required
def batch_results():
    batch_id = request.args.get('batch_id')
    batch_state = _build_batch_state_view(batch_id) if batch_id else None
    if batch_state is None or batch_state["total_claims"] == 0:
        return "Batch not found", 404
    
    # Default to regular template
    template = 'batch_results.html'
    
    review_type = batch_state["claims"][0].get("review_type", "regular")
    if review_type == 'llm':
        template = 'llm_screen_batch_results.html'
    
    return render_template(template, batch_id=batch_id)

# Add these new routes

@api.route('/browser', methods=['GET'])
@auth_required
def browser():
    return render_template(
        'browser.html',
        config=current_app.config,
        migration_status=_claim_store().migration_status(),
    )

@api.route('/api/v1/browse', methods=['GET'])
@auth_required
def browse_batches():
    try:
        search_term = request.args.get('search', '').lower()
        batches = []

        saved_jobs_dir = _saved_jobs_dir()
        queued_jobs_dir = _queued_jobs_dir()
        if not saved_jobs_dir.exists() and not queued_jobs_dir.exists():
            return jsonify({
                "error": "No batch directories found",
                "code": "NO_SAVED_JOBS"
            }), 404

        batch_ids = set(
            list_batch_ids(
                saved_jobs_root=saved_jobs_dir,
                queued_jobs_root=queued_jobs_dir,
            )
        )
        batch_ids.update(_claim_store().list_batch_tags())

        for batch_id in sorted(batch_ids):
            try:
                batch_state = _build_batch_state_view(batch_id)
                if batch_state is None or batch_state["total_claims"] == 0:
                    continue

                batch_match = search_term and search_term in batch_id.lower()
                matching_claims = [
                    claim
                    for claim in batch_state["claims"]
                    if (
                        not search_term
                        or batch_match
                        or search_term in str(claim.get("text", "")).lower()
                        or search_term in str(claim.get("claim_id", "")).lower()
                    )
                ]
                if not matching_claims and not batch_match:
                    continue

                preview_claims = sorted(
                    matching_claims if search_term else batch_state["claims"],
                    key=lambda claim: (claim.get("location") != "queued_jobs", claim.get("claim_id")),
                )[:5]

                if preview_claims or batch_match or not search_term:
                    batches.append({
                        "batch_id": batch_id,
                        "timestamp": batch_state["timestamp"],
                        "updated_at": batch_state["updated_at"],
                        "status": batch_state["status"],
                        "total_claims": batch_state["total_claims"],
                        "processed_claims": batch_state["processed_claims"],
                        "counts_by_status": batch_state["counts_by_status"],
                        "counts_by_location": batch_state["counts_by_location"],
                        "has_active_claims": batch_state["has_active_claims"],
                        "has_partial_resume": batch_state["has_partial_resume"],
                        "preview_claims": [_serialize_batch_claim(claim) for claim in preview_claims],
                    })

            except Exception as e:
                logger.error(f"Error processing batch {batch_id}: {str(e)}")
                continue

        # Sort batches by timestamp, newest first
        batches.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'batches': batches})
        
    except Exception as e:
        logger.error(f"Error browsing batches: {str(e)}")
        return jsonify({
            "error": "Failed to browse batches",
            "code": "BROWSE_ERROR",
            "details": str(e)
        }), 500


@api.route('/api/v1/batches/export', methods=['POST'])
@auth_required
def export_batches():
    payload = request.get_json(silent=True) or {}
    raw_batch_ids = payload.get("batch_ids")
    if not isinstance(raw_batch_ids, list):
        return jsonify({
            "error": "batch_ids must be provided as a list",
            "code": "INVALID_BATCH_IDS",
        }), 400

    batch_ids = []
    seen = set()
    for value in raw_batch_ids:
        if not isinstance(value, str):
            continue
        batch_id = value.strip()
        if not batch_id or batch_id in seen:
            continue
        seen.add(batch_id)
        batch_ids.append(batch_id)

    if not batch_ids:
        return jsonify({
            "error": "At least one batch ID must be selected",
            "code": "NO_BATCH_IDS",
        }), 400

    include_artifacts = payload.get("include_artifacts", True)
    include_traces = bool(include_artifacts or payload.get("include_traces", False))
    include_issues = bool(include_artifacts or payload.get("include_issues", False))

    export_data = build_export_document(
        batch_ids=batch_ids,
        saved_jobs_root=_saved_jobs_dir(),
        queued_jobs_root=_queued_jobs_dir(),
        trace_root=_trace_root_dir(),
        include_traces=include_traces,
        include_issues=include_issues,
    )

    if not export_data["batches"]:
        return jsonify({
            "error": "No requested batches were found",
            "code": "BATCHES_NOT_FOUND",
            "missing_batches": export_data["missing_batches"],
        }), 404

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_label = "_".join(batch_ids[:3])
    if len(batch_ids) > 3:
        batch_label = f"{batch_label}_plus{len(batch_ids) - 3}"
    filename = f"valsci_batch_export_{batch_label}_{timestamp}.json"

    return Response(
        json.dumps(export_data, indent=2, ensure_ascii=True) + "\n",
        mimetype="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@api.route('/api/v1/claims/<batch_id>/<claim_id>/resume', methods=['POST'])
@auth_required
def resume_claim(batch_id, claim_id):
    location, source_path, claim_data = _load_claim_data(batch_id, claim_id)
    if not source_path or claim_data is None:
        return jsonify({
            "error": "Claim not found",
            "claim_id": claim_id,
            "code": "CLAIM_NOT_FOUND",
        }), 404

    if location == "queued_jobs":
        return jsonify({
            "error": "Claim is already queued or in progress",
            "claim_id": claim_id,
            "code": "CLAIM_ALREADY_QUEUED",
        }), 409

    queued_batch_dir = _queued_jobs_dir() / batch_id
    queued_batch_dir.mkdir(parents=True, exist_ok=True)
    queued_path = queued_batch_dir / f"{claim_id}.txt"
    if queued_path.exists():
        return jsonify({
            "error": "Claim is already queued or in progress",
            "claim_id": claim_id,
            "code": "CLAIM_ALREADY_QUEUED",
        }), 409

    resume_stage, resume_reason = _infer_resume_stage(claim_data)
    claim_data = dict(claim_data)
    claim_data["status"] = resume_stage
    claim_data["batch_id"] = batch_id
    claim_data["claim_id"] = claim_id
    claim_data.pop("report", None)

    with queued_path.open('w', encoding='utf-8') as f:
        json.dump(claim_data, f, indent=2)
    source_path_obj = Path(source_path)
    if source_path_obj != queued_path and source_path_obj.exists():
        source_path_obj.unlink()

    return jsonify({
        "message": "Claim resumed successfully",
        "batch_id": batch_id,
        "claim_id": claim_id,
        "resume_to_status": resume_stage,
        "resume_reason": resume_reason,
    }), 200

@api.route('/api/v1/delete/claim/<batch_id>/<claim_id>', methods=['DELETE'])
@api.route('/api/v1/delete/claim/<claim_id>', methods=['DELETE'])
@auth_required
def delete_claim(claim_id, batch_id=None):
    try:
        store = _claim_store()
        run_record = store.find_run_by_legacy(batch_id, claim_id) if batch_id else store.get_run(claim_id)
        if run_record:
            transport = run_record.get("transport") or {}
            for root in [_saved_jobs_dir(), _queued_jobs_dir()]:
                candidate = root / transport.get("batch_id", "") / f"{transport.get('claim_id', '')}.txt"
                if candidate.exists():
                    candidate.unlink()
            for artifact_value in [
                (run_record.get("artifact_paths") or {}).get("trace_file"),
                (run_record.get("artifact_paths") or {}).get("issues_file"),
            ]:
                if artifact_value:
                    artifact = Path(artifact_value)
                    if artifact.exists():
                        artifact.unlink()
            store.delete_run(run_record["run_id"])
            return jsonify({
                "message": "Claim deleted successfully",
                "claim_id": claim_id
            }), 200

        candidate_files = []
        saved_jobs_dir = _saved_jobs_dir()
        if batch_id:
            candidate_files.append(saved_jobs_dir / batch_id / f"{claim_id}.txt")
        else:
            for root, dirs, files in os.walk(saved_jobs_dir):
                if f"{claim_id}.txt" in files:
                    candidate_files.append(Path(root) / f"{claim_id}.txt")

        for file_path in candidate_files:
            if file_path.exists():
                file_path.unlink()
                return jsonify({
                    "message": "Claim deleted successfully",
                    "claim_id": claim_id
                }), 200
        return jsonify({
            "error": "Claim not found",
            "claim_id": claim_id,
            "code": "CLAIM_NOT_FOUND"
        }), 404
    except Exception as e:
        logger.error(f"Error deleting claim {claim_id}: {str(e)}")
        return jsonify({
            "error": "Failed to delete claim",
            "claim_id": claim_id,
            "code": "DELETE_ERROR",
            "details": str(e)
        }), 500

@api.route('/api/v1/delete/batch/<batch_id>', methods=['DELETE'])
@auth_required
def delete_batch(batch_id):
    try:
        deleted_runs = _claim_store().delete_runs_by_batch_tag(batch_id)
        deleted = False
        candidate_dirs = {
            _saved_jobs_dir() / batch_id,
            _queued_jobs_dir() / batch_id,
        }
        trace_root_dir = _trace_root_dir()
        if trace_root_dir not in {_saved_jobs_dir(), _queued_jobs_dir()}:
            candidate_dirs.add(trace_root_dir / batch_id)

        for batch_dir in candidate_dirs:
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
                deleted = True
        if deleted or deleted_runs:
            return jsonify({
                "message": "Batch deleted successfully",
                "batch_id": batch_id,
                "deleted_runs": deleted_runs,
            }), 200
        return jsonify({
            "error": "Batch not found",
            "batch_id": batch_id,
            "code": "BATCH_NOT_FOUND"
        }), 404
    except Exception as e:
        logger.error(f"Error deleting batch {batch_id}: {str(e)}")
        return jsonify({
            "error": "Failed to delete batch",
            "batch_id": batch_id,
            "code": "DELETE_ERROR",
            "details": str(e)
        }), 500

@api.route('/api/v1/claims/<batch_id>/<claim_id>/download_citations', methods=['GET'])
@api.route('/api/v1/claims/<claim_id>/download_citations', methods=['GET'])
@auth_required
def download_citations(claim_id, batch_id=None):
    try:
        candidate_files = []
        saved_jobs_dir = _saved_jobs_dir()
        if batch_id:
            candidate_files.append(saved_jobs_dir / batch_id / f"{claim_id}.txt")
        else:
            for root, dirs, files in os.walk(saved_jobs_dir):
                if f"{claim_id}.txt" in files:
                    candidate_files.append(Path(root) / f"{claim_id}.txt")

        for claim_file in candidate_files:
            if not claim_file.exists():
                continue
            with claim_file.open('r', encoding='utf-8') as f:
                claim_data = json.load(f)
                report = claim_data.get('report', {})
                citations = []

                for paper in report.get('relevantPapers', []):
                    for citation in paper.get('citations', []):
                        if citation.get('citation'):
                            citations.append(citation['citation'])

                citation_file_path = saved_jobs_dir / f"{claim_id}_citations.ris"
                with citation_file_path.open('w', encoding='utf-8') as citation_file:
                    citation_file.write("\n\n".join(citations))
                return send_file(citation_file_path, as_attachment=True, download_name=f"{claim_id}_citations.ris")
        
        return jsonify({"error": "Claim not found"}), 404
    finally:
        if 'citation_file_path' in locals() and citation_file_path.exists():
            citation_file_path.unlink()

def generate_markdown_report(claim_data):
    """Helper function to generate consistent markdown reports"""
    report = claim_data.get('report', {})
    md_content = []
    
    # Basic info
    md_content.append(f"# Claim: {claim_data.get('text', '')}\n")
    md_content.append(f"**Status**: {claim_data.get('status', '')}\n")
    md_content.append(f"**Overall Rating**: {report.get('claimRating', 'N/A')}\n")
    md_content.append(f"**Explanation**:\n\n{report.get('explanation', 'No explanation available')}\n")
    
    # Add final reasoning if it exists
    if report.get('finalReasoning'):
        md_content.append(f"\n**Final Reasoning**:\n\n{report['finalReasoning']}\n")
    
    # Check if bibliometrics are enabled
    bibliometric_config = report.get('bibliometric_config', {})
    use_bibliometrics = bibliometric_config.get('use_bibliometrics', True) if bibliometric_config else True
    
    # Add relevant papers section
    md_content.append("\n## Relevant Papers\n")
    for paper in report.get('relevantPapers', []):
        md_content.append(f"\n### {paper.get('title', 'Untitled Paper')}\n")
        if paper.get('authors'):
            authors = ', '.join([f"{a.get('name')} (H-index: {a.get('hIndex', 'N/A')})" 
                               for a in paper['authors']])
            md_content.append(f"**Authors**: {authors}\n")
        md_content.append(f"**Relevance**: {paper.get('relevance', 'N/A')}\n")
        
        # Only show bibliometric impact if enabled
        if use_bibliometrics and 'bibliometric_impact' in paper:
            md_content.append(f"**Bibliometric Impact**: {paper.get('bibliometric_impact', 'N/A')}\n")
        
        if paper.get('excerpts'):
            md_content.append("\n**Excerpts**:\n")
            for excerpt in paper['excerpts']:
                md_content.append(f"- {excerpt}\n")
        
        if paper.get('explanations'):
            md_content.append("\n**Explanations**:\n")
            for explanation in paper['explanations']:
                md_content.append(f"- {explanation}\n")
        
        if paper.get('link'):
            md_content.append(f"\n[Read Paper]({paper['link']})\n")
    
    # Add non-relevant papers section
    if report.get('nonRelevantPapers'):
        md_content.append("\n## Other Reviewed Papers\n")
        for paper in report['nonRelevantPapers']:
            md_content.append(f"\n### {paper.get('title', 'Untitled Paper')}\n")
            if paper.get('explanation'):
                md_content.append(f"**Why Not Relevant**: {paper['explanation']}\n")
            if paper.get('link'):
                md_content.append(f"\n[Read Paper]({paper['link']})\n")
    
    # Add search queries section
    if report.get('searchQueries'):
        md_content.append("\n## Search Queries Used\n")
        for query in report['searchQueries']:
            md_content.append(f"- {query}\n")
    
    # Add usage stats if available
    usage = report.get('usage_summary') or report.get('usage_stats')
    if usage:
        md_content.append("\n## Usage Statistics\n")
        md_content.append(f"- Input Tokens: {usage.get('input_tokens', 0)}\n")
        md_content.append(f"- Output Tokens: {usage.get('output_tokens', 0)}\n")
        md_content.append(f"- Total Tokens: {usage.get('total_tokens', 0)}\n")
        md_content.append(f"- Estimated Cost: ${usage.get('cost_usd', 0):.4f}\n")
        md_content.append(f"- Token Counts Estimated: {usage.get('is_estimated', False)}\n")

    usage_by_stage = report.get('usage_by_stage')
    if usage_by_stage:
        md_content.append("\n## Usage By Stage\n")
        for stage, stats in usage_by_stage.items():
            md_content.append(
                f"- {stage}: in={stats.get('input_tokens', 0)} out={stats.get('output_tokens', 0)} "
                f"total={stats.get('total_tokens', 0)} cost=${stats.get('cost_usd', 0):.4f}\n"
            )

    if report.get('issues'):
        md_content.append("\n## Issues\n")
        for issue in report['issues']:
            md_content.append(
                f"- [{issue.get('severity', 'INFO')}] {issue.get('stage', 'system')}: {issue.get('message', '')}\n"
            )

    debug_trace = report.get('debug_trace')
    if debug_trace:
        summary = debug_trace.get('summary', {})
        md_content.append("\n## Debug Trace Summary\n")
        md_content.append(f"- LLM Calls: {summary.get('llm_calls', 0)}\n")
        md_content.append(f"- Models Used: {', '.join(summary.get('models_used', []))}\n")
        md_content.append(f"- Retries: {summary.get('retries', 0)}\n")
        md_content.append(
            f"- Context Overflow Prevented: {summary.get('context_overflow_prevented', 0)}\n"
        )
        md_content.append(f"- Trace File: {debug_trace.get('trace_file', '')}\n")
    
    # Add bibliometric configuration if available
    if bibliometric_config:
        md_content.append("\n## Bibliometric Configuration\n")
        md_content.append(f"- Use Bibliometrics: {bibliometric_config.get('use_bibliometrics', True)}\n")
        if use_bibliometrics:
            md_content.append(f"- Author Impact Weight: {bibliometric_config.get('author_impact_weight', 0.4)}\n")
            md_content.append(f"- Citation Impact Weight: {bibliometric_config.get('citation_impact_weight', 0.4)}\n")
            md_content.append(f"- Venue Impact Weight: {bibliometric_config.get('venue_impact_weight', 0.2)}\n")
    
    return "\n".join(md_content)

@api.route('/api/v1/claims/<batch_id>/<claim_id>/download_md', methods=['GET'])
@auth_required
def download_claim_md(batch_id, claim_id):
    """Download a single claim's final report as a markdown (.md) file."""
    saved_file = _saved_jobs_dir() / batch_id / f"{claim_id}.txt"
    if not saved_file.exists():
        return jsonify({"error": "Claim not found"}), 404

    with saved_file.open('r', encoding='utf-8') as f:
        claim_data = json.load(f)
    
    md_text = generate_markdown_report(claim_data)
    
    return Response(
        md_text,
        mimetype="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=claim_{claim_id}.md"
        }
    )

@api.route('/api/v1/batch/<batch_id>/download_markdown', methods=['GET'])
@auth_required
def download_batch_markdown(batch_id):
    """Download a zip of markdown files for all claims in a batch."""
    batch_state = _build_batch_state_view(batch_id)
    if batch_state is None or batch_state["total_claims"] == 0:
        return jsonify({"error": "Batch not found"}), 404
    if batch_state["status"] != "completed":
        return jsonify({
            "error": "Batch is still processing; markdown export is only available for completed batches",
            "code": "BATCH_NOT_COMPLETED",
            "status": batch_state["status"],
        }), 409

    import io
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for claim in batch_state["claims"]:
            claim_id = claim["claim_id"]
            claim_data = claim.get("claim_data")
            if not isinstance(claim_data, dict) or not claim_data:
                location, _, claim_data = _load_claim_data(batch_id, claim_id)
                if location is None or claim_data is None:
                    return jsonify(
                        {
                            "error": f"Claim data missing for {claim_id}",
                            "code": "CLAIM_DATA_MISSING",
                            "claim_id": claim_id,
                        }
                    ), 500
            md_text = generate_markdown_report(claim_data)
            zf.writestr(f"claim_{claim_id}.txt", md_text)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"batch_{batch_id}_reports.zip"
    )

@api.route('/login', methods=['GET', 'POST'])
def login():
    # Skip login if password is not required
    if not current_app.config['REQUIRE_PASSWORD']:
        return redirect(url_for('api.index'))
    
    error = None
    if request.method == 'POST':
        password = request.form.get('password')
        if verify_password(password):
            session['authenticated'] = True
            next_page = request.args.get('next', url_for('api.index'))
            return redirect(next_page)
        else:
            error = 'Invalid password. Please try again.'
    
    return render_template('login.html', error=error)

@api.route('/logout', methods=['GET'])
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('api.login'))
