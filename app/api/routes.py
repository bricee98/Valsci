from flask import Blueprint, request, jsonify, send_file, render_template, current_app, Response
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

api = Blueprint('api', __name__)

QUEUED_JOBS_DIR = 'queued_jobs'
SAVED_JOBS_DIR = 'saved_jobs'

logger = logging.getLogger(__name__)

# Initialize EmailService at module level
email_service = EmailService()


def _trace_root_dir() -> str:
    trace_dir = current_app.config.get('TRACE_DIR', SAVED_JOBS_DIR)
    return trace_dir or SAVED_JOBS_DIR


def _find_claim_artifact(batch_id: str, artifact_dir: str, claim_id: str, extension: str):
    """Resolve trace/issue artifact path with TRACE_DIR support and saved_jobs fallback."""
    candidates = [
        os.path.join(_trace_root_dir(), batch_id, artifact_dir, f"{claim_id}.{extension}"),
        os.path.join(_trace_root_dir(), batch_id, artifact_dir, f"{claim_id}.{extension}.gz"),
    ]
    # Backward compatibility fallback
    candidates.extend(
        [
            os.path.join(SAVED_JOBS_DIR, batch_id, artifact_dir, f"{claim_id}.{extension}"),
            os.path.join(SAVED_JOBS_DIR, batch_id, artifact_dir, f"{claim_id}.{extension}.gz"),
        ]
    )

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _read_jsonl_artifact(path: str) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid_lines = 0
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
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
        ("queued_jobs", os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt")),
        ("saved_jobs", os.path.join(SAVED_JOBS_DIR, batch_id, f"{claim_id}.txt")),
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
    return build_batch_state(
        batch_id=batch_id,
        saved_jobs_root=Path(SAVED_JOBS_DIR),
        queued_jobs_root=Path(QUEUED_JOBS_DIR),
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
        "text": claim.get("text", ""),
        "status": claim.get("status", "unknown"),
        "report": claim.get("report", {}),
        "review_type": claim.get("review_type", "regular"),
        "claim_location": claim.get("location"),
        "report_available": bool(claim.get("report_available", False)),
        "is_active": bool(claim.get("is_active", False)),
        "rating": claim.get("rating"),
    }

def save_claim_to_file(claim, batch_id, claim_id):
    claim_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
    os.makedirs(claim_dir, exist_ok=True)
    claim_file = os.path.join(claim_dir, f"{claim_id}.txt")
    
    # Ensure claim.text is a string, not a list
    claim_text = claim.text[0] if isinstance(claim.text, list) else claim.text
    
    with open(claim_file, 'w') as f:
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
    claim_file = os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt")
    if os.path.exists(claim_file):
        with open(claim_file, 'r') as f:
            claim_data = json.load(f)
            return jsonify({
                "claim_id": claim_id,
                "text": claim_data.get('text', ''),
                "status": claim_data.get('status', 'Unknown'),
                "additional_info": claim_data.get('additional_info', {}),
                "review_type": claim_data.get('review_type', 'regular')
            }), 200
    
    # Then check saved jobs
    claim_file = os.path.join(SAVED_JOBS_DIR, batch_id, f"{claim_id}.txt")
    if os.path.exists(claim_file):
        with open(claim_file, 'r') as f:
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
    if not trace_file or not os.path.exists(trace_file):
        return jsonify({"error": "Trace not found"}), 404
    download_name = f"{claim_id}_trace.jsonl.gz" if trace_file.endswith(".gz") else f"{claim_id}_trace.jsonl"
    return send_file(trace_file, as_attachment=True, download_name=download_name)


@api.route('/api/v1/claims/<batch_id>/<claim_id>/trace_records', methods=['GET'])
@auth_required
def get_claim_trace_records(batch_id, claim_id):
    trace_file = _find_claim_artifact(batch_id, "traces", claim_id, "jsonl")
    if not trace_file or not os.path.exists(trace_file):
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
        "compressed": trace_file.endswith(".gz"),
        "trace_file": f"traces/{claim_id}.jsonl" + (".gz" if trace_file.endswith(".gz") else ""),
        "records": records,
        **claim_metadata,
    }), 200

@api.route('/api/v1/claims/<batch_id>/<claim_id>/issues', methods=['GET'])
@auth_required
def download_claim_issues(batch_id, claim_id):
    issue_file = _find_claim_artifact(batch_id, "issues", claim_id, "jsonl")
    if not issue_file or not os.path.exists(issue_file):
        return jsonify({"error": "Issues not found"}), 404
    download_name = f"{claim_id}_issues.jsonl.gz" if issue_file.endswith(".gz") else f"{claim_id}_issues.jsonl"
    return send_file(issue_file, as_attachment=True, download_name=download_name)

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
        batch_id = str(uuid.uuid4())[:8]
        batch_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save claims file
        file_path = os.path.join(batch_dir, 'claims.txt')
        file.save(file_path)
        
        # Read claims
        with open(file_path, 'r', encoding='utf-8') as f:
            claims = [line.strip() for line in f if line.strip()]
        
        # Create claims with search config and save them
        batch_claims = []
        claim_ids = {}  # Map to store claim_text -> claim_id
        for claim_text in claims:
            claim = Claim(text=claim_text)
            claim.search_config = {
                'num_queries': num_queries,
                'results_per_query': results_per_query
            }
            claim_id = str(uuid.uuid4())[:8]
            claim_ids[claim_text] = claim_id
            
            # Save claim to file with bibliometric config
            claim_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
            os.makedirs(claim_dir, exist_ok=True)
            claim_file = os.path.join(claim_dir, f"{claim_id}.txt")
            
            # Ensure claim.text is a string, not a list
            claim_text = claim.text[0] if isinstance(claim.text, list) else claim.text
            
            with open(claim_file, 'w') as f:
                json.dump({
                    "text": claim_text,
                    "status": "queued",
                    "batch_id": batch_id,
                    "claim_id": claim_id,
                    "search_config": claim.search_config,
                    "bibliometric_config": bibliometric_config,
                    "model_overrides": model_overrides,
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
            
            batch_claims.append(claim)
        
        batch_job = BatchJob(claims=batch_claims)
        
        # Store claim_ids mapping for the batch
        with open(os.path.join(batch_dir, 'claim_ids.json'), 'w') as f:
            json.dump(claim_ids, f)
        
        # Save notification settings
        if notification_email:
            notification_file = os.path.join(batch_dir, 'notification.json')
            with open(notification_file, 'w') as f:
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
            "status": "processing"
        }), 202

@api.route('/api/v1/batch/<batch_id>/download', methods=['GET'])
@auth_required
def download_batch_reports(batch_id):
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if not os.path.exists(batch_dir):
        return jsonify({"error": "Batch not found"}), 404
    
    # Create a zip file of the batch directory
    zip_path = f"{batch_dir}.zip"
    shutil.make_archive(batch_dir, 'zip', batch_dir)
    return send_file(zip_path, as_attachment=True)

@api.route('/', methods=['GET'])
@auth_required
def index():
    saved_jobs_path = os.path.join(current_app.root_path, '..', 'saved_jobs')
    saved_jobs_exist = os.path.isdir(saved_jobs_path)
    return render_template('index.html', 
                         saved_jobs_exist=saved_jobs_exist,
                         config=current_app.config)

@api.route('/results', methods=['GET'])
@auth_required
def results():
    return render_template('results.html')


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
            for root, dirs, files in os.walk(SAVED_JOBS_DIR):
                if f"{claim_id}.txt" in files:
                    with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
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
    return render_template('browser.html')

@api.route('/api/v1/browse', methods=['GET'])
@auth_required
def browse_batches():
    try:
        search_term = request.args.get('search', '').lower()
        batches = []

        if not os.path.exists(SAVED_JOBS_DIR) and not os.path.exists(QUEUED_JOBS_DIR):
            return jsonify({
                "error": "No batch directories found",
                "code": "NO_SAVED_JOBS"
            }), 404

        for batch_id in list_batch_ids(
            saved_jobs_root=Path(SAVED_JOBS_DIR),
            queued_jobs_root=Path(QUEUED_JOBS_DIR),
        ):
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
        saved_jobs_root=Path(SAVED_JOBS_DIR),
        queued_jobs_root=Path(QUEUED_JOBS_DIR),
        trace_root=Path(_trace_root_dir()),
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

    queued_batch_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
    os.makedirs(queued_batch_dir, exist_ok=True)
    queued_path = os.path.join(queued_batch_dir, f"{claim_id}.txt")
    if os.path.exists(queued_path):
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

    with open(queued_path, 'w', encoding='utf-8') as f:
        json.dump(claim_data, f, indent=2)
    if source_path != queued_path and os.path.exists(source_path):
        os.remove(source_path)

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
        candidate_files = []
        if batch_id:
            candidate_files.append(os.path.join(SAVED_JOBS_DIR, batch_id, f"{claim_id}.txt"))
        else:
            for root, dirs, files in os.walk(SAVED_JOBS_DIR):
                if f"{claim_id}.txt" in files:
                    candidate_files.append(os.path.join(root, f"{claim_id}.txt"))

        for file_path in candidate_files:
            if os.path.exists(file_path):
                os.remove(file_path)
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
        deleted = False
        candidate_dirs = {
            os.path.join(SAVED_JOBS_DIR, batch_id),
            os.path.join(QUEUED_JOBS_DIR, batch_id),
        }
        trace_root_dir = _trace_root_dir()
        if trace_root_dir not in {SAVED_JOBS_DIR, QUEUED_JOBS_DIR}:
            candidate_dirs.add(os.path.join(trace_root_dir, batch_id))

        for batch_dir in candidate_dirs:
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
                deleted = True
        if deleted:
            return jsonify({
                "message": "Batch deleted successfully",
                "batch_id": batch_id
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
        if batch_id:
            candidate_files.append(os.path.join(SAVED_JOBS_DIR, batch_id, f"{claim_id}.txt"))
        else:
            for root, dirs, files in os.walk(SAVED_JOBS_DIR):
                if f"{claim_id}.txt" in files:
                    candidate_files.append(os.path.join(root, f"{claim_id}.txt"))

        for claim_file in candidate_files:
            if not os.path.exists(claim_file):
                continue
            with open(claim_file, 'r') as f:
                claim_data = json.load(f)
                report = claim_data.get('report', {})
                citations = []

                for paper in report.get('relevantPapers', []):
                    for citation in paper.get('citations', []):
                        if citation.get('citation'):
                            citations.append(citation['citation'])

                citation_file_path = os.path.join(SAVED_JOBS_DIR, f"{claim_id}_citations.ris")
                with open(citation_file_path, 'w') as citation_file:
                    citation_file.write("\n\n".join(citations))
                return send_file(citation_file_path, as_attachment=True, download_name=f"{claim_id}_citations.ris")
        
        return jsonify({"error": "Claim not found"}), 404
    finally:
        if 'citation_file_path' in locals() and os.path.exists(citation_file_path):
            os.remove(citation_file_path)

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
    saved_file = os.path.join(SAVED_JOBS_DIR, batch_id, f"{claim_id}.txt")
    if not os.path.exists(saved_file):
        return jsonify({"error": "Claim not found"}), 404

    with open(saved_file, 'r') as f:
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
            claim_data = claim["claim_data"]
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
