from flask import Blueprint, request, jsonify, send_file, render_template, current_app, Response
from flask import session, redirect, url_for, flash
import os
import uuid
import shutil
import threading
import json
from app.services.claim_processor import ClaimProcessor
from app.models.claim import Claim
from app.models.batch_job import BatchJob
from app.models.paper import Paper
from datetime import datetime
import asyncio
from typing import List
import math
from app.services.email_service import EmailService
import logging
import traceback
from functools import wraps

api = Blueprint('api', __name__)

QUEUED_JOBS_DIR = 'queued_jobs'
SAVED_JOBS_DIR = 'saved_jobs'

logger = logging.getLogger(__name__)

# Initialize EmailService at module level
email_service = EmailService()

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
    claim_file = os.path.join(SAVED_JOBS_DIR, batch_id, f"{claim_id}.txt")
    if os.path.exists(claim_file):
        with open(claim_file, 'r') as f:
            claim_data = json.load(f)
            return jsonify({
                "claim_id": claim_id,
                "text": claim_data.get('text', ''),
                "status": claim_data.get('status', ''),
                "report": claim_data.get('report', {})  # Get report directly
            }), 200
    return jsonify({"error": "Claim not found"}), 404

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

@api.route('/progress', methods=['GET'])
@auth_required
def progress():
    claim_id = request.args.get('claim_id')
    batch_id = request.args.get('batch_id')
    
    # If we have a claim_id, check its review type
    if claim_id:
        for root, dirs, files in os.walk(SAVED_JOBS_DIR):
            if f"{claim_id}.txt" in files:
                with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
                    claim_data = json.load(f)
                    try:
                        additional_info = json.loads(claim_data.get('additional_info', '{}'))
                        if 'overall_rating' in additional_info and 'plausibility_level' in additional_info:
                            return render_template('llm_screen_results.html', claim_id=claim_id)
                    except json.JSONDecodeError:
                        pass
                    
                    # Regular claim
                    return render_template('results.html', claim_id=claim_id)
    
    # Default to progress template for batches or not-found claims
    return render_template('progress.html', claim_id=claim_id, batch_id=batch_id)

@api.route('/api/v1/batch/<batch_id>', methods=['GET'])
@auth_required
def get_batch_status(batch_id):
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if not os.path.exists(batch_dir):
        return jsonify({"error": "Batch not found"}), 404
    
    claims = []
    for file in os.listdir(batch_dir):
        if file.endswith('.txt') and not file.endswith('claims.txt'):
            file_path = os.path.join(batch_dir, file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        claims.append({
                            "claim_id": file[:-4],
                            "text": "",
                            "status": "Error",
                            "report": {"error": "Empty file"}
                        })
                    else:
                        claim_data = json.loads(content)
                        claims.append({
                            "claim_id": file[:-4],
                            "text": claim_data.get('text', ''),
                            "status": claim_data.get('status', 'Unknown'),
                            "report": claim_data.get('report', {}),  # Changed from additional_info to report
                            "review_type": claim_data.get('review_type', 'regular')
                        })
            except json.JSONDecodeError as e:
                claims.append({
                    "claim_id": file[:-4],
                    "text": "",
                    "status": "Error",
                    "report": {"error": f"Invalid JSON in file: {str(e)}"}
                })
            except Exception as e:
                claims.append({
                    "claim_id": file[:-4],
                    "text": "",
                    "status": "Error",
                    "report": {"error": f"Error reading file: {str(e)}"}
                })
    
    overall_status = "processed" if all(claim['status'] == 'processed' for claim in claims) else "processing"
    
    return jsonify({
        "batch_id": batch_id,
        "status": overall_status,
        "claims": claims,
        "review_type": "regular"  # Default to regular review type
    }), 200

@api.route('/api/v1/batch/<batch_id>/progress', methods=['GET'])
@auth_required
def get_batch_progress(batch_id):
    """Get overall batch progress and detailed status breakdown."""
    detailed_counts = {
        "queued": 0,
        "ready_for_search": 0, 
        "ready_for_analysis": 0,
        "processed": 0,
        "error": 0,
        "unknown": 0
    }
    
    total_claims = 0
    processed_claims = 0
    current_claim_id = None
    
    # Process queued jobs
    queued_batch_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
    if os.path.exists(queued_batch_dir):
        for filename in os.listdir(queued_batch_dir):
            if filename.endswith('.txt') and filename != 'claims.txt':
                total_claims += 1
                file_path = os.path.join(queued_batch_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    status = data.get('status', 'unknown')
                    if status in detailed_counts:
                        detailed_counts[status] += 1
                    else:
                        detailed_counts["unknown"] += 1
                        
                    # Track currently processing claim
                    if status not in ('processed', 'error'):
                        current_claim_id = filename[:-4]  # Remove .txt
                except Exception as e:
                    detailed_counts["unknown"] += 1
    
    # Process saved jobs
    saved_batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if os.path.exists(saved_batch_dir):
        for filename in os.listdir(saved_batch_dir):
            if filename.endswith('.txt') and filename != 'claims.txt':
                total_claims += 1
                processed_claims += 1
                file_path = os.path.join(saved_batch_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    status = data.get('status', 'unknown')
                    if status in detailed_counts:
                        detailed_counts[status] += 1
                    else:
                        detailed_counts["unknown"] += 1
                except Exception as e:
                    detailed_counts["unknown"] += 1
    
    # Calculate overall status
    status = "processing"
    if total_claims == 0:
        status = "initializing"
    elif processed_claims == total_claims:
        status = "completed"
    elif detailed_counts["error"] == total_claims:
        status = "error"
    
    return jsonify({
        "status": status,
        "total_claims": total_claims,
        "processed_claims": processed_claims,
        "current_claim_id": current_claim_id,
        "detailed_counts": detailed_counts
    })

@api.route('/batch_results', methods=['GET'])
@auth_required
def batch_results():
    batch_id = request.args.get('batch_id')
    # Fetch batch information to determine review_type
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if not os.path.exists(batch_dir):
        return "Batch not found", 404
    
    # Default to regular template
    template = 'batch_results.html'
    
    # Read the first claim to determine the review_type
    for file in os.listdir(batch_dir):
        if file.endswith('.txt') and not file.endswith('claims.txt'):
            with open(os.path.join(batch_dir, file), 'r') as f:
                claim_data = json.load(f)
                review_type = claim_data.get('review_type', 'regular')
                if review_type == 'llm':
                    template = 'llm_screen_batch_results.html'
                break  # Only need to check the first claim
    
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

        if not os.path.exists(SAVED_JOBS_DIR):
            return jsonify({
                "error": "No saved jobs directory found",
                "code": "NO_SAVED_JOBS"
            }), 404

        for batch_id in os.listdir(SAVED_JOBS_DIR):
            try:
                batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
                if not os.path.isdir(batch_dir):
                    continue

                # Get batch timestamp from the oldest file in the directory
                batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.txt')]
                if not batch_files:
                    continue

                # Get timestamp from the first file's modification time
                first_file = os.path.join(batch_dir, batch_files[0])
                timestamp = datetime.fromtimestamp(os.path.getmtime(first_file)).isoformat()

                # Get preview claims (up to 5)
                preview_claims = []
                for filename in batch_files[:5]:  # Limit to first 5 files
                    if filename == 'claims.txt':
                        continue
                        
                    file_path = os.path.join(batch_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            claim_data = json.load(f)
                            
                            # Get rating from report if it exists
                            rating = None
                            if 'report' in claim_data:
                                rating = claim_data['report'].get('claimRating')
                            
                            claim_info = {
                                "claim_id": filename[:-4],  # Remove .txt
                                "text": claim_data.get('text', ''),
                                "status": claim_data.get('status', 'Unknown'),
                                "rating": rating
                            }
                            
                            # Only add if matches search term
                            if (not search_term or 
                                search_term in claim_info['text'].lower() or
                                search_term in batch_id.lower()):
                                preview_claims.append(claim_info)
                    except Exception as e:
                        logger.error(f"Error reading claim file {file_path}: {str(e)}")
                        continue

                # Only add batch if it has matching claims or batch ID matches search
                if preview_claims or (search_term and search_term in batch_id.lower()):
                    batches.append({
                        "batch_id": batch_id,
                        "timestamp": timestamp,
                        "total_claims": len(batch_files),
                        "preview_claims": preview_claims
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

@api.route('/api/v1/delete/claim/<claim_id>', methods=['DELETE'])
@auth_required
def delete_claim(claim_id):
    try:
        for root, dirs, files in os.walk(SAVED_JOBS_DIR):
            if f"{claim_id}.txt" in files:
                os.remove(os.path.join(root, f"{claim_id}.txt"))
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
        batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
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

@api.route('/api/v1/claims/<claim_id>/download_citations', methods=['GET'])
@auth_required
def download_citations(claim_id):
    try:
        # Find and load the claim data
        for root, dirs, files in os.walk(SAVED_JOBS_DIR):
            if f"{claim_id}.txt" in files:
                with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
                    claim_data = json.load(f)
                    report = json.loads(claim_data.get('additional_info', '{}'))
                    citations = []
                    
                    for paper in report.get('supportingPapers', []):
                        for citation in paper.get('citations', []):
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
    if report.get('usage_stats'):
        stats = report['usage_stats']
        md_content.append("\n## Usage Statistics\n")
        md_content.append(f"- Prompt Tokens: {stats.get('prompt_tokens', 0)}\n")
        md_content.append(f"- Completion Tokens: {stats.get('completion_tokens', 0)}\n")
        md_content.append(f"- Total Tokens: {stats.get('total_tokens', 0)}\n")
        md_content.append(f"- Estimated Cost: ${stats.get('total_cost', 0):.4f}\n")
    
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
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if not os.path.exists(batch_dir):
        return jsonify({"error": "Batch not found"}), 404

    claim_files = [f for f in os.listdir(batch_dir) if f.endswith('.txt') and f != 'claims.txt']
    if not claim_files:
        return jsonify({"error": "No processed claims found"}), 404

    import io
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for file_name in claim_files:
            claim_id = file_name[:-4]  # remove .txt
            saved_file = os.path.join(batch_dir, file_name)
            with open(saved_file, 'r') as f:
                claim_data = json.load(f)
            
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
