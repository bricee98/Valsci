from flask import Blueprint, request, jsonify, send_file, render_template, current_app, Response
import os
import uuid
import shutil
import threading
import json
from app.services.claim_processor import ClaimProcessor
from app.models.claim import Claim
from app.models.batch_job import BatchJob
from app.models.paper import Paper
from app.services.literature_searcher import LiteratureSearcher
from datetime import datetime
import asyncio
from typing import List
import math
from app.services.email_service import EmailService
import logging
import traceback

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
            "additional_info": ""
        }, f, indent=2)

def verify_password(password):
    if current_app.config['REQUIRE_PASSWORD']:
        if not password:
            return False
        return password == current_app.config['ACCESS_PASSWORD']
    return True

@api.route('/api/v1/claims/<batch_id>/<claim_id>', methods=['GET'])
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
def start_batch_job():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    if not verify_password(request.form.get('password')):
        return jsonify({"error": "Invalid password"}), 403
    
    # Get search configuration
    num_queries = int(request.form.get('numQueries', 5))
    results_per_query = int(request.form.get('resultsPerQuery', 5))
    
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
            save_claim_to_file(claim, batch_id, claim_id)
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

def update_batch_progress(batch_id: str, processed_claims: int, total_claims: int, status: str = "processing", current_claim_id: str = None):
    progress_file = os.path.join(SAVED_JOBS_DIR, batch_id, 'progress.json')
    with open(progress_file, 'w') as f:
        json.dump({
            "processed_claims": processed_claims,
            "total_claims": total_claims,
            "status": status,
            "current_claim_id": current_claim_id
        }, f)

@api.route('/api/v1/batch/<batch_id>/download', methods=['GET'])
def download_batch_reports(batch_id):
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if not os.path.exists(batch_dir):
        return jsonify({"error": "Batch not found"}), 404
    
    # Create a zip file of the batch directory
    zip_path = f"{batch_dir}.zip"
    shutil.make_archive(batch_dir, 'zip', batch_dir)
    return send_file(zip_path, as_attachment=True)

@api.route('/', methods=['GET'])
def index():
    saved_jobs_path = os.path.join(current_app.root_path, '..', 'saved_jobs')
    saved_jobs_exist = os.path.isdir(saved_jobs_path)
    return render_template('index.html', 
                         saved_jobs_exist=saved_jobs_exist,
                         config=current_app.config)

@api.route('/results', methods=['GET'])
def results():
    return render_template('results.html')

@api.route('/progress', methods=['GET'])
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
def get_batch_progress(batch_id):
    progress_file = os.path.join(SAVED_JOBS_DIR, batch_id, 'progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Batch not found"}), 404

@api.route('/batch_results', methods=['GET'])
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
def browser():
    return render_template('browser.html')

@api.route('/api/v1/browse', methods=['GET'])
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
                
                # Rest of the existing code...
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_id}: {str(e)}")
                continue

        return jsonify({'batches': batches})
    except Exception as e:
        logger.error(f"Error browsing batches: {str(e)}")
        return jsonify({
            "error": "Failed to browse batches",
            "code": "BROWSE_ERROR",
            "details": str(e)
        }), 500

@api.route('/api/v1/delete/claim/<claim_id>', methods=['DELETE'])
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

async def process_claim_post_search(claim_id: str, papers: List[Paper], batch_id: str, sem: asyncio.Semaphore, claim_text: str):
    """Process a claim after papers have been retrieved."""
    async with sem:  # Limit concurrent processing
        try:
            # Initialize claim processor
            claim_processor = ClaimProcessor()
            
            # Save initial claim status
            save_claim_to_file(
                Claim(text=claim_text),  # We need the original claim text
                batch_id, 
                claim_id
            )
            
            # Update status to show we're analyzing papers
            update_batch_progress(
                batch_id, 
                0,  # We'll update this as we process
                1,  # This is a single claim
                current_claim_id=claim_id
            )
            
            # Process the papers (we'll need to modify ClaimProcessor for this)
            await claim_processor.process_claim_with_papers(
                papers=papers,
                batch_id=batch_id,
                claim_id=claim_id,
                claim_text=claim_text
            )
            
            # Update progress to show completion
            update_batch_progress(
                batch_id,
                1,  # This claim is done
                1,  # Total is still 1
                current_claim_id=claim_id
            )
            
        except Exception as e:
            logger.error(f"Error processing claim {claim_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update status to show error
            update_batch_progress(
                batch_id,
                1,  # Mark as processed even though it errored
                1,
                status="error",
                current_claim_id=claim_id
            )
