from flask import Blueprint, request, jsonify, send_file, render_template, current_app, Response
import os
import uuid
import shutil
import threading
import json
from app.services.claim_processor import ClaimProcessor
from app.models.claim import Claim
from app.models.batch_job import BatchJob
from datetime import datetime
import asyncio
from typing import List
import math
from app.services.openai_service import OpenAIService

api = Blueprint('api', __name__)

SAVED_JOBS_DIR = 'saved_jobs'

def save_claim_to_file(claim, batch_id, claim_id):
    claim_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    os.makedirs(claim_dir, exist_ok=True)
    claim_file = os.path.join(claim_dir, f"{claim_id}.txt")
    with open(claim_file, 'w') as f:
        json.dump({
            "text": claim.text,
            "status": "queued",
            "additional_info": ""
        }, f, indent=2)

def process_claim_in_background(claim, batch_id, claim_id):
    claim_processor = ClaimProcessor()
    claim_processor.process_claim(claim, batch_id, claim_id)

def verify_password(password):
    if current_app.config['REQUIRE_PASSWORD']:
        if not password:
            return False
        return password == current_app.config['ACCESS_PASSWORD']
    return True

@api.route('/api/v1/claims', methods=['POST'])
def upload_claim():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "Missing claim text"}), 400
    
    if not verify_password(data.get('password')):
        return jsonify({"error": "Invalid password"}), 403
    
    # Get search configuration
    search_config = data.get('searchConfig', {})
    num_queries = search_config.get('numQueries', 10)
    results_per_query = search_config.get('resultsPerQuery', 1)
    
    claim = Claim(text=data['text'], source=data.get('source', 'user'))
    claim.search_config = {
        'num_queries': num_queries,
        'results_per_query': results_per_query
    }
    
    batch_id = str(uuid.uuid4())[:8]
    claim_id = str(uuid.uuid4())[:8]
    
    save_claim_to_file(claim, batch_id, claim_id)
    
    threading.Thread(target=process_claim_in_background, args=(claim, batch_id, claim_id)).start()
    
    return jsonify({"claim_id": claim_id, "batch_id": batch_id, "status": "queued"}), 202

@api.route('/api/v1/claims/<claim_id>', methods=['GET'])
def get_claim_status(claim_id):
    for root, dirs, files in os.walk(SAVED_JOBS_DIR):
        if f"{claim_id}.txt" in files:
            with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
                claim_data = json.load(f)
            return jsonify({
                "claim_id": claim_id,
                "text": claim_data.get('text', ''),
                "status": claim_data.get('status', 'Unknown'),
                "additional_info": claim_data.get('additional_info', ''),
                "suggested_claim": claim_data.get('suggested_claim', '')
            }), 200
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/claims/<claim_id>/report', methods=['GET'])
def get_claim_report(claim_id):
    for root, dirs, files in os.walk(SAVED_JOBS_DIR):
        if f"{claim_id}.txt" in files:
            with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
                claim_data = json.load(f)
            return jsonify({
                "claim_id": claim_id,
                "text": claim_data.get('text', ''),
                "status": claim_data.get('status', ''),
                "report": json.loads(claim_data.get('additional_info', '{}'))
            }), 200
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/batch', methods=['POST'])
def start_batch_job():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    if not verify_password(request.form.get('password')):
        return jsonify({"error": "Invalid password"}), 403
    
    # Get search configuration
    search_config = {
        'num_queries': int(request.form.get('numQueries', 10)),
        'results_per_query': int(request.form.get('resultsPerQuery', 1))
    }
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.txt'):
        batch_id = str(uuid.uuid4())[:8]
        batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        file_path = os.path.join(batch_dir, 'claims.txt')
        file.save(file_path)
        
        with open(file_path, 'r') as f:
            claims = [line.strip() for line in f if line.strip()]
        
        # Create claims and set search config after initialization
        batch_claims = []
        for claim_text in claims:
            claim = Claim(text=claim_text)
            claim.search_config = search_config
            batch_claims.append(claim)
        
        batch_job = BatchJob(claims=batch_claims)
        
        threading.Thread(target=process_batch_in_background, args=(batch_job, batch_id)).start()
        
        return jsonify({"batch_id": batch_id, "status": "processing"}), 202

def process_batch_in_background(batch_job: BatchJob, batch_id: str):
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    progress_file = os.path.join(batch_dir, 'progress.json')
    
    # Initialize progress file
    with open(progress_file, 'w') as f:
        json.dump({
            "processed_claims": 0,
            "total_claims": len(batch_job.claims),
            "status": "processing",
            "current_claim_id": None
        }, f)

    for i, claim in enumerate(batch_job.claims):
        claim_id = str(uuid.uuid4())[:8]
        save_claim_to_file(claim, batch_id, claim_id)
        update_batch_progress(batch_id, i, len(batch_job.claims), current_claim_id=claim_id)
        claim_processor = ClaimProcessor()
        claim_processor.process_claim(claim, batch_id, claim_id)
        update_batch_progress(batch_id, i + 1, len(batch_job.claims))

    # Update status to completed when all claims are processed
    update_batch_progress(batch_id, len(batch_job.claims), len(batch_job.claims), status="completed", current_claim_id=None)

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
                            "additional_info": "Empty file"
                        })
                    else:
                        claim_data = json.loads(content)
                        claims.append({
                            "claim_id": file[:-4],
                            "text": claim_data.get('text', ''),
                            "status": claim_data.get('status', 'Unknown'),
                            "additional_info": claim_data.get('additional_info', '')
                        })
            except json.JSONDecodeError as e:
                claims.append({
                    "claim_id": file[:-4],
                    "text": "",
                    "status": "Error",
                    "additional_info": f"Invalid JSON in file: {str(e)}"
                })
            except Exception as e:
                claims.append({
                    "claim_id": file[:-4],
                    "text": "",
                    "status": "Error",
                    "additional_info": f"Error reading file: {str(e)}"
                })
    
    overall_status = "processed" if all(claim['status'] == 'processed' for claim in claims) else "processing"
    
    return jsonify({
        "batch_id": batch_id,
        "status": overall_status,
        "claims": claims
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
    return render_template('batch_results.html', batch_id=batch_id)

# Add these new routes

@api.route('/browser', methods=['GET'])
def browser():
    return render_template('browser.html')

@api.route('/api/v1/browse', methods=['GET'])
def browse_batches():
    search_term = request.args.get('search', '').lower()
    batches = []

    for batch_id in os.listdir(SAVED_JOBS_DIR):
        batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
        if os.path.isdir(batch_dir):
            claims = []
            for file in os.listdir(batch_dir):
                if file.endswith('.txt') and file != 'claims.txt':
                    with open(os.path.join(batch_dir, file), 'r') as f:
                        claim_data = json.load(f)
                        additional_info = claim_data.get('additional_info', '{}')
                        try:
                            additional_info_json = json.loads(additional_info)
                        except json.JSONDecodeError:
                            additional_info_json = {}

                        if search_term in claim_data.get('text', '').lower():
                            claims.append({
                                'text': claim_data.get('text', ''),
                                'status': claim_data.get('status', ''),
                                'rating': additional_info_json.get('claimRating', 'N/A')
                            })

            if claims or search_term in batch_id.lower():
                batches.append({
                    'batch_id': batch_id,
                    'total_claims': len(claims),
                    'preview_claims': claims[:3]
                })

    return jsonify({'batches': batches})

@api.route('/api/v1/delete/claim/<claim_id>', methods=['DELETE'])
def delete_claim(claim_id):
    for root, dirs, files in os.walk(SAVED_JOBS_DIR):
        if f"{claim_id}.txt" in files:
            os.remove(os.path.join(root, f"{claim_id}.txt"))
            return jsonify({"message": "Claim deleted successfully"}), 200
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/delete/batch/<batch_id>', methods=['DELETE'])
def delete_batch(batch_id):
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)
        return jsonify({"message": "Batch deleted successfully"}), 200
    return jsonify({"error": "Batch not found"}), 404

@api.route('/api/v1/claims/<claim_id>/download_citations', methods=['GET'])
def download_citations(claim_id):
    for root, dirs, files in os.walk(SAVED_JOBS_DIR):
        if f"{claim_id}.txt" in files:
            with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
                claim_data = json.load(f)
                report = json.loads(claim_data.get('additional_info', '{}'))
                citations = []

                for paper in report.get('supportingPapers', []):
                    for citation in paper.get('citations', []):
                        citations.append(citation['citation'])

                # Create a temporary file for the citations
                citation_file_path = os.path.join(SAVED_JOBS_DIR, f"{claim_id}_citations.ris")
                with open(citation_file_path, 'w') as citation_file:
                    citation_file.write("\n\n".join(citations))

                return send_file(citation_file_path, as_attachment=True, download_name=f"{claim_id}_citations.ris")

    return jsonify({"error": "Claim not found"}), 404

def update_enhance_progress(batch_id: str, progress: dict):
    progress_file = os.path.join(SAVED_JOBS_DIR, batch_id, 'enhance_progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

@api.route('/api/v1/enhance-batch', methods=['POST'])
def enhance_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    if not verify_password(request.form.get('password')):
        return jsonify({"error": "Invalid password"}), 403
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    batch_id = str(uuid.uuid4())[:8]
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    # Initialize progress file
    update_enhance_progress(batch_id, {
        "processed_claims": 0,
        "total_claims": 0,
        "status": "initializing"
    })
    
    # Read claims
    claims = [line.strip() for line in file.stream.read().decode('utf-8').splitlines() if line.strip()]
    
    # Start background task
    def process_in_background():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Process claims in batches
            openai_service = OpenAIService()
            batch_size = 100
            all_results = []
            
            for i in range(0, len(claims), batch_size):
                batch = claims[i:i + batch_size]
                batch_results = loop.run_until_complete(openai_service.process_claims_batch(batch))
                all_results.extend(batch_results)
                
                # Update progress
                update_enhance_progress(batch_id, {
                    "processed_claims": min(i + batch_size, len(claims)),
                    "total_claims": len(claims),
                    "status": "processing"
                })
            
            # Save final results
            results_file = os.path.join(batch_dir, 'enhanced_claims.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'claims': all_results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            # Update final progress
            update_enhance_progress(batch_id, {
                "processed_claims": len(claims),
                "total_claims": len(claims),
                "status": "completed"
            })
            
        except Exception as e:
            print(f"Error in background task: {str(e)}")
            update_enhance_progress(batch_id, {
                "processed_claims": 0,
                "total_claims": len(claims),
                "status": "error",
                "error": str(e)
            })
        finally:
            loop.close()
    
    # Start processing in background
    threading.Thread(target=process_in_background).start()
    
    return jsonify({
        "batch_id": batch_id,
        "message": "Processing started"
    }), 202

@api.route('/api/v1/enhance-batch/<batch_id>/progress', methods=['GET'])
def get_enhance_progress(batch_id):
    progress_file = os.path.join(SAVED_JOBS_DIR, batch_id, 'enhance_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Batch not found"}), 404

@api.route('/enhance-results', methods=['GET'])
def enhance_results():
    batch_id = request.args.get('batch_id')
    if not batch_id:
        return "No batch ID provided", 400
    
    results_file = os.path.join(SAVED_JOBS_DIR, batch_id, 'enhanced_claims.json')
    if not os.path.exists(results_file):
        return "Results not found", 404
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return render_template('enhance_results.html', batch_id=batch_id, results=results)

@api.route('/api/v1/enhanced-claims/<batch_id>/download', methods=['GET'])
def download_enhanced_claims(batch_id):
    results_file = os.path.join(SAVED_JOBS_DIR, batch_id, 'enhanced_claims.json')
    if not os.path.exists(results_file):
        return jsonify({"error": "Results not found"}), 404
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create text content with one enhanced claim per line
    output = "\n".join(claim["suggested"] for claim in results['claims'])
    
    return Response(
        output,
        mimetype="text/plain",
        headers={"Content-disposition": f"attachment; filename=enhanced_claims_{batch_id}.txt"}
    )

@api.route('/enhance-progress', methods=['GET'])
def enhance_progress():
    batch_id = request.args.get('batch_id')
    if not batch_id:
        return "No batch ID provided", 400
    return render_template('enhance_progress.html', batch_id=batch_id)
