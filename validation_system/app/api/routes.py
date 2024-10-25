from flask import Blueprint, request, jsonify, send_file, render_template
import os
import uuid
import shutil
import threading
import json
from app.services.claim_processor import ClaimProcessor
from app.services.literature_searcher import LiteratureSearcher
from app.services.paper_analyzer import PaperAnalyzer
from app.services.evidence_scorer import EvidenceScorer
from app.models.claim import Claim
from app.models.batch_job import BatchJob

api = Blueprint('api', __name__)

claim_processor = ClaimProcessor()
literature_searcher = LiteratureSearcher()
paper_analyzer = PaperAnalyzer()
evidence_scorer = EvidenceScorer()

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
    claim_processor.process_claim(claim, batch_id, claim_id)

@api.route('/api/v1/claims', methods=['POST'])
def upload_claim():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "Missing claim text"}), 400
    
    claim = Claim(text=data['text'], source=data.get('source', 'user'))
    
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
                "status": claim_data['status'],
                "additional_info": claim_data.get('additional_info', ''),
                "suggested_claim": claim_data.get('suggested_claim', '')
            }), 200
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/claims/<claim_id>/report', methods=['GET'])
def get_claim_report(claim_id):
    # Retrieve the report from the filesystem
    for root, dirs, files in os.walk(SAVED_JOBS_DIR):
        if f"{claim_id}.txt" in files:
            with open(os.path.join(root, f"{claim_id}.txt"), 'r') as f:
                report = f.read()
            return jsonify({"claim_id": claim_id, "report": report}), 200
    return jsonify({"error": "Claim not found"}), 404

@api.route('/api/v1/batch', methods=['POST'])
def start_batch_job():
    data = request.json
    if 'claims' not in data or not isinstance(data['claims'], list):
        return jsonify({"error": "Invalid batch job format"}), 400
    
    batch_id = str(uuid.uuid4())[:8]
    batch_job = BatchJob(claims=[Claim(text=claim) for claim in data['claims']])
    processed_job = claim_processor.process_batch(batch_job)
    
    for claim in processed_job.claims:
        save_claim_to_file(claim, batch_id, str(uuid.uuid4())[:8])
    
    return jsonify({"job_id": batch_id, "status": "processing"}), 202

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
    return render_template('index.html')

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
        if file.endswith('.txt'):
            with open(os.path.join(batch_dir, file), 'r') as f:
                claim_data = json.load(f)
                claims.append({
                    "claim_id": file[:-4],
                    "status": claim_data['status'],
                    "additional_info": claim_data.get('additional_info', '')
                })
    
    overall_status = "processed" if all(claim['status'] == 'processed' for claim in claims) else "processing"
    
    return jsonify({
        "batch_id": batch_id,
        "status": overall_status,
        "claims": claims
    }), 200
