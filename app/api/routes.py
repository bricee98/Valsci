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
from app.services.openai_service import OpenAIService
from app.services.email_service import EmailService
import logging
import traceback

api = Blueprint('api', __name__)

SAVED_JOBS_DIR = 'saved_jobs'

logger = logging.getLogger(__name__)

def save_claim_to_file(claim, batch_id, claim_id):
    claim_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    os.makedirs(claim_dir, exist_ok=True)
    claim_file = os.path.join(claim_dir, f"{claim_id}.txt")
    
    # Ensure claim.text is a string, not a list
    claim_text = claim.text[0] if isinstance(claim.text, list) else claim.text
    
    with open(claim_file, 'w') as f:
        json.dump({
            "text": claim_text,  # Store the string directly
            "status": "queued",
            "additional_info": ""
        }, f, indent=2)

def process_claim_in_background(claim, batch_id, claim_id):
    try:
        claim_processor = ClaimProcessor()
        claim_processor.process_claim(claim, batch_id, claim_id)
    except Exception as e:
        logger.error(f"Error processing claim: {str(e)}")
        logger.error(traceback.format_exc())

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
    num_queries = search_config.get('numQueries', 5)
    results_per_query = search_config.get('resultsPerQuery', 5)
    
    # Create claim object
    claim_text = data['text'][0] if isinstance(data['text'], list) else data['text']
    claim = Claim(text=claim_text, source=data.get('source', 'user'))
    claim.search_config = {
        'num_queries': num_queries,
        'results_per_query': results_per_query
    }
    
    batch_id = str(uuid.uuid4())[:8]
    claim_id = str(uuid.uuid4())[:8]
    
    save_claim_to_file(claim, batch_id, claim_id)
    
    # Start processing in background
    threading.Thread(
        target=process_claim_in_background, 
        args=(claim, batch_id, claim_id)
    ).start()
    
    return jsonify({
        "claim_id": claim_id, 
        "batch_id": batch_id, 
        "status": "queued"
    }), 202

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
                "additional_info": claim_data.get('additional_info', {}),
                "review_type": claim_data.get('review_type', 'regular')
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
        batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save claims file
        file_path = os.path.join(batch_dir, 'claims.txt')
        file.save(file_path)
        
        # Read claims
        with open(file_path, 'r', encoding='utf-8') as f:
            claims = [line.strip() for line in f if line.strip()]
        
        # Create claims with search config
        batch_claims = []
        for claim_text in claims:
            claim = Claim(text=claim_text)
            claim.search_config = {
                'num_queries': num_queries,
                'results_per_query': results_per_query
            }
            batch_claims.append(claim)
        
        batch_job = BatchJob(claims=batch_claims)
        
        # Save notification settings
        if notification_email:
            notification_file = os.path.join(batch_dir, 'notification.json')
            with open(notification_file, 'w') as f:
                json.dump({
                    'email': notification_email,
                    'num_claims': len(claims)
                }, f)
        
        # Send start notification if email provided
        if notification_email:
            EmailService.send_batch_start_notification(
                notification_email,
                batch_id,
                len(claims),
                'regular'
            )
        
        # Start processing
        app = current_app._get_current_object()
        thread = threading.Thread(
            target=process_batch_in_background,
            args=(app, batch_job, batch_id)
        )
        thread.start()
        
        return jsonify({
            "batch_id": batch_id,
            "status": "processing"
        }), 202

def process_batch_in_background(app, batch_job: BatchJob, batch_id: str):
    """Process a batch of claims in the background thread."""
    batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
    progress_file = os.path.join(batch_dir, 'progress.json')
    notification_file = os.path.join(batch_dir, 'notification.json')
    
    # Use app context in the background thread
    with app.app_context():
        try:
            # Initialize progress file
            with open(progress_file, 'w') as f:
                json.dump({
                    "processed_claims": 0,
                    "total_claims": len(batch_job.claims),
                    "status": "processing",
                    "current_claim_id": None
                }, f)

            # Set up and run the asyncio event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    run_batch_processing(loop, batch_job, batch_id)
                )
            finally:
                loop.close()

            # Update status to completed when all claims are processed
            update_batch_progress(
                batch_id, 
                len(batch_job.claims), 
                len(batch_job.claims), 
                status="completed"
            )
            
            # After processing is complete, send completion email notification if configured
            if os.path.exists(notification_file):
                with open(notification_file, 'r') as f:
                    notification_data = json.load(f)
                    
                    # Send completion email notification
                    EmailService.send_batch_completion_notification(
                        notification_data['email'],
                        batch_id,
                        notification_data['num_claims'],
                        notification_data['review_type']
                    )

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(traceback.format_exc())
            update_batch_progress(
                batch_id,
                len(batch_job.claims),
                len(batch_job.claims),
                status="error"
            )
            # Send error notification email
            if os.path.exists(notification_file):
                with open(notification_file, 'r') as f:
                    notification_data = json.load(f)
                    EmailService.send_batch_error_notification(
                        notification_data['email'],
                        batch_id,
                        str(e)
                    )

async def run_batch_processing(loop, batch_job: BatchJob, batch_id: str):
    """Orchestrate the async processing of claims in the batch."""
    # Initialize structures for the two-phase processing
    search_queue = asyncio.Queue()
    results_dict = {}
    
    # Start the search worker
    search_worker_task = asyncio.create_task(
        search_worker(search_queue, results_dict)
    )
    
    # Enqueue all claims for searching
    for i, claim in enumerate(batch_job.claims):
        claim_id = str(uuid.uuid4())[:8]
        await search_queue.put((claim_id, claim))
        
    # Signal search worker to terminate after processing all claims
    await search_queue.put(None)
    
    # Wait for all searches to complete
    await search_queue.join()
    await search_worker_task
    
    # Process claims post-search with limited concurrency
    sem = asyncio.Semaphore(5)  # limit to 5 concurrent post-search tasks
    
    post_search_tasks = []
    for claim_id, papers in results_dict.items():
        task = asyncio.create_task(
            process_claim_post_search(
                claim_id=claim_id,
                papers=papers,
                batch_id=batch_id,
                sem=sem
            )
        )
        post_search_tasks.append(task)
    
    # Wait for all post-search processing to complete
    await asyncio.gather(*post_search_tasks)

async def search_worker(queue: asyncio.Queue, results_dict: dict):
    """Worker that processes search requests at a rate-limited pace."""
    while True:
        # Get the next item from the queue
        item = await queue.get()
        if item is None:  # Check for termination signal
            queue.task_done()
            break
            
        claim_id, claim = item
        
        try:
            # Rate-limit: wait 1 second between searches
            await asyncio.sleep(1)
            
            # Perform the search (we'll need to modify literature_searcher to be async)
            literature_searcher = LiteratureSearcher()
            papers = await literature_searcher.search_papers(claim)
            
            # Store the results
            results_dict[claim_id] = papers
            
        except Exception as e:
            logger.error(f"Error searching papers for claim {claim_id}: {str(e)}")
            results_dict[claim_id] = []  # Store empty list on error
            
        finally:
            queue.task_done()

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
    search_term = request.args.get('search', '').lower()
    batches = []

    for batch_id in os.listdir(SAVED_JOBS_DIR):
        batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
        if os.path.isdir(batch_dir):
            claims = []
            latest_timestamp = None  # Track the latest timestamp in the batch
            
            for file in os.listdir(batch_dir):
                if file.endswith('.txt') and file != 'claims.txt':
                    with open(os.path.join(batch_dir, file), 'r') as f:
                        claim_data = json.load(f)
                        additional_info = claim_data.get('additional_info', {})
                        
                        # Handle additional_info whether it's a string or dict
                        if isinstance(additional_info, str):
                            try:
                                additional_info_json = json.loads(additional_info)
                            except json.JSONDecodeError:
                                additional_info_json = {}
                        else:
                            additional_info_json = additional_info

                        # Track the latest timestamp
                        timestamp = claim_data.get('timestamp')
                        if timestamp:
                            if not latest_timestamp or timestamp > latest_timestamp:
                                latest_timestamp = timestamp

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
                    'preview_claims': claims[:3],
                    'timestamp': latest_timestamp or '1970-01-01T00:00:00'  # Use Unix epoch as fallback
                })

    # Sort batches by timestamp in descending order
    batches.sort(key=lambda x: x['timestamp'], reverse=True)
    
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
                batch_results = loop.run_until_complete(openai_service.enhance_claims_batch(batch))
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

@api.route('/llm-screen-results', methods=['GET'])
def llm_screen_results():
    claim_id = request.args.get('claim_id')
    if not claim_id:
        return "No claim ID provided", 400
    return render_template('llm_screen_results.html', claim_id=claim_id)

@api.route('/llm-screen-batch-results', methods=['GET'])
def llm_screen_batch_results():
    batch_id = request.args.get('batch_id')
    if not batch_id:
        return "No batch ID provided", 400
    return render_template('llm_screen_batch_results.html', batch_id=batch_id)

@api.route('/api/v1/enhance-claim', methods=['POST'])
def enhance_claim():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing claim text"}), 400

    try:
        openai_service = OpenAIService()
        # Use the same system prompt as in batch enhancement
        system_prompt = """
        You are an AI assistant tasked with evaluating scientific claims and optimizing them for search. 
        Respond with a JSON object containing 'is_valid' (boolean), 'explanation' (string), and 'suggested' (string).
        The 'is_valid' field should be true if the input is a proper scientific claim, and false otherwise. 
        The 'explanation' field should provide a brief reason for your decision.
        The 'suggested' field should always contain an optimized version of the claim, 
        even if the original claim is invalid. For invalid claims, provide a corrected or improved version.

        Examples of valid scientific claims (note: these may or may not be true, but they are properly formed claims):
         1. "Increased consumption of processed foods is linked to higher rates of obesity in urban populations."
         2. "The presence of certain gut bacteria can influence mood and cognitive function in humans."
         3. "Exposure to blue light from electronic devices before bedtime does not disrupt the circadian rhythm."
         4. "Regular meditation practice can lead to structural changes in the brain's gray matter."
         5. "Higher levels of atmospheric CO2 have no effect on global average temperatures."
         6. "Calcium channels are affected by AMP."
         7. "People who drink soda are much healthier than those who don't."

        Examples of non-claims (these are not valid scientific claims):
         1. "The sky is beautiful." (This is an opinion, not a testable claim)
         2. "What is the effect of exercise on heart health?" (This is a question, not a claim)
         3. "Scientists should study climate change more." (This is a recommendation, not a claim)
         4. "Drink more water!" (This is a command, not a claim),
         5. "Investigating the cognitive effects of BRCA2 mutations on intelligence quotient (IQ) levels." (This doesn't make a claim about anything)

        Reject claims that include ambiguous abbreviations or shorthand, unless it's clear to you what they mean. Remember, a valid scientific claim should be a specific, testable assertion about a phenomenon or relationship between variables. It doesn't have to be true, but it should be a testable assertion.

        For the 'suggested' field, focus on using clear, concise language with relevant scientific terms that would be 
        likely to appear in academic papers. Avoid colloquialisms and ensure the suggested version maintains the 
        original meaning (even if you think it's not true) while being more search-friendly.
        """
        
        result = openai_service.generate_json(
            prompt=f"Evaluate and optimize the following claim for search:\n\n{data['text']}",
            system_prompt=system_prompt
        )
        
        return jsonify({
            "original": data['text'],
            "suggested": result.get('suggested', data['text']),
            "is_valid": result.get('is_valid', True),
            "explanation": result.get('explanation', '')
        })

    except Exception as e:
        logger.error(f"Error enhancing claim: {str(e)}")
        return jsonify({"error": str(e)}), 500

async def process_claim_post_search(claim_id: str, papers: List[Paper], batch_id: str, sem: asyncio.Semaphore):
    """Process a claim after papers have been retrieved."""
    async with sem:  # Limit concurrent processing
        try:
            # Initialize claim processor
            claim_processor = ClaimProcessor()
            
            # Save initial claim status
            save_claim_to_file(
                Claim(text=papers[0].text if papers else ""),  # We need the original claim text
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
                claim_id=claim_id
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
