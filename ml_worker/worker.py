"""
==============================================================================
ML WORKER SERVICE — GPU-Accelerated Machine Translation Microservice
==============================================================================

Purpose:
  This module acts as an isolated microservice (Flask API) that manages:
  1. GPU health checks and hardware status reporting (PyTorch CUDA support).
  2. Asynchronous background model training on GPU (`/train` endpoint).
  3. Real-time English-to-Arabic text translation inference (`/translate`).
  4. High-throughput batch translation inference (`/translate/batch`).

Architecture:
  - Communicates directly with the Django Backend (and React Frontend).
  - Offloads long-running training jobs to background daemon threads to keep the web server non-blocking.
  - Dynamically imports shared ML pipelines from `backend.ml`.
==============================================================================
"""

import os
import sys
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# ----------------------------------------------------------------------------
# 1. Environment & Path Configurations
# ----------------------------------------------------------------------------
# Load environment variables from .env file into os.environ
load_dotenv(find_dotenv())

# Add the Django backend ML directory to system path so worker.py can import modules from backend/ml.
# - First path: Resolves relative path to 'backend' directory on local dev machine.
# - Second path: '/app' is the root working directory inside the Docker container.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
sys.path.insert(0, '/app')

# ----------------------------------------------------------------------------
# 2. Flask Application Setup & Cross-Origin Resource Sharing (CORS)
# ----------------------------------------------------------------------------
# Initialize the Flask Web Application instance
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from Frontend (e.g., React on port 3000)
CORS(app)

# Global in-memory data structure to store active & completed background training jobs
# Format: { 'job_id_string': { 'status': 'running' | 'completed' | 'failed', 'results': {...}, 'error': '...' } }
active_jobs = {}


# ----------------------------------------------------------------------------
# 3. API Endpoints
# ----------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    """
    Root Endpoint - Basic Service Information & API Directory.
    
    Returns:
        JSON response with service status, GPU hardware detection, and list of available endpoints.
    """
    import torch  # Lazy import PyTorch to check CUDA availability on demand
    
    # Query PyTorch for CUDA GPU hardware acceleration
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else 'N/A (CPU Mode)'
    
    return jsonify({
        'status': 'running',
        'service': 'ML Worker (GPU-Accelerated Translation Engine)',
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'endpoints': {
            'root': 'GET / - Service overview and GPU detection',
            'health': 'GET /health - Deep system health check & VRAM statistics',
            'train': 'POST /train - Trigger asynchronous background model training',
            'train_status': 'GET /train/status/<job_id> - Poll training job progress',
            'translate': 'POST /translate - Translate a single string (English -> Arabic)',
            'translate_batch': 'POST /translate/batch - Translate a list of strings in batch'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health Check Endpoint - Detailed System Diagnostics.
    
    Used by load balancers, Docker healthchecks, and frontend status widgets to verify:
    - Service responsiveness
    - GPU device detection and total available VRAM (in Gigabytes)
    - Total count of active background jobs currently executing
    
    Returns:
        JSON object containing health metrics and HTTP status 200.
    """
    import torch
    
    gpu_available = torch.cuda.is_available()
    
    # Calculate GPU VRAM memory size in GB (Gigabytes) if CUDA is available
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if gpu_available else 0
    
    return jsonify({
        'status': 'healthy',
        'gpu_available': gpu_available,
        'gpu_name': torch.cuda.get_device_name(0) if gpu_available else 'N/A',
        'gpu_memory_gb': vram_gb,
        'active_jobs': len(active_jobs),  # Number of tracked training tasks
    })


@app.route('/train', methods=['POST'])
def start_training():
    """
    Start Training Endpoint - Launch Asynchronous Fine-Tuning Job.
    
    Accepts JSON body with training configuration:
    - `dataset_path`: (Required) Path to preprocessed CSV dataset file containing 'en' and 'ar' pairs.
    - `job_id`: Unique identifier for tracking job status (defaults to 'default').
    - Optional hyperparameter overrides: `model_name`, `batch_size`, `learning_rate`, `max_epochs`, `fp16`, etc.
    
    Flow:
    1. Validates input parameters and dataset file existence.
    2. Registers job ID in global `active_jobs` dict with status 'running'.
    3. Spawns a non-blocking background thread `threading.Thread` to execute `ml.trainer.train_model`.
    4. Immediately returns HTTP 202 (Accepted) with job details so caller is not blocked.
    """
    from ml.trainer import train_model
    import pandas as pd

    data = request.json or {}
    dataset_path = data.get('dataset_path')
    job_id = data.get('job_id', 'default')

    # Step 1: Validate that dataset file path is supplied and actually exists on disk
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': f'Dataset path missing or file not found on disk: {dataset_path}'}), 400

    # Step 2: Prevent starting duplicate jobs with the same job_id simultaneously
    if job_id in active_jobs and active_jobs[job_id].get('status') == 'running':
        return jsonify({'error': f'Job ID "{job_id}" is already actively running.'}), 409

    def run_training():
        """
        Background Worker Function.
        Executes inside a separate thread to run the complete training & evaluation loop.
        """
        try:
            # Read dataset into pandas DataFrame
            df = pd.read_csv(dataset_path)
            
            # Execute model fine-tuning routine (HuggingFace Seq2Seq / PyTorch)
            results = train_model(
                df=df,
                model_name=data.get('model_name', os.getenv('MODEL_NAME', 'Helsinki-NLP/opus-mt-en-ar')),
                save_dir=data.get('save_dir', os.getenv('MODEL_SAVE_DIR', '/app/models')),
                charts_dir=data.get('charts_dir', os.getenv('CHARTS_DIR', '/app/charts')),
                batch_size=data.get('batch_size', int(os.getenv('BATCH_SIZE', 4))),
                gradient_accumulation_steps=data.get('gradient_accumulation', int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 8))),
                learning_rate=data.get('learning_rate', float(os.getenv('LEARNING_RATE', 5e-5))),
                max_epochs=data.get('max_epochs', int(os.getenv('MAX_EPOCHS', 10))),
                fp16=data.get('fp16', os.getenv('FP16', 'True').lower() == 'true'),
                weight_decay=data.get('weight_decay', 0.01),
                early_stopping_patience=data.get('early_stopping_patience', int(os.getenv('EARLY_STOPPING_PATIENCE', 3))),
            )
            # Update job registry upon successful completion
            active_jobs[job_id] = {'status': 'completed', 'results': results}
        except Exception as e:
            # Capture any runtime exception or out-of-memory error during training
            active_jobs[job_id] = {'status': 'failed', 'error': str(e)}

    # Mark status as running in the global job tracker
    active_jobs[job_id] = {'status': 'running'}
    
    # Spawn background daemon thread (daemon=True ensures thread terminates if main server stops)
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'started', 'message': 'Training job launched in background thread.'}), 202


@app.route('/train/status/<job_id>', methods=['GET'])
def training_status(job_id):
    """
    Job Status Polling Endpoint.
    
    Parameters:
        job_id (str): Unique identifier passed during /train request.
        
    Returns:
        JSON response with job state ('running', 'completed', or 'failed') and output metrics or error details.
    """
    # Check if job ID exists in our active tracking dictionary
    if job_id not in active_jobs:
        return jsonify({'error': f'Job with ID "{job_id}" was not found.'}), 404
        
    return jsonify(active_jobs[job_id])


@app.route('/translate', methods=['POST'])
def translate():
    """
    Single Translation Endpoint - English to Arabic.
    
    JSON Payload:
    - `text`: (Required) String in English to translate.
    - `model_path`: (Optional) Custom checkpoint directory path to override default model.
    
    Returns:
        JSON response containing translated text, execution time, and model metadata.
    """
    # Import inference logic on demand to reduce start-up memory usage
    from ml.inference import translate as do_translate

    data = request.json or {}
    text = data.get('text', '').strip()
    model_path = data.get('model_path')

    # Validate that text field is not empty
    if not text:
        return jsonify({'error': 'Field "text" is required and cannot be empty.'}), 400

    # Call the translation inference function
    result = do_translate(text=text, model_path=model_path)
    return jsonify(result)


@app.route('/translate/batch', methods=['POST'])
def translate_batch():
    """
    Batch Translation Endpoint - Process Multiple English Texts Simultaneously.
    
    JSON Payload:
    - `texts`: (Required) List of English strings.
    - `model_path`: (Optional) Custom checkpoint directory path.
    
    Returns:
        JSON object with array of translation results for each sentence.
    """
    from ml.inference import translate_batch as do_batch

    data = request.json or {}
    texts = data.get('texts', [])
    model_path = data.get('model_path')

    # Ensure texts is a non-empty list
    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'Field "texts" must be a non-empty array of strings.'}), 400

    # Perform batch inference leveraging GPU parallelism
    translations = do_batch(texts=texts, model_path=model_path)
    return jsonify({'translations': translations})


# ----------------------------------------------------------------------------
# 4. Server Execution Entry Point
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Retrieve port configuration from environment variable (default: 8001)
    port = int(os.getenv('ML_WORKER_PORT', 8001))
    print(f"=========================================================")
    print(f" ML Worker Service starting on port {port}...")
    print(f"=========================================================")
    
    # Run Flask HTTP server binding to all network interfaces (0.0.0.0)
    app.run(host='0.0.0.0', port=port, debug=False)

