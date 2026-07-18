"""
ML Worker — GPU-enabled training and inference service.

Exposes a Flask API for the Django backend to dispatch training jobs
and inference requests to the GPU container.
"""
import os
import sys
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Add ml module to path (both local Windows parent folder and Docker /app fallback)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
sys.path.insert(0, '/app')

app = Flask(__name__)
CORS(app)

# Track active jobs
active_jobs = {}


@app.route('/', methods=['GET'])
def index():
    """ML Worker root endpoint."""
    import torch
    return jsonify({
        'status': 'running',
        'service': 'ML Worker',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'endpoints': {
            'root': '/',
            'health': '/health',
            'train': '/train',
            'translate': '/translate',
            'translate_batch': '/translate/batch'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    import torch
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
        'active_jobs': len(active_jobs),
    })


@app.route('/train', methods=['POST'])
def start_training():
    """Start a training job on GPU."""
    from ml.trainer import train_model
    import pandas as pd

    data = request.json
    dataset_path = data.get('dataset_path')
    job_id = data.get('job_id', 'default')

    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': f'Dataset not found: {dataset_path}'}), 400

    if job_id in active_jobs:
        return jsonify({'error': f'Job {job_id} already running'}), 409

    def run_training():
        try:
            df = pd.read_csv(dataset_path)
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
            active_jobs[job_id] = {'status': 'completed', 'results': results}
        except Exception as e:
            active_jobs[job_id] = {'status': 'failed', 'error': str(e)}

    active_jobs[job_id] = {'status': 'running'}
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'started'}), 202


@app.route('/train/status/<job_id>', methods=['GET'])
def training_status(job_id):
    """Check training job status."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(active_jobs[job_id])


@app.route('/translate', methods=['POST'])
def translate():
    """Translate text using the GPU model."""
    from ml.inference import translate as do_translate

    data = request.json
    text = data.get('text', '')
    model_path = data.get('model_path')

    if not text:
        return jsonify({'error': 'text is required'}), 400

    result = do_translate(text=text, model_path=model_path)
    return jsonify(result)


@app.route('/translate/batch', methods=['POST'])
def translate_batch():
    """Batch translate texts using the GPU model."""
    from ml.inference import translate_batch as do_batch

    data = request.json
    texts = data.get('texts', [])
    model_path = data.get('model_path')

    if not texts:
        return jsonify({'error': 'texts array is required'}), 400

    translations = do_batch(texts=texts, model_path=model_path)
    return jsonify({'translations': translations})


if __name__ == '__main__':
    port = int(os.getenv('ML_WORKER_PORT', 8001))
    print(f"ML Worker starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
