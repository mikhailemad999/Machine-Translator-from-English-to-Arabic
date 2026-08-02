"""
Training Views — Start and monitor model training (Step 7).

POST /api/train/start — Start a training job
GET  /api/train/status/{job_id}/ — Get training status and progress
GET  /api/train/list/ — List all training jobs
"""
import os
import threading
import pandas as pd
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from django.utils import timezone

from api.models import Dataset, PreprocessingRun, TrainingJob
from api.serializers import TrainingJobSerializer, TrainingStartSerializer
from mongodb_client import MongoDBClient


# Track active training threads
_active_trainings = {}


def _run_training(job_id):
    """Background training function."""
    from ml.trainer import train_model

    try:
        job = TrainingJob.objects.get(pk=job_id)
        job.status = 'running'
        job.save()

        # Load dataset (cleaned or raw fallback)
        cleaned_csv = os.path.join(str(settings.DATA_DIR), f'dataset_{job.dataset_id}_cleaned.csv')
        if os.path.exists(cleaned_csv):
            df = pd.read_csv(cleaned_csv)
        else:
            dataset = job.dataset
            cleaned_collection = f'{dataset.mongo_collection}_cleaned'
            records = MongoDBClient.get_raw_dataset(cleaned_collection)
            if not records:
                records = MongoDBClient.get_raw_dataset(dataset.mongo_collection)
            if records:
                df = pd.DataFrame(records)
            else:
                raw_csv = os.path.join(str(settings.DATA_DIR), 'en_ar_dataset.csv')
                if os.path.exists(raw_csv):
                    df = pd.read_csv(raw_csv)
                else:
                    raise ValueError(f"No dataset records found for dataset ID {job.dataset_id}")

        def progress_callback(epoch_info):
            """Update job with epoch progress."""
            job.refresh_from_db()
            epoch_data = job.epoch_data or []
            epoch_data.append(epoch_info)
            job.epoch_data = epoch_data
            job.save()

            # Log to MongoDB
            MongoDBClient.log_experiment(job_id, {
                'event': 'epoch_completed',
                **epoch_info,
            })

        # Run training
        results = train_model(
            df=df,
            model_name=job.model_name,
            save_dir=str(settings.MODEL_SAVE_DIR),
            charts_dir=str(settings.CHARTS_DIR),
            batch_size=job.batch_size,
            gradient_accumulation_steps=job.gradient_accumulation,
            learning_rate=job.learning_rate,
            max_epochs=job.max_epochs,
            fp16=job.fp16,
            weight_decay=job.weight_decay,
            early_stopping_patience=job.early_stopping_patience,
            progress_callback=progress_callback,
        )

        # Update job with results
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.best_epoch = results['best_epoch']
        job.best_val_loss = results['best_val_loss']
        job.best_val_bleu = results['best_val_bleu']
        job.learning_curve_path = results.get('learning_curve_path', '')
        job.model_checkpoint_path = results.get('model_checkpoint_path', '')
        job.train_size = results['train_size']
        job.val_size = results['val_size']
        job.test_size = results['test_size']
        job.epoch_data = results['epoch_data']
        job.diagnosis = results['diagnosis']['status']
        job.diagnosis_notes = results['diagnosis']['notes']
        job.save()

        # Update dataset status
        dataset = job.dataset
        dataset.status = 'completed'
        dataset.save()

    except Exception as e:
        try:
            job = TrainingJob.objects.get(pk=job_id)
            job.status = 'failed'
            job.diagnosis_notes = f'Training failed: {str(e)}'
            job.save()

            # Reset dataset status to 'ready' so it's not locked in 'training' status forever
            dataset = job.dataset
            dataset.status = 'ready'
            dataset.save()
        except Exception:
            pass

    finally:
        _active_trainings.pop(job_id, None)


@api_view(['POST'])
def start_training(request):
    """Start a new training job."""
    serializer = TrainingStartSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    dataset_id = data['dataset_id']

    try:
        dataset = Dataset.objects.get(pk=dataset_id)
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

    if dataset.status in ('training', 'preprocessing'):
        return Response(
            {'error': f'Dataset is currently in status "{dataset.status}". Please wait for ongoing operations to complete.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Create training job
    job = TrainingJob.objects.create(
        dataset=dataset,
        preprocessing_run_id=data.get('preprocessing_run_id'),
        model_name=settings.MODEL_NAME,
        batch_size=data.get('batch_size', 4),
        gradient_accumulation=data.get('gradient_accumulation', 8),
        learning_rate=data.get('learning_rate', 5e-5),
        max_epochs=data.get('max_epochs', 10),
        fp16=data.get('fp16', True),
        weight_decay=data.get('weight_decay', 0.01),
        early_stopping_patience=data.get('early_stopping_patience', 3),
        status='queued',
    )

    # Start training in background thread
    thread = threading.Thread(target=_run_training, args=(job.id,), daemon=True)
    thread.start()
    _active_trainings[job.id] = thread

    dataset.status = 'training'
    dataset.save()

    return Response({
        'job_id': job.id,
        'status': 'queued',
        'message': 'Training job started. Poll /api/train/status/{job_id}/ for progress.',
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def get_training_status(request, job_id):
    """Get training job status and progress."""
    try:
        job = TrainingJob.objects.get(pk=job_id)
    except TrainingJob.DoesNotExist:
        return Response({'error': 'Training job not found'}, status=status.HTTP_404_NOT_FOUND)

    serializer = TrainingJobSerializer(job)
    data = serializer.data

    # Add live info
    data['is_running'] = job_id in _active_trainings

    return Response(data)


@api_view(['GET'])
def list_training_jobs(request):
    """List all training jobs."""
    dataset_id = request.query_params.get('dataset_id')
    jobs = TrainingJob.objects.all()
    if dataset_id:
        jobs = jobs.filter(dataset_id=dataset_id)

    serializer = TrainingJobSerializer(jobs, many=True)
    return Response(serializer.data)
