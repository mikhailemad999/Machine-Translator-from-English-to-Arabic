"""
Evaluation Views — Model evaluation with metrics (Step 8).

GET  /api/evaluate/{job_id}/ — Evaluate a trained model
POST /api/evaluate/run/{job_id}/ — Trigger evaluation
"""
import os
import pandas as pd
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

from api.models import TrainingJob, EvaluationResult
from api.serializers import EvaluationResultSerializer


@api_view(['POST'])
def run_evaluation(request, job_id):
    """Run evaluation on a completed training job."""
    try:
        job = TrainingJob.objects.get(pk=job_id)
    except TrainingJob.DoesNotExist:
        return Response({'error': 'Training job not found'}, status=status.HTTP_404_NOT_FOUND)

    if job.status != 'completed':
        return Response(
            {'error': f'Training not complete. Current status: {job.status}'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Check if already evaluated
    if hasattr(job, 'evaluation'):
        serializer = EvaluationResultSerializer(job.evaluation)
        return Response({
            'evaluation': serializer.data,
            'message': 'Evaluation already exists.',
        })

    try:
        from ml.evaluator import evaluate_model

        # Load test set
        test_path = os.path.join(str(settings.MODEL_SAVE_DIR), 'test_set.csv')
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
        else:
            # Fallback: create test set from cleaned data
            cleaned_csv = os.path.join(str(settings.DATA_DIR), f'dataset_{job.dataset_id}_cleaned.csv')
            df = pd.read_csv(cleaned_csv)
            test_df = df.sample(frac=0.1, random_state=42)

        # Run evaluation
        results = evaluate_model(
            test_df=test_df,
            model_path=job.model_checkpoint_path,
            baseline_model_name=job.model_name,
            charts_dir=str(settings.CHARTS_DIR),
        )

        # Store results
        evaluation = EvaluationResult.objects.create(
            training_job=job,
            bleu_score=results['fine_tuned']['bleu'],
            chrf_score=results['fine_tuned']['chrf'],
            ter_score=results['fine_tuned']['ter'],
            baseline_bleu=results['baseline']['bleu'],
            baseline_chrf=results['baseline']['chrf'],
            baseline_ter=results['baseline']['ter'],
            example_translations=results['example_translations'],
            meets_target=results['meets_target'],
            target_notes=results['target_notes'],
        )

        serializer = EvaluationResultSerializer(evaluation)
        return Response({
            'evaluation': serializer.data,
            'improvement': results['improvement'],
            'comparison_chart': results.get('comparison_chart_path', ''),
            'meets_target': results['meets_target'],
            'target_notes': results['target_notes'],
            'message': 'Evaluation complete.',
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {'error': f'Evaluation failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_evaluation(request, job_id):
    """Get evaluation results for a training job."""
    try:
        job = TrainingJob.objects.get(pk=job_id)
    except TrainingJob.DoesNotExist:
        return Response({'error': 'Training job not found'}, status=status.HTTP_404_NOT_FOUND)

    try:
        evaluation = job.evaluation
    except EvaluationResult.DoesNotExist:
        return Response(
            {'error': 'No evaluation found. Run POST /api/evaluate/run/{job_id}/ first.'},
            status=status.HTTP_404_NOT_FOUND
        )

    serializer = EvaluationResultSerializer(evaluation)
    return Response(serializer.data)
