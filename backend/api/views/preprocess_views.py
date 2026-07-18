"""
Preprocessing Views — Run the data preprocessing pipeline (Steps 2-6).

POST /api/preprocess/run — Run full preprocessing pipeline
GET  /api/preprocess/{id}/ — Get preprocessing run results
GET  /api/preprocess/list/{dataset_id}/ — List runs for a dataset
"""
import os
import pandas as pd
from datetime import datetime
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from django.utils import timezone

from api.models import Dataset, PreprocessingRun
from api.serializers import PreprocessingRunSerializer
from mongodb_client import MongoDBClient
from ml.data_loader import explore_dataset
from ml.duplicates import handle_duplicates
from ml.missing_values import handle_missing_values
from ml.outliers import handle_outliers
from ml.visualizations import generate_all_charts
from ml.imbalance import handle_imbalance


@api_view(['POST'])
def run_preprocessing(request):
    """
    Run the full preprocessing pipeline (Steps 1-6).

    Expects: { "dataset_id": <int> }
    """
    dataset_id = request.data.get('dataset_id')
    if not dataset_id:
        return Response({'error': 'dataset_id is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        dataset = Dataset.objects.get(pk=dataset_id)
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

    # Load raw data from MongoDB
    raw_records = MongoDBClient.get_raw_dataset(dataset.mongo_collection)
    if not raw_records:
        return Response({'error': 'No raw data found in MongoDB'}, status=status.HTTP_404_NOT_FOUND)

    df = pd.DataFrame(raw_records)

    # Create preprocessing run record
    run = PreprocessingRun.objects.create(
        dataset=dataset,
        status='running',
    )

    try:
        # Update dataset status
        dataset.status = 'preprocessing'
        dataset.save()

        # ============================================================
        # STEP 1: Explore (already done on upload, re-run for report)
        # ============================================================
        exploration_report = explore_dataset(df)
        run.original_shape = exploration_report['shape']
        run.dtypes_info = exploration_report['dtypes']
        run.sample_pairs = exploration_report['sample_pairs'][:10]
        run.encoding_notes = exploration_report.get('encoding_notes', '')

        # ============================================================
        # STEP 2: Handle Duplicates
        # ============================================================
        df, dup_report = handle_duplicates(df)
        run.duplicate_count_full = dup_report['duplicates_full_pair']['count']
        run.duplicate_count_en = dup_report['duplicates_en_only']['count']
        run.duplicate_count_ar = dup_report['duplicates_ar_only']['count']
        run.duplicate_pct = dup_report['duplicates_full_pair']['percentage']
        run.pairs_after_dedup = len(df)

        # ============================================================
        # STEP 3: Handle Missing Values
        # ============================================================
        df, missing_report = handle_missing_values(df)
        run.missing_en_count = missing_report['summary']['en_missing']
        run.missing_ar_count = missing_report['summary']['ar_missing']
        run.missing_en_pct = missing_report['summary']['en_missing_pct']
        run.missing_ar_pct = missing_report['summary']['ar_missing_pct']
        run.pairs_after_missing = len(df)

        # ============================================================
        # STEP 4: Handle Outliers
        # ============================================================
        df, outlier_report = handle_outliers(df)
        run.outlier_count_zscore = outlier_report['outliers']['zscore']['count']
        run.outlier_count_iqr = outlier_report['outliers']['iqr']['count']
        run.outlier_pct = outlier_report.get('outlier_pct', 0)
        run.pairs_after_outliers = len(df)
        run.length_ratio_min = outlier_report['length_stats'].get('length_ratio', {}).get('min', 0)
        run.length_ratio_max = outlier_report['length_stats'].get('length_ratio', {}).get('max', 0)

        # ============================================================
        # STEP 5: Generate Visualizations
        # ============================================================
        charts_dir = str(settings.CHARTS_DIR)
        chart_paths = generate_all_charts(df, charts_dir)
        run.chart_paths = chart_paths

        # ============================================================
        # STEP 6: Check Imbalance
        # ============================================================
        df_balanced, imbalance_report = handle_imbalance(df, strategy='none')
        run.imbalance_applicable = imbalance_report['is_applicable']
        run.class_distribution_before = imbalance_report.get('distribution_before', {})
        run.class_distribution_after = imbalance_report.get('distribution_after', {})

        # ============================================================
        # FINAL: Save cleaned data
        # ============================================================
        run.final_pairs = len(df)
        run.status = 'completed'
        run.completed_at = timezone.now()
        run.save()

        # Update dataset
        dataset.cleaned_pairs = len(df)
        dataset.status = 'ready'
        dataset.save()

        # Store cleaned data in MongoDB (for ML worker access)
        cleaned_collection = f'{dataset.mongo_collection}_cleaned'
        cleaned_records = df[['en', 'ar']].to_dict(orient='records')
        MongoDBClient.store_cleaned_pairs(cleaned_collection, cleaned_records)

        # Also save as CSV for training
        cleaned_csv_path = os.path.join(str(settings.DATA_DIR), f'dataset_{dataset.id}_cleaned.csv')
        df[['en', 'ar']].to_csv(cleaned_csv_path, index=False)

        # Store full reports in MongoDB for EDA dashboard
        MongoDBClient.store_eda_report(dataset.id, {
            'type': 'preprocessing_complete',
            'preprocessing_run_id': run.id,
            'step_reports': {
                'step_1_exploration': exploration_report,
                'step_2_duplicates': dup_report,
                'step_3_missing_values': missing_report,
                'step_4_outliers': outlier_report,
                'step_5_charts': chart_paths,
                'step_6_imbalance': imbalance_report,
            },
        })

        serializer = PreprocessingRunSerializer(run)
        return Response({
            'preprocessing_run': serializer.data,
            'step_reports': {
                'step_1': exploration_report,
                'step_2': dup_report,
                'step_3': missing_report,
                'step_4': {k: v for k, v in outlier_report.items() if k != 'outlier_samples'},
                'step_5': chart_paths,
                'step_6': imbalance_report,
            },
            'final_pairs': len(df),
            'cleaned_csv_path': cleaned_csv_path,
            'message': 'Preprocessing complete (Steps 1-6). Dataset is ready for training.',
        })

    except Exception as e:
        run.status = 'failed'
        run.save()
        dataset.status = 'uploaded'
        dataset.save()
        return Response(
            {'error': f'Preprocessing failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_preprocessing_run(request, pk):
    """Get a specific preprocessing run's details."""
    try:
        run = PreprocessingRun.objects.get(pk=pk)
    except PreprocessingRun.DoesNotExist:
        return Response({'error': 'Preprocessing run not found'}, status=status.HTTP_404_NOT_FOUND)

    serializer = PreprocessingRunSerializer(run)
    return Response(serializer.data)


@api_view(['GET'])
def list_preprocessing_runs(request, dataset_id):
    """List all preprocessing runs for a dataset."""
    runs = PreprocessingRun.objects.filter(dataset_id=dataset_id)
    serializer = PreprocessingRunSerializer(runs, many=True)
    return Response(serializer.data)
