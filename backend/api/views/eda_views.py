"""
EDA Views — Serve charts and exploration statistics.

GET /api/eda/report/{dataset_id}/ — Get full EDA report
GET /api/eda/chart/{filename} — Serve a chart image
"""
import os
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from django.http import FileResponse, Http404

from mongodb_client import MongoDBClient
from api.models import Dataset, PreprocessingRun


@api_view(['GET'])
def get_eda_report(request, dataset_id):
    """Get the full EDA report for a dataset."""
    try:
        dataset = Dataset.objects.get(pk=dataset_id)
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

    # Get the latest preprocessing run
    run = PreprocessingRun.objects.filter(
        dataset=dataset, status='completed'
    ).first()

    # Get MongoDB EDA report
    mongo_report = MongoDBClient.get_eda_report(dataset_id)

    response_data = {
        'dataset_id': dataset_id,
        'dataset_name': dataset.name,
        'total_pairs': dataset.total_pairs,
        'cleaned_pairs': dataset.cleaned_pairs,
        'status': dataset.status,
    }

    if run:
        response_data['preprocessing'] = {
            'run_id': run.id,
            'original_shape': run.original_shape,
            'duplicate_count': run.duplicate_count_full,
            'duplicate_pct': run.duplicate_pct,
            'missing_en_pct': run.missing_en_pct,
            'missing_ar_pct': run.missing_ar_pct,
            'outlier_pct': run.outlier_pct,
            'final_pairs': run.final_pairs,
            'chart_paths': run.chart_paths,
            'imbalance_applicable': run.imbalance_applicable,
            'class_distribution_before': run.class_distribution_before,
            'class_distribution_after': run.class_distribution_after,
        }

    if mongo_report:
        # Remove MongoDB internal fields
        mongo_report.pop('_id', None)
        response_data['detailed_report'] = mongo_report

    return Response(response_data)


@api_view(['GET'])
def serve_chart(request, filename):
    """Serve a chart image file."""
    chart_path = os.path.join(str(settings.CHARTS_DIR), filename)

    if not os.path.exists(chart_path):
        return Response({'error': f'Chart not found: {filename}'}, status=status.HTTP_404_NOT_FOUND)

    response = FileResponse(
        open(chart_path, 'rb'),
        content_type='image/png',
    )
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


@api_view(['GET'])
def list_charts(request):
    """List all available chart files."""
    charts_dir = str(settings.CHARTS_DIR)
    if not os.path.exists(charts_dir):
        return Response({'charts': []})

    charts = []
    for f in os.listdir(charts_dir):
        if f.endswith(('.png', '.jpg', '.svg')):
            charts.append({
                'filename': f,
                'url': f'/api/eda/chart/{f}',
                'size_bytes': os.path.getsize(os.path.join(charts_dir, f)),
            })

    return Response({'charts': charts})
