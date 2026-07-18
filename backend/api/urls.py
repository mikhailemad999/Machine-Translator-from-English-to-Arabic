"""API URL Configuration — All endpoints."""
from django.urls import path
from .views import (
    dataset_views,
    preprocess_views,
    eda_views,
    training_views,
    evaluation_views,
    translate_views,
)

urlpatterns = [
    # Dataset endpoints
    path('dataset/', dataset_views.list_datasets, name='dataset-list'),
    path('dataset/<int:pk>/', dataset_views.get_dataset, name='dataset-detail'),
    path('dataset/upload/', dataset_views.upload_dataset, name='dataset-upload'),
    path('dataset/download-sample/', dataset_views.download_sample_dataset, name='dataset-download'),

    # Preprocessing endpoints
    path('preprocess/run/', preprocess_views.run_preprocessing, name='preprocess-run'),
    path('preprocess/<int:pk>/', preprocess_views.get_preprocessing_run, name='preprocess-detail'),
    path('preprocess/list/<int:dataset_id>/', preprocess_views.list_preprocessing_runs, name='preprocess-list'),

    # EDA endpoints
    path('eda/report/<int:dataset_id>/', eda_views.get_eda_report, name='eda-report'),
    path('eda/chart/<str:filename>', eda_views.serve_chart, name='eda-chart'),
    path('eda/charts/', eda_views.list_charts, name='eda-charts-list'),

    # Training endpoints
    path('train/start/', training_views.start_training, name='train-start'),
    path('train/status/<int:job_id>/', training_views.get_training_status, name='train-status'),
    path('train/list/', training_views.list_training_jobs, name='train-list'),

    # Evaluation endpoints
    path('evaluate/run/<int:job_id>/', evaluation_views.run_evaluation, name='evaluate-run'),
    path('evaluate/<int:job_id>/', evaluation_views.get_evaluation, name='evaluate-detail'),

    # Translation endpoints
    path('translate/', translate_views.translate_text, name='translate'),
    path('translate/batch/', translate_views.translate_batch, name='translate-batch'),
    path('translate/history/', translate_views.translation_history, name='translate-history'),
    path('', translate_views.api_root, name='api-root'),
]
