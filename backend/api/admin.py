"""
==============================================================================
DJANGO ADMIN MODULE — Database Administration Dashboard Registrations
==============================================================================

Purpose:
  This module registers all database models with the Django Admin site,
  enabling superusers to view, inspect, and manage Datasets, Preprocessing Runs,
  Training Jobs, Evaluation Results, and Translation Audit Logs via the web UI.
==============================================================================
"""
from django.contrib import admin
from .models import Dataset, PreprocessingRun, TrainingJob, EvaluationResult, TranslationLog

# Register core models to Django Admin interface
admin.site.register(Dataset)
admin.site.register(PreprocessingRun)
admin.site.register(TrainingJob)
admin.site.register(EvaluationResult)
admin.site.register(TranslationLog)
