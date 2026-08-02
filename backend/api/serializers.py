"""
==============================================================================
API SERIALIZERS MODULE — Django REST Framework (DRF) Data Normalization
==============================================================================

Purpose:
  This module defines DRF Serializers responsible for:
  1. Converting Django database model instances into JSON representations (Serialization).
  2. Parsing and validating incoming HTTP Request JSON payloads (Deserialization & Validation).
  3. Specifying input hyperparameter defaults for dataset uploads and training runs.
==============================================================================
"""
from rest_framework import serializers
from .models import Dataset, PreprocessingRun, TrainingJob, EvaluationResult, TranslationLog


# ----------------------------------------------------------------------------
# 1. Dataset Serializers
# ----------------------------------------------------------------------------

class DatasetSerializer(serializers.ModelSerializer):
    """
    ModelSerializer for reading, creating, and updating Dataset database records.
    Maps all fields from the `Dataset` model (`id`, `name`, `file_path`, `row_count`, etc.).
    """
    class Meta:
        model = Dataset
        fields = '__all__'


class DatasetUploadSerializer(serializers.Serializer):
    """
    Form & Payload Validator for multipart file upload requests (`POST /api/datasets/upload/`).
    
    Validates:
    - `file`: The uploaded binary file stream (.csv, .tsv, or .json).
    - `name`: Human-readable dataset title.
    - `description`: Optional user notes or source documentation.
    - `en_column`: Source English text column header name (defaults to 'en').
    - `ar_column`: Target Arabic text column header name (defaults to 'ar').
    """
    file = serializers.FileField()
    name = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, default='', allow_blank=True)
    en_column = serializers.CharField(max_length=100, default='en')
    ar_column = serializers.CharField(max_length=100, default='ar')


# ----------------------------------------------------------------------------
# 2. Preprocessing Serializers
# ----------------------------------------------------------------------------

class PreprocessingRunSerializer(serializers.ModelSerializer):
    """
    Serializer for recording and displaying data cleaning and text preprocessing operations.
    Includes read-only `dataset_name` fetched via foreign key relationship.
    """
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = PreprocessingRun
        fields = '__all__'


# ----------------------------------------------------------------------------
# 3. Model Training Serializers
# ----------------------------------------------------------------------------

class TrainingJobSerializer(serializers.ModelSerializer):
    """
    Serializer for training job database records tracking status, loss, and BLEU metrics.
    """
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = TrainingJob
        fields = '__all__'


class TrainingStartSerializer(serializers.Serializer):
    """
    Request Payload Validator for starting model fine-tuning (`POST /api/training/start/`).
    
    Hyperparameters & Defaults:
    - `dataset_id`: Database primary key of dataset to train on.
    - `preprocessing_run_id`: Optional ID of preprocessed dataset version.
    - `batch_size`: Per-device batch size (default: 4).
    - `gradient_accumulation`: Gradient accumulation steps to simulate larger batch sizes (default: 8).
    - `learning_rate`: AdamW learning rate (default: 5e-5).
    - `max_epochs`: Maximum training epochs (default: 10).
    - `fp16`: Enables 16-bit floating point precision on NVIDIA GPUs (default: True).
    - `weight_decay`: L2 regularization coefficient (default: 0.01).
    - `early_stopping_patience`: Epoch count to wait before stopping if validation loss stops improving (default: 3).
    """
    dataset_id = serializers.IntegerField()
    preprocessing_run_id = serializers.IntegerField(required=False, allow_null=True)
    batch_size = serializers.IntegerField(default=4)
    gradient_accumulation = serializers.IntegerField(default=8)
    learning_rate = serializers.FloatField(default=5e-5)
    max_epochs = serializers.IntegerField(default=10)
    fp16 = serializers.BooleanField(default=True)
    weight_decay = serializers.FloatField(default=0.01)
    early_stopping_patience = serializers.IntegerField(default=3)


# ----------------------------------------------------------------------------
# 4. Model Evaluation Serializers
# ----------------------------------------------------------------------------

class EvaluationResultSerializer(serializers.ModelSerializer):
    """
    Serializer for automated metric reports (BLEU, TER, chrF, Perplexity).
    """
    training_job_status = serializers.CharField(source='training_job.status', read_only=True)

    class Meta:
        model = EvaluationResult
        fields = '__all__'


# ----------------------------------------------------------------------------
# 5. Translation Endpoint Serializers
# ----------------------------------------------------------------------------

class TranslationRequestSerializer(serializers.Serializer):
    """
    Validator for single translation HTTP requests (`POST /api/translate/`).
    
    Fields:
    - `text`: Input English sentence (max 5000 chars).
    - `model_path`: Optional specific checkpoint directory path or model tag.
    """
    text = serializers.CharField(max_length=5000)
    model_path = serializers.CharField(required=False, default='', allow_blank=True)


class TranslationResponseSerializer(serializers.Serializer):
    """
    Response formatter returned to UI for translation requests.
    """
    source_text = serializers.CharField()
    translated_text = serializers.CharField()
    translation_time_ms = serializers.FloatField()
    model_used = serializers.CharField()


class TranslationLogSerializer(serializers.ModelSerializer):
    """
    Serializer for reading and searching historical translation query logs.
    """
    class Meta:
        model = TranslationLog
        fields = '__all__'

