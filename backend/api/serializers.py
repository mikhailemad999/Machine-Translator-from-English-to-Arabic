"""Serializers for Machine Translator API."""
from rest_framework import serializers
from .models import Dataset, PreprocessingRun, TrainingJob, EvaluationResult, TranslationLog


class DatasetSerializer(serializers.ModelSerializer):
    """Serializer for the Dataset metadata model."""
    class Meta:
        model = Dataset
        fields = '__all__'


class DatasetUploadSerializer(serializers.Serializer):
    """Serializer for file upload endpoint validation."""
    file = serializers.FileField()
    name = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, default='', allow_blank=True)
    en_column = serializers.CharField(max_length=100, default='en')
    ar_column = serializers.CharField(max_length=100, default='ar')


class PreprocessingRunSerializer(serializers.ModelSerializer):
    """Serializer for details of a preprocessing run."""
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = PreprocessingRun
        fields = '__all__'


class TrainingJobSerializer(serializers.ModelSerializer):
    """Serializer for training job details and hyperparameters."""
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = TrainingJob
        fields = '__all__'


class TrainingStartSerializer(serializers.Serializer):
    """Serializer for validation configuration when starting a training job."""
    dataset_id = serializers.IntegerField()
    preprocessing_run_id = serializers.IntegerField(required=False)
    batch_size = serializers.IntegerField(default=4)
    gradient_accumulation = serializers.IntegerField(default=8)
    learning_rate = serializers.FloatField(default=5e-5)
    max_epochs = serializers.IntegerField(default=10)
    fp16 = serializers.BooleanField(default=True)
    weight_decay = serializers.FloatField(default=0.01)
    early_stopping_patience = serializers.IntegerField(default=3)


class EvaluationResultSerializer(serializers.ModelSerializer):
    """Serializer for model translation evaluation metrics."""
    training_job_status = serializers.CharField(source='training_job.status', read_only=True)

    class Meta:
        model = EvaluationResult
        fields = '__all__'


class TranslationRequestSerializer(serializers.Serializer):
    """Serializer validating standard translation requests."""
    text = serializers.CharField(max_length=5000)
    model_path = serializers.CharField(required=False, default='', allow_blank=True)


class TranslationResponseSerializer(serializers.Serializer):
    """Serializer formatting responses for translation requests."""
    source_text = serializers.CharField()
    translated_text = serializers.CharField()
    translation_time_ms = serializers.FloatField()
    model_used = serializers.CharField()


class TranslationLogSerializer(serializers.ModelSerializer):
    """Serializer for tracking history of translator logs."""
    class Meta:
        model = TranslationLog
        fields = '__all__'
