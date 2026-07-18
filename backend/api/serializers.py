"""Serializers for Machine Translator API."""
from rest_framework import serializers
from .models import Dataset, PreprocessingRun, TrainingJob, EvaluationResult, TranslationLog


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'


class DatasetUploadSerializer(serializers.Serializer):
    """Serializer for file upload endpoint."""
    file = serializers.FileField()
    name = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, default='', allow_blank=True)
    en_column = serializers.CharField(max_length=100, default='en')
    ar_column = serializers.CharField(max_length=100, default='ar')


class PreprocessingRunSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = PreprocessingRun
        fields = '__all__'


class TrainingJobSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = TrainingJob
        fields = '__all__'


class TrainingStartSerializer(serializers.Serializer):
    """Serializer for starting a training job."""
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
    training_job_status = serializers.CharField(source='training_job.status', read_only=True)

    class Meta:
        model = EvaluationResult
        fields = '__all__'


class TranslationRequestSerializer(serializers.Serializer):
    """Serializer for translation requests."""
    text = serializers.CharField(max_length=5000)
    model_path = serializers.CharField(required=False, default='', allow_blank=True)


class TranslationResponseSerializer(serializers.Serializer):
    """Serializer for translation responses."""
    source_text = serializers.CharField()
    translated_text = serializers.CharField()
    translation_time_ms = serializers.FloatField()
    model_used = serializers.CharField()


class TranslationLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TranslationLog
        fields = '__all__'
