"""
SQL Server Models for Machine Translator.

These models store structured/cleaned data, training results, and metrics.
Raw data and unstructured artifacts go to MongoDB (via pymongo).
"""
from django.db import models


class Dataset(models.Model):
    """Metadata about an uploaded dataset."""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    file_type = models.CharField(max_length=20, choices=[
        ('csv', 'CSV'),
        ('tsv', 'TSV'),
        ('json', 'JSON'),
        ('parallel', 'Parallel Files'),
    ])
    total_pairs = models.IntegerField(default=0)
    cleaned_pairs = models.IntegerField(default=0)
    mongo_collection = models.CharField(max_length=255, help_text='MongoDB collection name for raw data')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=30, choices=[
        ('uploaded', 'Uploaded'),
        ('exploring', 'Exploring'),
        ('preprocessing', 'Preprocessing'),
        ('ready', 'Ready for Training'),
        ('training', 'Training'),
        ('completed', 'Completed'),
    ], default='uploaded')

    class Meta:
        ordering = ['-uploaded_at']
        db_table = 'datasets'

    def __str__(self):
        return f"{self.name} ({self.total_pairs} pairs)"


class PreprocessingRun(models.Model):
    """Records of a preprocessing pipeline run (Steps 1-6)."""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='preprocessing_runs')
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='running')

    # Step 1: Exploration
    original_shape = models.JSONField(default=dict, help_text='{"rows": N, "cols": N}')
    dtypes_info = models.JSONField(default=dict)
    sample_pairs = models.JSONField(default=list, help_text='First 10 sentence pairs')
    encoding_notes = models.TextField(blank=True, default='')

    # Step 2: Duplicates
    duplicate_count_full = models.IntegerField(default=0, help_text='Duplicates on EN+AR pair')
    duplicate_count_en = models.IntegerField(default=0, help_text='Duplicates on EN only')
    duplicate_count_ar = models.IntegerField(default=0, help_text='Duplicates on AR only')
    duplicate_pct = models.FloatField(default=0.0)
    pairs_after_dedup = models.IntegerField(default=0)

    # Step 3: Missing Values
    missing_en_count = models.IntegerField(default=0)
    missing_ar_count = models.IntegerField(default=0)
    missing_en_pct = models.FloatField(default=0.0)
    missing_ar_pct = models.FloatField(default=0.0)
    pairs_after_missing = models.IntegerField(default=0)

    # Step 4: Outliers
    outlier_count_zscore = models.IntegerField(default=0)
    outlier_count_iqr = models.IntegerField(default=0)
    outlier_pct = models.FloatField(default=0.0)
    pairs_after_outliers = models.IntegerField(default=0)
    length_ratio_min = models.FloatField(default=0.0)
    length_ratio_max = models.FloatField(default=0.0)

    # Step 5: Visualizations
    chart_paths = models.JSONField(default=list, help_text='List of chart file paths')

    # Step 6: Imbalance
    imbalance_applicable = models.BooleanField(default=False)
    class_distribution_before = models.JSONField(default=dict)
    class_distribution_after = models.JSONField(default=dict)

    # Final cleaned dataset info
    final_pairs = models.IntegerField(default=0)

    class Meta:
        ordering = ['-started_at']
        db_table = 'preprocessing_runs'

    def __str__(self):
        return f"Preprocessing #{self.id} for {self.dataset.name}"


class TrainingJob(models.Model):
    """A training run for the translation model (Step 7)."""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='training_jobs')
    preprocessing_run = models.ForeignKey(PreprocessingRun, on_delete=models.SET_NULL, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='queued')

    # Hyperparameters
    model_name = models.CharField(max_length=255, default='Helsinki-NLP/opus-mt-en-ar')
    batch_size = models.IntegerField(default=4)
    gradient_accumulation = models.IntegerField(default=8)
    learning_rate = models.FloatField(default=5e-5)
    max_epochs = models.IntegerField(default=10)
    fp16 = models.BooleanField(default=True)
    weight_decay = models.FloatField(default=0.01)
    early_stopping_patience = models.IntegerField(default=3)

    # Training data split
    train_size = models.IntegerField(default=0)
    val_size = models.IntegerField(default=0)
    test_size = models.IntegerField(default=0)

    # Results
    best_epoch = models.IntegerField(null=True, blank=True)
    best_val_loss = models.FloatField(null=True, blank=True)
    best_val_bleu = models.FloatField(null=True, blank=True)
    learning_curve_path = models.CharField(max_length=500, blank=True, default='')
    model_checkpoint_path = models.CharField(max_length=500, blank=True, default='')

    # Overfitting / Underfitting diagnosis
    diagnosis = models.CharField(max_length=30, choices=[
        ('overfitting', 'Overfitting'),
        ('underfitting', 'Underfitting'),
        ('well_fit', 'Well Fit'),
        ('unknown', 'Unknown'),
    ], default='unknown')
    diagnosis_notes = models.TextField(blank=True, default='')

    # Epoch-by-epoch data
    epoch_data = models.JSONField(default=list, help_text='[{epoch, train_loss, val_loss, val_bleu}]')

    class Meta:
        ordering = ['-started_at']
        db_table = 'training_jobs'

    def __str__(self):
        return f"Training #{self.id} - {self.status}"


class EvaluationResult(models.Model):
    """Evaluation metrics for a trained model (Step 8)."""
    training_job = models.OneToOneField(TrainingJob, on_delete=models.CASCADE, related_name='evaluation')
    evaluated_at = models.DateTimeField(auto_now_add=True)

    # MT Metrics
    bleu_score = models.FloatField(default=0.0)
    chrf_score = models.FloatField(default=0.0)
    ter_score = models.FloatField(default=0.0)

    # Baseline comparison
    baseline_bleu = models.FloatField(default=0.0)
    baseline_chrf = models.FloatField(default=0.0)
    baseline_ter = models.FloatField(default=0.0)

    # Example translations (qualitative)
    example_translations = models.JSONField(
        default=list,
        help_text='[{source, reference, baseline_output, model_output}]'
    )

    # Exit condition check
    meets_target = models.BooleanField(default=False)
    target_notes = models.TextField(blank=True, default='')

    class Meta:
        db_table = 'evaluation_results'

    def __str__(self):
        return f"Eval for Training #{self.training_job_id}: BLEU={self.bleu_score:.2f}"


class TranslationLog(models.Model):
    """Log of real-time translation requests."""
    source_text = models.TextField()
    translated_text = models.TextField()
    model_used = models.CharField(max_length=255)
    translation_time_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        db_table = 'translation_logs'

    def __str__(self):
        return f"Translation: {self.source_text[:50]}..."
