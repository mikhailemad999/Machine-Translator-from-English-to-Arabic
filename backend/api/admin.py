from django.contrib import admin
from .models import Dataset, PreprocessingRun, TrainingJob, EvaluationResult, TranslationLog

admin.site.register(Dataset)
admin.site.register(PreprocessingRun)
admin.site.register(TrainingJob)
admin.site.register(EvaluationResult)
admin.site.register(TranslationLog)
