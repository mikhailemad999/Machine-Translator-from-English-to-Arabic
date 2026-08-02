import os
import unittest
import pandas as pd
import numpy as np
from django.test import TestCase

from ml.data_loader import explore_dataset
from ml.duplicates import handle_duplicates
from ml.missing_values import handle_missing_values
from ml.outliers import handle_outliers
from ml.visualizations import generate_all_charts
from ml.imbalance import handle_imbalance
from ml.evaluator import _compute_metrics


class MLPipelineTestCase(unittest.TestCase):
    """Unit tests for Steps 1-8 ML and Preprocessing pipeline."""

    def setUp(self):
        # Create a mock dataframe for testing
        self.mock_data = pd.DataFrame({
            'en': [
                'Hello world',
                'This is a test',
                'English sentence',
                'Another one',
                'Yet another English sentence',
                'Hello world',  # Duplicate
                None,           # Missing
                'Outlier check with long words that is very long ' * 20  # Outlier
            ],
            'ar': [
                'مرحبا بالعالم',
                'هذا اختبار',
                'جملة إنجليزية',
                'واحدة أخرى',
                'جملة إنجليزية أخرى',
                'مرحبا بالعالم',  # Duplicate
                'مفقود',
                'تحقق من القيم المتطرفة'
            ]
        })

    def test_step_1_explore(self):
        """Test dataset exploration."""
        report = explore_dataset(self.mock_data)
        self.assertEqual(report['shape']['rows'], len(self.mock_data))
        self.assertEqual(report['shape']['cols'], 2)
        self.assertIn('sample_pairs', report)

    def test_step_2_duplicates(self):
        """Test duplicate handling."""
        df_cleaned, report = handle_duplicates(self.mock_data)
        self.assertEqual(report['duplicates_full_pair']['count'], 2)
        self.assertEqual(report['removed_count'], 1)
        self.assertEqual(len(df_cleaned), len(self.mock_data) - 1)

    def test_step_3_missing_values(self):
        """Test missing values detection and removal."""
        df_dedup, _ = handle_duplicates(self.mock_data)
        df_cleaned, report = handle_missing_values(df_dedup)
        # 1 row has null 'en', should be dropped
        self.assertEqual(report['summary']['en_missing'], 1)
        self.assertEqual(len(df_cleaned), len(df_dedup) - 1)

    def test_step_4_outliers(self):
        """Test outliers handling."""
        df_dedup, _ = handle_duplicates(self.mock_data)
        df_no_missing, _ = handle_missing_values(df_dedup)
        df_cleaned, report = handle_outliers(df_no_missing)
        # Outlier row should be capped or filtered
        self.assertLessEqual(len(df_cleaned), len(df_no_missing))
        self.assertIn('outliers', report)

    def test_step_6_imbalance(self):
        """Test imbalance analysis."""
        df_dedup, _ = handle_duplicates(self.mock_data)
        df_no_missing, _ = handle_missing_values(df_dedup)
        df_cleaned, _ = handle_outliers(df_no_missing)
        df_balanced, report = handle_imbalance(df_cleaned, strategy='none')
        self.assertIn('distribution_before', report)

    def test_step_8_metrics(self):
        """Test evaluation metrics helper."""
        predictions = ['هذا هو كتاب اختبار للمترجم الآلي', 'مرحبا بك في العالم الجديد اليوم']
        references = ['هذا هو كتاب اختبار للمترجم الآلي', 'مرحبا بك في العالم الجديد اليوم']
        metrics = _compute_metrics(predictions, references)
        self.assertEqual(metrics['bleu'], 100.0)
        self.assertEqual(metrics['chrf'], 100.0)
        self.assertEqual(metrics['ter'], 0.0)


    def test_step_1_explore_50_samples(self):
        """Test dataset exploration returns up to 50 preview pairs and rich statistical metrics."""
        large_mock_data = pd.DataFrame({
            'en': [f'Hello sentence {i} for MT testing' for i in range(100)],
            'ar': [f'مرحبا جملة رقم {i} للاختبار' for i in range(100)]
        })
        report = explore_dataset(large_mock_data)
        self.assertEqual(report['shape']['rows'], 100)
        self.assertEqual(len(report['sample_pairs']), 50)
        self.assertIn('dataset_health_score', report)
        self.assertIn('database_metrics', report)
        self.assertIn('length_ratios', report)
        self.assertEqual(report['dataset_health_score'], 100.0)


class APIRoutingTestCase(TestCase):
    """Test case for API routing and endpoints."""

    def test_api_root_status_code(self):
        """Test that accessing /api/ returns 200 OK."""
        response = self.client.get('/api/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], 'running')
        self.assertIn('endpoints', response.data)

    def test_dataset_upload_api(self):
        """Test POST /api/dataset/upload/ creates record and returns 50 sample preview."""
        import io
        csv_content = 'en,ar\n' + '\n'.join([f'Sentence {i},جملة {i}' for i in range(60)])
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = 'test_upload.csv'

        response = self.client.post('/api/dataset/upload/', {
            'file': csv_file,
            'name': 'Test Upload Corpus',
            'description': 'Integration test upload',
            'en_column': 'en',
            'ar_column': 'ar',
        }, format='multipart')

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['total_pairs'], 60)
        self.assertIn('exploration_report', response.data)
        self.assertEqual(len(response.data['exploration_report']['sample_pairs']), 50)

    def test_start_training_api(self):
        """Test POST /api/train/start/ queued response on uploaded dataset."""
        from api.models import Dataset
        ds = Dataset.objects.create(
            name='Test Training Dataset',
            file_type='csv',
            total_pairs=100,
            mongo_collection='dataset_test_training_raw',
            status='uploaded'
        )

        response = self.client.post('/api/train/start/', {
            'dataset_id': ds.id,
            'batch_size': 4,
            'gradient_accumulation': 8,
            'learning_rate': 5e-5,
            'max_epochs': 1,
            'early_stopping_patience': 3,
            'fp16': False,
        }, format='json')

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['status'], 'queued')
        self.assertIn('job_id', response.data)

