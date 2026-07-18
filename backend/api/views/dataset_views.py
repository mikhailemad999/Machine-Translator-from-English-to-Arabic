"""
Dataset Views — Upload and manage EN-AR datasets.

POST /api/dataset/upload — Upload raw EN-AR corpus, store in MongoDB
GET  /api/dataset/ — List all datasets
GET  /api/dataset/{id}/ — Get dataset details
POST /api/dataset/download-sample/ — Download sample dataset from HuggingFace
"""
import os
import io
import pandas as pd
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

from api.models import Dataset
from api.serializers import DatasetSerializer, DatasetUploadSerializer
from mongodb_client import MongoDBClient
from ml.data_loader import load_dataset, explore_dataset


@api_view(['GET'])
def list_datasets(request):
    """List all uploaded datasets."""
    datasets = Dataset.objects.all()
    serializer = DatasetSerializer(datasets, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def get_dataset(request, pk):
    """Get a specific dataset's details."""
    try:
        dataset = Dataset.objects.get(pk=pk)
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

    serializer = DatasetSerializer(dataset)

    # Also fetch MongoDB info
    mongo_count = MongoDBClient.get_raw_dataset_count(dataset.mongo_collection)
    data = serializer.data
    data['mongo_document_count'] = mongo_count

    return Response(data)


@api_view(['POST'])
def upload_dataset(request):
    """
    Upload a dataset file (CSV/TSV/JSON).
    Stores raw data in MongoDB, metadata in SQL Server.
    """
    serializer = DatasetUploadSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    file = serializer.validated_data['file']
    name = serializer.validated_data['name']
    description = serializer.validated_data.get('description', '')
    en_column = serializer.validated_data.get('en_column', 'en')
    ar_column = serializer.validated_data.get('ar_column', 'ar')

    # Detect file type
    filename = file.name.lower()
    if filename.endswith('.csv'):
        file_type = 'csv'
    elif filename.endswith('.tsv'):
        file_type = 'tsv'
    elif filename.endswith('.json'):
        file_type = 'json'
    else:
        return Response(
            {'error': f'Unsupported file type: {filename}. Use CSV, TSV, or JSON.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Load and parse the file
        df = load_dataset(
            file_obj=file,
            file_type=file_type,
            en_column=en_column,
            ar_column=ar_column
        )

        # Create dataset record in SQL Server
        dataset = Dataset.objects.create(
            name=name,
            description=description,
            file_type=file_type,
            total_pairs=len(df),
            mongo_collection=f'dataset_{name.replace(" ", "_").lower()}_raw',
            status='uploaded',
        )

        # Store raw data in MongoDB
        records = df.to_dict(orient='records')
        stored_count = MongoDBClient.store_raw_dataset(
            dataset.mongo_collection, records
        )

        # Run initial exploration (Step 1)
        exploration_report = explore_dataset(df)

        # Store exploration report in MongoDB
        MongoDBClient.store_eda_report(dataset.id, {
            'type': 'exploration',
            'step': 1,
            'report': exploration_report,
        })

        return Response({
            'dataset_id': dataset.id,
            'name': dataset.name,
            'total_pairs': len(df),
            'stored_in_mongodb': stored_count,
            'mongo_collection': dataset.mongo_collection,
            'exploration_report': exploration_report,
            'message': 'Dataset uploaded successfully. Step 1 (Exploration) complete.',
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {'error': f'Failed to process file: {str(e)}'},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['POST'])
def download_sample_dataset(request):
    """
    Download a sample EN-AR dataset from HuggingFace.
    Uses opus_books en-ar subset.
    """
    max_samples = request.data.get('max_samples', 10000)

    try:
        from ml.data_loader import load_from_huggingface

        df = load_from_huggingface(
            dataset_name='opus_books',
            lang_pair='en-ar',
            max_samples=int(max_samples)
        )

        # Save to disk
        save_path = os.path.join(settings.DATA_DIR, 'opus_books_en_ar.csv')
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        df.to_csv(save_path, index=False)

        # Create dataset record
        dataset = Dataset.objects.create(
            name='OPUS Books EN-AR',
            description=f'OPUS Books English-Arabic parallel corpus ({len(df)} pairs)',
            file_type='csv',
            total_pairs=len(df),
            mongo_collection='dataset_opus_books_en_ar_raw',
            status='uploaded',
        )

        # Store in MongoDB
        records = df.to_dict(orient='records')
        stored = MongoDBClient.store_raw_dataset(dataset.mongo_collection, records)

        # Run exploration
        exploration_report = explore_dataset(df)
        MongoDBClient.store_eda_report(dataset.id, {
            'type': 'exploration',
            'step': 1,
            'report': exploration_report,
        })

        return Response({
            'dataset_id': dataset.id,
            'name': dataset.name,
            'total_pairs': len(df),
            'saved_to': save_path,
            'exploration_report': exploration_report,
            'message': f'Downloaded {len(df)} sentence pairs from OPUS Books.',
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {'error': f'Failed to download dataset: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
