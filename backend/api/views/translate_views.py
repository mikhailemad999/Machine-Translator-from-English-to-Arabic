"""
Translation Views — Real-time EN→AR translation.

POST /api/translate — Translate English text to Arabic
GET  /api/translate/history/ — Get translation history
"""
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.models import TranslationLog
from api.serializers import TranslationRequestSerializer, TranslationResponseSerializer, TranslationLogSerializer


@api_view(['POST'])
def translate_text(request):
    """
    Translate English text to Arabic.

    Expects: { "text": "Hello, how are you?", "model_path": "" (optional) }
    """
    serializer = TranslationRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    text = serializer.validated_data['text']
    model_path = serializer.validated_data.get('model_path', '')

    try:
        from ml.inference import translate

        result = translate(
            text=text,
            model_path=model_path if model_path else None,
        )

        # Log the translation
        TranslationLog.objects.create(
            source_text=text,
            translated_text=result['translated_text'],
            model_used=result['model_used'],
            translation_time_ms=result['translation_time_ms'],
        )

        response_data = {
            'source_text': text,
            'translated_text': result['translated_text'],
            'translation_time_ms': result['translation_time_ms'],
            'model_used': result['model_used'],
        }

        return Response(response_data)

    except Exception as e:
        return Response(
            {'error': f'Translation failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def translation_history(request):
    """Get recent translation history."""
    limit = int(request.query_params.get('limit', 20))
    logs = TranslationLog.objects.all()[:limit]
    serializer = TranslationLogSerializer(logs, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def translate_batch(request):
    """
    Translate a batch of English texts to Arabic.
    Expects: { "texts": ["sentence1", "sentence2"], "model_path": "path" (optional) }
    """
    texts = request.data.get('texts', [])
    model_path = request.data.get('model_path', '')

    if not isinstance(texts, list) or not texts:
        return Response({'error': 'texts must be a non-empty list'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        from ml.inference import translate_batch as do_batch

        translations = do_batch(
            texts=texts,
            model_path=model_path if model_path else None
        )

        return Response({'translations': translations})

    except Exception as e:
        return Response(
            {'error': f'Batch translation failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def api_root(request):
    """API Root — Check API status and endpoints."""
    return Response({
        'status': 'running',
        'message': 'Welcome to the English-to-Arabic Machine Translator API',
        'endpoints': {
            'dataset': '/api/dataset/',
            'dataset_upload': '/api/dataset/upload/',
            'dataset_download_sample': '/api/dataset/download-sample/',
            'preprocess_run': '/api/preprocess/run/',
            'eda_report': '/api/eda/report/<dataset_id>/',
            'eda_charts': '/api/eda/charts/',
            'train_start': '/api/train/start/',
            'train_list': '/api/train/list/',
            'translate': '/api/translate/',
            'translate_batch': '/api/translate/batch/',
            'translate_history': '/api/translate/history/',
        }
    })

