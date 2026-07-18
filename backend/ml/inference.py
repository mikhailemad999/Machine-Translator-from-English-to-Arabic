"""
Inference Module — Real-time EN→AR translation.

Loads the best saved model checkpoint and translates single sentences.
"""
import os
import time
import torch


# Global model cache
_model = None
_tokenizer = None
_model_path = None


def load_model(model_path=None, model_name='Helsinki-NLP/opus-mt-en-ar'):
    """
    Load translation model for inference.

    Uses a global cache to avoid reloading on every request.
    Falls back to pretrained model if no fine-tuned checkpoint exists.
    """
    global _model, _tokenizer, _model_path

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Resolve directories relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, 'models')):
            break
        parent = os.path.dirname(project_root)
        if parent == project_root:
            break
        project_root = parent

    models_dir = os.path.join(project_root, 'models')

    # Normalize/clean model_path
    if isinstance(model_path, str):
        model_path = model_path.strip()
    if model_path == '':
        model_path = None

    target_path = None

    # Helper to check candidates
    def find_existing_path(path_str):
        if not path_str:
            return None
        if os.path.exists(path_str):
            return os.path.abspath(path_str)
        # Try relative to models directory or project root
        for parent_dir in [models_dir, project_root]:
            candidate = os.path.join(parent_dir, path_str)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    # Check if baseline was explicitly requested
    is_baseline_requested = model_path in (model_name, 'baseline', 'opus-mt-en-ar')

    if is_baseline_requested:
        # Load local base model first
        target_path = find_existing_path('opus-mt-en-ar') or model_name
    elif model_path:
        # Check if the requested path exists
        target_path = find_existing_path(model_path)
        if not target_path:
            # If specified path not found, fall back to best model, then baseline
            target_path = find_existing_path('best_model') or find_existing_path('opus-mt-en-ar') or model_name
    else:
        # No path specified: default to best_model, then local base model, then online baseline
        target_path = find_existing_path('best_model') or find_existing_path('opus-mt-en-ar') or model_name

    # Check if already loaded
    if _model is not None and _model_path == target_path:
        return _model, _tokenizer

    # Load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {target_path} (device: {device})")

    _tokenizer = AutoTokenizer.from_pretrained(target_path)
    _model = AutoModelForSeq2SeqLM.from_pretrained(target_path).to(device)
    _model.eval()
    _model_path = target_path

    return _model, _tokenizer



def translate(text, model_path=None, max_length=128, num_beams=4):
    """
    Translate English text to Arabic.

    Args:
        text: English input text
        model_path: Optional path to model checkpoint
        max_length: Maximum output length
        num_beams: Beam search width

    Returns:
        dict with translated_text, translation_time_ms, model_used
    """
    model, tokenizer = load_model(model_path)
    device = next(model.parameters()).device

    start_time = time.time()

    inputs = tokenizer(
        text, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    elapsed_ms = (time.time() - start_time) * 1000

    return {
        'translated_text': translated,
        'translation_time_ms': round(elapsed_ms, 2),
        'model_used': _model_path or 'unknown',
    }


def translate_batch(texts, model_path=None, max_length=128, num_beams=4, batch_size=16):
    """
    Translate a batch of English texts to Arabic.

    Args:
        texts: List of English input texts
        model_path: Optional path to model checkpoint
        max_length: Maximum output length
        num_beams: Beam search width
        batch_size: Processing batch size

    Returns:
        List of translated texts
    """
    model, tokenizer = load_model(model_path)
    device = next(model.parameters()).device
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

    return translations
