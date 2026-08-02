"""
==============================================================================
INFERENCE MODULE — Real-Time & Batch English-to-Arabic Machine Translation
==============================================================================

Purpose:
  This module manages transformer model loading and text generation for inference.
  
Key Features:
  1. Global In-Memory Model Caching: Avoids re-loading multi-hundred megabyte transformer
     weights from disk on every HTTP translation request.
  2. Flexible Checkpoint Resolution: Dynamically locates fine-tuned models (`best_model`),
     local base models (`opus-mt-en-ar`), or online HuggingFace model repos.
  3. Single-Sentence Translation (`translate`): Fast inference with execution timing.
  4. Batch Translation (`translate_batch`): High-throughput vectorized inference for arrays.
==============================================================================
"""
import os
import time
import torch

# ----------------------------------------------------------------------------
# Global In-Memory Model Cache Variables
# ----------------------------------------------------------------------------
# Keeps loaded model instance, tokenizer, and target model path in VRAM/RAM
_model = None
_tokenizer = None
_model_path = None


def load_model(model_path=None, model_name='Helsinki-NLP/opus-mt-en-ar'):
    """
    Load HuggingFace MarianMT model and tokenizer into RAM/VRAM with caching.

    Checks if the requested model checkpoint matches the currently cached instance.
    If already loaded, returns the cached instances immediately.
    Otherwise, loads the tokenizer & model from local disk or HuggingFace Hub.

    Args:
        model_path (str, optional): Target model directory path or identifier.
        model_name (str): Baseline HuggingFace model name fallback.

    Returns:
        tuple: (AutoModelForSeq2SeqLM, AutoTokenizer)
    """
    global _model, _tokenizer, _model_path

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # ------------------------------------------------------------------------
    # Step A: Resolve Root and Models Directories
    # ------------------------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    env_models_dir = os.getenv('MODEL_SAVE_DIR')

    candidate_search_dirs = []
    if env_models_dir and os.path.exists(env_models_dir):
        candidate_search_dirs.append(env_models_dir)
    candidate_search_dirs.append(os.path.join(project_root, 'models'))
    candidate_search_dirs.append(project_root)

    # Normalize/clean model_path string input
    if isinstance(model_path, str):
        model_path = model_path.strip()

    target_path = None

    def find_existing_path(path_str):
        """
        Helper: Searches for path_str locally under exact path, MODEL_SAVE_DIR, models directory, or project root.
        """
        if not path_str:
            return None
        if os.path.exists(path_str):
            return os.path.abspath(path_str)
        # Search candidate parent directories
        for parent_dir in candidate_search_dirs:
            candidate = os.path.join(parent_dir, path_str)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    # ------------------------------------------------------------------------
    # Step B: Determine Checkpoint Path Priority
    # ------------------------------------------------------------------------
    # Check if baseline model was explicitly requested
    is_baseline_requested = model_path in (model_name, 'baseline', 'opus-mt-en-ar', '')

    if is_baseline_requested:
        # Load local base model first, or fall back to HuggingFace online ID
        target_path = find_existing_path('opus-mt-en-ar') or model_name
    elif model_path is not None:
        target_path = find_existing_path(model_path)
        if not target_path:
            # Fall back to best_model -> local base -> online baseline
            target_path = find_existing_path('best_model') or find_existing_path('opus-mt-en-ar') or model_name
    else:
        # Default priority: fine-tuned best_model -> local base -> online baseline
        target_path = find_existing_path('best_model') or find_existing_path('opus-mt-en-ar') or model_name

    # ------------------------------------------------------------------------
    # Step C: Cache Validation Check
    # ------------------------------------------------------------------------
    norm_target = os.path.normpath(target_path).lower() if target_path else None
    norm_cached = os.path.normpath(_model_path).lower() if _model_path else None

    # Skip loading from disk if the requested model is already cached in memory
    if _model is not None and norm_cached == norm_target:
        return _model, _tokenizer

    from ml.utils import to_device_safe

    # ------------------------------------------------------------------------
    # Step D: Load Tokenizer and Model to Hardware Device (GPU / CPU)
    # ------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference Engine] Loading model from: {target_path} (Hardware: {device})")

    _tokenizer = AutoTokenizer.from_pretrained(target_path)
    _model = AutoModelForSeq2SeqLM.from_pretrained(target_path)
    _model = to_device_safe(_model, device)
    _model.eval()  # Set model to evaluation mode (disables dropout layers)
    _model_path = target_path

    return _model, _tokenizer


def translate(text, model_path=None, max_length=128, num_beams=4):
    """
    Translate a single English text string into Arabic.

    Args:
        text (str): Source English input sentence or paragraph.
        model_path (str, optional): Custom path to model checkpoint.
        max_length (int): Maximum token length for target Arabic generation.
        num_beams (int): Beam search width for decoding quality.

    Returns:
        dict: {
            'translated_text': str,
            'translation_time_ms': float,
            'model_used': str
        }
    """
    # Step 1: Obtain model and tokenizer instances (using cache)
    model, tokenizer = load_model(model_path)
    device = next(model.parameters()).device

    start_time = time.time()

    # Step 2: Tokenize English input text into PyTorch Tensors
    inputs = tokenizer(
        text, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length
    ).to(device)

    # Step 3: Run autoregressive sequence generation with Beam Search (no gradient computation needed)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Step 4: Decode output token IDs back into human-readable Arabic text
    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    elapsed_ms = (time.time() - start_time) * 1000

    return {
        'translated_text': translated,
        'translation_time_ms': round(elapsed_ms, 2),
        'model_used': _model_path or 'unknown',
    }


def translate_batch(texts, model_path=None, max_length=128, num_beams=4, batch_size=16):
    """
    Translate a list of English texts into Arabic in parallel batches.

    Args:
        texts (list of str): Array of English input sentences.
        model_path (str, optional): Custom path to model checkpoint.
        max_length (int): Maximum output token length per sentence.
        num_beams (int): Beam search width.
        batch_size (int): Number of sentences to process per GPU forward pass.

    Returns:
        list of str: Translated Arabic sentences matching input order.
    """
    model, tokenizer = load_model(model_path)
    device = next(model.parameters()).device
    translations = []

    # Process sentences in sub-batches to manage GPU memory usage
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Vectorized tokenization for the current sub-batch
        inputs = tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(device)

        # Vectorized sequence generation
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Batch decoding of generated tokens to Arabic text strings
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

    return translations

