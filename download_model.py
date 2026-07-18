"""
Download the Helsinki-NLP/opus-mt-en-ar model from HuggingFace.

Saves the model and tokenizer to models/opus-mt-en-ar/ for local inference.
Also runs a quick test translation to verify the download.
"""
import os
import sys

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'opus-mt-en-ar')


def download_model():
    """Download and save the translation model."""
    print("=" * 60)
    print("  Downloading Helsinki-NLP/opus-mt-en-ar")
    print("  (Marian Neural Machine Translation)")
    print("=" * 60)
    print()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = 'Helsinki-NLP/opus-mt-en-ar'

    # Download tokenizer
    print("[1/3] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Download model
    print("[2/3] Downloading model (~300MB)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Save locally
    print(f"[3/3] Saving to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    print()
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print()

    # Test translation
    print("Running test translation...")
    print("-" * 40)

    test_sentences = [
        "Hello, how are you?",
        "Machine learning is changing the world.",
        "I love studying computer science.",
        "The weather is beautiful today.",
        "Thank you for your help.",
    ]

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = model.to(device)
    model.eval()

    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=128, num_beams=4)
        translation = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"  EN: {sentence}")
        try:
            print(f"  AR: {translation}")
        except UnicodeEncodeError:
            # Fallback for Windows consoles that do not support UTF-8 display by default
            print(f"  AR (safe): {translation.encode('ascii', 'xmlcharrefreplace').decode('ascii')}")
        print()

    print("=" * 60)
    print("  Model download and verification COMPLETE!")
    print("=" * 60)
    return MODEL_DIR


if __name__ == '__main__':
    download_model()
