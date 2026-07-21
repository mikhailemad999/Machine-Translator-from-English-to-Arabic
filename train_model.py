"""
Train the EN-AR translation model locally.

Uses the existing ml/trainer.py pipeline with settings from .env.
Run after download_model.py and download_dataset.py.
"""
import os
import sys

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add backend to path so we can import ml modules
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def main():
    """Run the training pipeline."""
    import pandas as pd
    import torch

    print("=" * 60)
    print("  EN-AR Translation Model Training")
    print("=" * 60)
    print()

    # Configuration from .env
    dataset_path = os.path.join(PROJECT_ROOT, 'data', 'en_ar_dataset.csv')
    model_name = os.getenv('MODEL_NAME', 'Helsinki-NLP/opus-mt-en-ar')
    save_dir = os.getenv('MODEL_SAVE_DIR', os.path.join(PROJECT_ROOT, 'models'))
    charts_dir = os.getenv('CHARTS_DIR', os.path.join(PROJECT_ROOT, 'charts'))
    batch_size = int(os.getenv('BATCH_SIZE', 4))
    grad_accum = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 8))
    learning_rate = float(os.getenv('LEARNING_RATE', 5e-5))
    max_epochs = int(os.getenv('MAX_EPOCHS', 10))
    fp16 = os.getenv('FP16', 'False').lower() == 'true'
    early_stopping = int(os.getenv('EARLY_STOPPING_PATIENCE', 3))

    # Check for local model first
    local_model = os.path.join(PROJECT_ROOT, 'models', 'opus-mt-en-ar')
    if os.path.exists(local_model):
        model_name = local_model
        print(f"Using local model: {local_model}")
    else:
        print(f"Using HuggingFace model: {model_name}")
        print("  (Run download_model.py first to avoid re-downloading)")

    # Check dataset
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run download_dataset.py first!")
        sys.exit(1)

    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: Training on CPU will be slow!")
        print("  Consider reducing MAX_EPOCHS or dataset size.")

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total pairs: {len(df)}")

    # For CPU training, use a smaller subset to be practical
    if device == 'cpu' and len(df) > 5000:
        print(f"\nCPU mode: Using 5,000 pairs (from {len(df)}) for practical training time")
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    print()
    print("Training Configuration:")
    print(f"  Model:            {model_name}")
    print(f"  Dataset size:     {len(df)} pairs")
    print(f"  Batch size:       {batch_size}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch:  {batch_size * grad_accum}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Max epochs:       {max_epochs}")
    print(f"  FP16:             {fp16}")
    print(f"  Early stopping:   {early_stopping} epochs")
    print(f"  Save dir:         {save_dir}")
    print(f"  Charts dir:       {charts_dir}")
    print()

    # Confirm before starting
    print("Starting training in 3 seconds... (Ctrl+C to cancel)")
    import time
    time.sleep(3)

    # Import trainer
    from ml.trainer import train_model

    # Train
    results = train_model(
        df=df,
        model_name=model_name,
        save_dir=save_dir,
        charts_dir=charts_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        fp16=fp16,
        weight_decay=0.01,
        early_stopping_patience=early_stopping,
    )

    # Print results
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best epoch:       {results['best_epoch']}")
    print(f"  Best val loss:    {results['best_val_loss']}")
    print(f"  Best val BLEU:    {results['best_val_bleu']}")
    print(f"  Total epochs run: {results['total_epochs']}")
    print(f"  Diagnosis:        {results['diagnosis']['status']}")
    print(f"  Notes:            {results['diagnosis']['notes']}")
    print(f"  Model saved to:   {results['model_checkpoint_path']}")
    print(f"  Learning curves:  {results['learning_curve_path']}")
    print("=" * 60)

    # Save results JSON
    import json
    results_path = os.path.join(save_dir, 'training_results.json')
    # Convert non-serializable values
    results_json = {k: v for k, v in results.items()}
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
