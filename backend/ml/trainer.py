"""
Step 7: Address Overfitting and Underfitting.

Fine-tune Helsinki-NLP/opus-mt-en-ar with:
- fp16 mixed precision (6GB VRAM constraint)
- Small batch size (4) with gradient accumulation (8) = effective batch 32
- Early stopping, weight decay, dropout
- Learning curve plotting
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TranslationDataset(TorchDataset):
    """PyTorch Dataset for EN-AR sentence pairs."""

    def __init__(self, en_texts, ar_texts, tokenizer, max_length=128):
        self.en_texts = en_texts
        self.ar_texts = ar_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, idx):
        en = str(self.en_texts[idx])
        ar = str(self.ar_texts[idx])

        # Tokenize source (English)
        source = self.tokenizer(
            en, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        # Tokenize target (Arabic)
        target = self.tokenizer(
            text_target=ar, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        input_ids = source['input_ids'].squeeze()
        attention_mask = source['attention_mask'].squeeze()
        labels = target['input_ids'].squeeze()

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split dataset into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)

    if n < 3:
        # Too small to split, duplicate or use all for train/val/test
        return df, df, df

    # For small datasets, guarantee at least 1 sample in validation and test sets
    if n < 10:
        val_df = df.iloc[[0]].reset_index(drop=True)
        test_df = df.iloc[[1]].reset_index(drop=True)
        train_df = df.iloc[2:].reset_index(drop=True)
        return train_df, val_df, test_df

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


def compute_bleu_score(predictions, references):
    """Compute BLEU score using sacrebleu."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def train_model(
    df,
    model_name='Helsinki-NLP/opus-mt-en-ar',
    save_dir='./models',
    charts_dir='./charts',
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    max_epochs=10,
    fp16=True,
    weight_decay=0.01,
    early_stopping_patience=3,
    max_length=128,
    progress_callback=None,
):
    """
    Fine-tune the translation model.

    Args:
        df: DataFrame with 'en' and 'ar' columns
        model_name: HuggingFace model name
        save_dir: Directory to save model checkpoints
        charts_dir: Directory to save learning curves
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        fp16: Use mixed precision training
        weight_decay: Weight decay for AdamW
        early_stopping_patience: Stop if val loss doesn't improve for N epochs
        max_length: Maximum sequence length
        progress_callback: Optional callback(epoch_data_dict) for progress updates

    Returns:
        dict with training results
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # Split data
    train_df, val_df, test_df = split_dataset(df)
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    train_dataset = TranslationDataset(
        train_df['en'].tolist(), train_df['ar'].tolist(), tokenizer, max_length
    )
    val_dataset = TranslationDataset(
        val_df['en'].tolist(), val_df['ar'].tolist(), tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if fp16 and device.type == 'cuda' else None

    # Training loop
    epoch_data = []
    best_val_loss = float('inf')
    best_val_bleu = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'best_model')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    for epoch in range(max_epochs):
        start_time = time.time()

        # ---- TRAINING ----
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            train_losses.append(loss.item() * gradient_accumulation_steps)

        avg_train_loss = np.mean(train_losses)

        # ---- VALIDATION ----
        model.eval()
        val_losses = []
        val_predictions = []
        val_references = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_losses.append(outputs.loss.item())

                # Generate translations for BLEU
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=4,
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                val_predictions.extend(decoded)

                # Decode reference labels
                label_ids = labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                ref_decoded = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                val_references.extend(ref_decoded)

        avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else 0.0

        # Compute BLEU
        if val_predictions and val_references:
            try:
                val_bleu = compute_bleu_score(val_predictions, val_references)
            except Exception:
                val_bleu = 0.0
        else:
            val_bleu = 0.0

        elapsed = time.time() - start_time

        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': round(float(avg_train_loss), 4),
            'val_loss': round(float(avg_val_loss), 4),
            'val_bleu': round(float(val_bleu), 2),
            'elapsed_seconds': round(elapsed, 1),
            'loss_gap': round(float(avg_val_loss - avg_train_loss), 4),
        }
        epoch_data.append(epoch_info)

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val BLEU: {val_bleu:.2f} | "
            f"Gap: {avg_val_loss - avg_train_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if progress_callback:
            progress_callback(epoch_info)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_bleu = val_bleu
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  -> Saved best model (BLEU: {val_bleu:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

    # ---- DIAGNOSIS ----
    diagnosis = diagnose_fit(epoch_data)

    # ---- LEARNING CURVES ----
    curve_path = plot_learning_curves(epoch_data, charts_dir)

    # ---- RESULTS ----
    results = {
        'model_name': model_name,
        'best_epoch': best_epoch,
        'best_val_loss': round(float(best_val_loss), 4),
        'best_val_bleu': round(float(best_val_bleu), 2),
        'total_epochs': len(epoch_data),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'epoch_data': epoch_data,
        'diagnosis': diagnosis,
        'model_checkpoint_path': best_model_path,
        'learning_curve_path': curve_path,
        'hyperparameters': {
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'learning_rate': learning_rate,
            'max_epochs': max_epochs,
            'fp16': fp16,
            'weight_decay': weight_decay,
            'early_stopping_patience': early_stopping_patience,
            'max_length': max_length,
        },
    }

    # Save test set for evaluation
    test_path = os.path.join(save_dir, 'test_set.csv')
    test_df.to_csv(test_path, index=False)
    results['test_set_path'] = test_path

    return results


def diagnose_fit(epoch_data):
    """
    Diagnose overfitting/underfitting from training curves.

    Returns:
        dict with diagnosis and notes
    """
    if len(epoch_data) < 2:
        return {'status': 'unknown', 'notes': 'Not enough epochs to diagnose.'}

    final = epoch_data[-1]
    train_loss = final['train_loss']
    val_loss = final['val_loss']
    val_bleu = final['val_bleu']
    gap = val_loss - train_loss

    # Check if training loss is still decreasing
    losses = [e['train_loss'] for e in epoch_data]
    loss_decreasing = losses[-1] < losses[0]

    # Check if validation loss diverges from training
    val_losses = [e['val_loss'] for e in epoch_data]
    val_increasing = len(val_losses) > 3 and val_losses[-1] > val_losses[-3]

    if gap > 0.5 and val_increasing:
        return {
            'status': 'overfitting',
            'notes': (
                f'Train-val loss gap = {gap:.4f} and val loss is increasing. '
                f'Recommendations: increase dropout, add more weight decay, '
                f'reduce model complexity, or use data augmentation (back-translation).'
            ),
        }
    elif val_bleu < 10 and train_loss > 2.0:
        return {
            'status': 'underfitting',
            'notes': (
                f'BLEU = {val_bleu:.2f} and train loss = {train_loss:.4f} still high. '
                f'Recommendations: train for more epochs, unfreeze more layers, '
                f'increase learning rate, or use a larger model if VRAM allows.'
            ),
        }
    else:
        return {
            'status': 'well_fit',
            'notes': (
                f'Model appears well-fit. Train-val gap = {gap:.4f}, '
                f'BLEU = {val_bleu:.2f}. Training and validation curves converge well.'
            ),
        }


def plot_learning_curves(epoch_data, charts_dir):
    """Plot training and validation loss + BLEU over epochs."""
    epochs = [e['epoch'] for e in epoch_data]
    train_losses = [e['train_loss'] for e in epoch_data]
    val_losses = [e['val_loss'] for e in epoch_data]
    val_bleus = [e['val_bleu'] for e in epoch_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax1.fill_between(epochs, train_losses, val_losses, alpha=0.1, color='red')
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_title('Training vs Validation Loss', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # BLEU curve
    ax2.plot(epochs, val_bleus, 'g-o', label='Val BLEU', linewidth=2)
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Target BLEU (25)')
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('BLEU Score', fontsize=13)
    ax2.set_title('Validation BLEU Score', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(charts_dir, 'learning_curves.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
