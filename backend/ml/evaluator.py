"""
Step 8: Evaluate Model Performance Using Metrics.

MT metrics: BLEU, chrF, TER (via sacrebleu).
Baseline comparison: pretrained vs fine-tuned.
Qualitative examples: source, reference, baseline output, model output.
"""
import os
import time
import pandas as pd
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_model(
    test_df,
    model_path,
    baseline_model_name='Helsinki-NLP/opus-mt-en-ar',
    charts_dir='./charts',
    max_length=128,
    num_examples=10,
):
    """
    Evaluate the fine-tuned model and compare to baseline.

    Args:
        test_df: DataFrame with 'en' and 'ar' columns (test set)
        model_path: Path to fine-tuned model checkpoint
        baseline_model_name: HuggingFace model name for baseline
        charts_dir: Directory to save evaluation charts
        max_length: Maximum sequence length for generation
        num_examples: Number of qualitative examples to include

    Returns:
        dict with evaluation results
    """
    import sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(charts_dir, exist_ok=True)

    sources = test_df['en'].tolist()
    references = test_df['ar'].tolist()

    # ---- BASELINE EVALUATION ----
    print("Evaluating baseline model...")
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model_name).to(device)

    baseline_translations = _batch_translate(
        baseline_model, baseline_tokenizer, sources, device, max_length
    )
    baseline_metrics = _compute_metrics(baseline_translations, references)

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- FINE-TUNED MODEL EVALUATION ----
    print("Evaluating fine-tuned model...")
    ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    ft_translations = _batch_translate(
        ft_model, ft_tokenizer, sources, device, max_length
    )
    ft_metrics = _compute_metrics(ft_translations, references)

    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- QUALITATIVE EXAMPLES ----
    example_indices = np.random.choice(len(sources), min(num_examples, len(sources)), replace=False)
    examples = []
    for idx in example_indices:
        examples.append({
            'source': sources[idx],
            'reference': references[idx],
            'baseline_output': baseline_translations[idx],
            'model_output': ft_translations[idx],
        })

    # ---- COMPARISON CHART ----
    chart_path = _plot_comparison(baseline_metrics, ft_metrics, charts_dir)

    # ---- EXIT CONDITION CHECK ----
    meets_target = (
        ft_metrics['bleu'] >= 25.0 and
        ft_metrics['bleu'] - baseline_metrics['bleu'] > 0  # Improvement over baseline
    )

    target_notes = _generate_target_notes(baseline_metrics, ft_metrics, meets_target)

    results = {
        'baseline': {
            'model_name': baseline_model_name,
            'bleu': baseline_metrics['bleu'],
            'chrf': baseline_metrics['chrf'],
            'ter': baseline_metrics['ter'],
        },
        'fine_tuned': {
            'model_path': model_path,
            'bleu': ft_metrics['bleu'],
            'chrf': ft_metrics['chrf'],
            'ter': ft_metrics['ter'],
        },
        'improvement': {
            'bleu_delta': round(ft_metrics['bleu'] - baseline_metrics['bleu'], 2),
            'chrf_delta': round(ft_metrics['chrf'] - baseline_metrics['chrf'], 2),
            'ter_delta': round(baseline_metrics['ter'] - ft_metrics['ter'], 2),  # Lower is better
        },
        'example_translations': examples,
        'comparison_chart_path': chart_path,
        'test_set_size': len(sources),
        'meets_target': meets_target,
        'target_notes': target_notes,
    }

    return results


def _batch_translate(model, tokenizer, texts, device, max_length=128, batch_size=16):
    """Translate a list of texts in batches."""
    model.eval()
    translations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs, max_length=max_length, num_beams=4
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def _compute_metrics(predictions, references):
    """Compute BLEU, chrF, and TER using sacrebleu."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    ter = sacrebleu.corpus_ter(predictions, [references])

    return {
        'bleu': round(bleu.score, 2),
        'chrf': round(chrf.score, 2),
        'ter': round(ter.score, 2),
    }


def _plot_comparison(baseline_metrics, ft_metrics, charts_dir):
    """Plot baseline vs fine-tuned metrics comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['BLEU ↑', 'chrF ↑', 'TER ↓']
    baseline_vals = [baseline_metrics['bleu'], baseline_metrics['chrf'], baseline_metrics['ter']]
    ft_vals = [ft_metrics['bleu'], ft_metrics['chrf'], ft_metrics['ter']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Pretrained)',
                   color='#e74c3c', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, ft_vals, width, label='Fine-tuned',
                   color='#2ecc71', alpha=0.8, edgecolor='white')

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Baseline vs Fine-tuned Model Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)

    # Add value labels
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(charts_dir, 'model_comparison.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _generate_target_notes(baseline_metrics, ft_metrics, meets_target):
    """Generate human-readable notes about target achievement."""
    notes = []

    # BLEU check
    if ft_metrics['bleu'] >= 25:
        notes.append(f"✅ BLEU target met: {ft_metrics['bleu']:.2f} >= 25.0")
    else:
        notes.append(f"❌ BLEU target NOT met: {ft_metrics['bleu']:.2f} < 25.0")
        notes.append("   → Consider: more training data, more epochs, or data cleaning improvements")

    # Improvement check
    bleu_delta = ft_metrics['bleu'] - baseline_metrics['bleu']
    if bleu_delta > 0:
        notes.append(f"✅ Improved over baseline by +{bleu_delta:.2f} BLEU")
    else:
        notes.append(f"❌ No improvement over baseline (delta: {bleu_delta:.2f})")
        notes.append("   → Consider: loop back to Step 4 (data cleaning) or Step 7 (hyperparameters)")

    # Overall
    if meets_target:
        notes.append("\n🎯 EXIT CONDITION MET — Model is ready for deployment.")
    else:
        notes.append("\n🔄 EXIT CONDITION NOT MET — Loop back for iteration:")
        if ft_metrics['bleu'] < 15:
            notes.append("   → Return to Step 7: increase epochs, unfreeze layers, try larger model")
        else:
            notes.append("   → Return to Step 3/4: improve data cleaning, remove more outliers")

    return '\n'.join(notes)
