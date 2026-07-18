"""
Step 5: Make Visualizations to Check Problems, Distribution, and Balance.

Generate 7+ charts for the EDA dashboard:
1. Histogram: EN sentence length distribution
2. Histogram: AR sentence length distribution
3. KDE: Overlaid EN vs AR length distributions
4. Scatter: EN length vs AR length (with regression line)
5. Boxplot: EN and AR lengths side by side
6. Heatmap: Correlation of numeric features
7. Bar chart: Sentence length bucket distribution
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from .utils import count_tokens, categorize_length


# Set style
sns.set_theme(style='darkgrid', palette='deep')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12


def prepare_viz_data(df):
    """Add visualization features to the DataFrame."""
    df = df.copy()
    df['en_tokens'] = df['en'].apply(count_tokens)
    df['ar_tokens'] = df['ar'].apply(count_tokens)
    df['en_chars'] = df['en'].str.len()
    df['ar_chars'] = df['ar'].str.len()
    df['length_ratio'] = df.apply(
        lambda r: r['en_tokens'] / r['ar_tokens'] if r['ar_tokens'] > 0 else 0,
        axis=1
    )
    df['en_length_category'] = df['en_tokens'].apply(categorize_length)
    df['ar_length_category'] = df['ar_tokens'].apply(categorize_length)
    return df


def chart_1_en_length_histogram(df, save_dir):
    """Chart 1: English sentence length distribution (histogram)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['en_tokens'], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Number of Tokens', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Distribution of English Sentence Lengths', fontsize=15, fontweight='bold')
    ax.axvline(df['en_tokens'].mean(), color='red', linestyle='--', label=f"Mean: {df['en_tokens'].mean():.1f}")
    ax.axvline(df['en_tokens'].median(), color='orange', linestyle='--', label=f"Median: {df['en_tokens'].median():.1f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_1_en_length_hist.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_2_ar_length_histogram(df, save_dir):
    """Chart 2: Arabic sentence length distribution (histogram)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['ar_tokens'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Number of Tokens', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Distribution of Arabic Sentence Lengths', fontsize=15, fontweight='bold')
    ax.axvline(df['ar_tokens'].mean(), color='red', linestyle='--', label=f"Mean: {df['ar_tokens'].mean():.1f}")
    ax.axvline(df['ar_tokens'].median(), color='orange', linestyle='--', label=f"Median: {df['ar_tokens'].median():.1f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_2_ar_length_hist.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_3_kde_overlay(df, save_dir):
    """Chart 3: KDE overlay of EN vs AR length distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df['en_tokens'], ax=ax, label='English', color='#3498db', fill=True, alpha=0.3)
    sns.kdeplot(df['ar_tokens'], ax=ax, label='Arabic', color='#2ecc71', fill=True, alpha=0.3)
    ax.set_xlabel('Number of Tokens', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('EN vs AR Sentence Length Distributions (KDE)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_3_kde_overlay.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_4_scatter_lengths(df, save_dir):
    """Chart 4: Scatter plot of EN length vs AR length."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample if dataset is large
    plot_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    ax.scatter(plot_df['en_tokens'], plot_df['ar_tokens'],
               alpha=0.3, s=10, color='#9b59b6')

    # Add regression line
    z = np.polyfit(plot_df['en_tokens'], plot_df['ar_tokens'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(plot_df['en_tokens'].min(), plot_df['en_tokens'].max(), 100)
    ax.plot(x_range, p(x_range), 'r--', linewidth=2,
            label=f'Trend: AR = {z[0]:.2f}×EN + {z[1]:.2f}')

    # Diagonal (equal length)
    max_val = max(plot_df['en_tokens'].max(), plot_df['ar_tokens'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal length')

    ax.set_xlabel('English Tokens', fontsize=13)
    ax.set_ylabel('Arabic Tokens', fontsize=13)
    ax.set_title('English vs Arabic Sentence Lengths', fontsize=15, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_4_scatter_lengths.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_5_boxplot_lengths(df, save_dir):
    """Chart 5: Side-by-side boxplots of EN and AR lengths."""
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = pd.DataFrame({
        'English': df['en_tokens'],
        'Arabic': df['ar_tokens'],
    })

    bp = ax.boxplot([box_data['English'], box_data['Arabic']],
                    labels=['English', 'Arabic'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))

    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#2ecc71')

    ax.set_ylabel('Number of Tokens', fontsize=13)
    ax.set_title('Sentence Length Distribution (Boxplot)', fontsize=15, fontweight='bold')

    # Add stats annotation
    for i, col in enumerate(['English', 'Arabic']):
        vals = box_data[col]
        ax.annotate(f'μ={vals.mean():.1f}\nσ={vals.std():.1f}',
                    xy=(i + 1, vals.max()),
                    fontsize=10, ha='center', va='bottom')

    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_5_boxplot_lengths.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_6_correlation_heatmap(df, save_dir):
    """Chart 6: Correlation heatmap of numeric features."""
    fig, ax = plt.subplots(figsize=(8, 6))

    numeric_cols = ['en_tokens', 'ar_tokens', 'en_chars', 'ar_chars', 'length_ratio']
    corr_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
    corr_matrix = corr_df.corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, square=True, ax=ax,
                xticklabels=['EN Tokens', 'AR Tokens', 'EN Chars', 'AR Chars', 'Ratio'],
                yticklabels=['EN Tokens', 'AR Tokens', 'EN Chars', 'AR Chars', 'Ratio'])

    ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_6_correlation_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_7_length_category_distribution(df, save_dir):
    """Chart 7: Bar chart of sentence length categories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    categories_order = ['very_short', 'short', 'medium', 'long', 'very_long']
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

    # EN categories
    en_counts = df['en_length_category'].value_counts().reindex(categories_order, fill_value=0)
    axes[0].bar(en_counts.index, en_counts.values, color=colors, edgecolor='white')
    axes[0].set_title('English Sentence Length Categories', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # AR categories
    ar_counts = df['ar_length_category'].value_counts().reindex(categories_order, fill_value=0)
    axes[1].bar(ar_counts.index, ar_counts.values, color=colors, edgecolor='white')
    axes[1].set_title('Arabic Sentence Length Categories', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(save_dir, 'chart_7_length_categories.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_charts(df, save_dir):
    """
    Generate all 7 visualization charts.

    Args:
        df: DataFrame with 'en' and 'ar' columns
        save_dir: Directory to save chart images

    Returns:
        List of chart file paths
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data with features
    viz_df = prepare_viz_data(df)

    chart_paths = []
    chart_generators = [
        ('English Length Histogram', chart_1_en_length_histogram),
        ('Arabic Length Histogram', chart_2_ar_length_histogram),
        ('KDE Overlay EN vs AR', chart_3_kde_overlay),
        ('Scatter EN vs AR Lengths', chart_4_scatter_lengths),
        ('Boxplot Lengths', chart_5_boxplot_lengths),
        ('Correlation Heatmap', chart_6_correlation_heatmap),
        ('Length Category Distribution', chart_7_length_category_distribution),
    ]

    for name, generator in chart_generators:
        try:
            path = generator(viz_df, save_dir)
            chart_paths.append({
                'name': name,
                'path': path,
                'filename': os.path.basename(path),
            })
        except Exception as e:
            chart_paths.append({
                'name': name,
                'path': None,
                'error': str(e),
            })

    return chart_paths
