"""
Step 4: Check Outliers and Handle Them.

Detect sentence pairs with abnormal length or length-ratio using Z-score and IQR.
Visualize with boxplots and scatter plots. Remove extreme outliers.
"""
import pandas as pd
import numpy as np
from scipy import stats
from .utils import count_tokens, count_chars


def compute_length_features(df):
    """
    Add length-based features to the dataset for outlier analysis.

    Returns:
        DataFrame with added columns: en_tokens, ar_tokens, en_chars, ar_chars, length_ratio
    """
    df = df.copy()

    # Token counts
    df['en_tokens'] = df['en'].apply(count_tokens)
    df['ar_tokens'] = df['ar'].apply(count_tokens)

    # Character counts
    df['en_chars'] = df['en'].apply(count_chars)
    df['ar_chars'] = df['ar'].apply(count_chars)

    # Length ratio (EN:AR) — guard against division by zero
    df['length_ratio'] = df.apply(
        lambda row: row['en_tokens'] / row['ar_tokens']
        if row['ar_tokens'] > 0 else float('inf'),
        axis=1
    )

    return df


def detect_outliers_zscore(df, column, threshold=3.0):
    """
    Detect outliers using Z-score method.

    Args:
        df: DataFrame with the column
        column: Column name to check
        threshold: Z-score threshold (default 3.0)

    Returns:
        Boolean mask of outliers
    """
    values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return pd.Series([False] * len(df), index=df.index)

    z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
    mask = pd.Series([False] * len(df), index=df.index)
    mask[values.index] = z_scores > threshold
    return mask


def detect_outliers_iqr(df, column, factor=1.5):
    """
    Detect outliers using IQR method.

    Args:
        df: DataFrame with the column
        column: Column name to check
        factor: IQR multiplier (default 1.5)

    Returns:
        Boolean mask of outliers
    """
    values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return pd.Series([False] * len(df), index=df.index)

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    mask = pd.Series([False] * len(df), index=df.index)
    mask[values.index] = (values < lower) | (values > upper)
    return mask


def detect_outliers(df):
    """
    Full outlier detection pipeline.

    Checks:
    - Z-score (>3) on en_tokens, ar_tokens, length_ratio
    - IQR (1.5*IQR) on en_tokens, ar_tokens, length_ratio
    - Hard length-ratio filter: outside [0.3, 3.0]

    Returns:
        (df_with_features, report_dict)
    """
    df = compute_length_features(df)

    report = {
        'total_rows': len(df),
        'length_stats': {},
        'outliers': {},
    }

    # Length statistics
    for col in ['en_tokens', 'ar_tokens', 'en_chars', 'ar_chars', 'length_ratio']:
        valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        report['length_stats'][col] = {
            'mean': round(float(valid.mean()), 2),
            'std': round(float(valid.std()), 2),
            'min': round(float(valid.min()), 2),
            'max': round(float(valid.max()), 2),
            'median': round(float(valid.median()), 2),
            'q25': round(float(valid.quantile(0.25)), 2),
            'q75': round(float(valid.quantile(0.75)), 2),
        }

    # Z-score outliers
    zscore_mask = pd.Series([False] * len(df), index=df.index)
    for col in ['en_tokens', 'ar_tokens']:
        zscore_mask |= detect_outliers_zscore(df, col)
    # Also check length ratio
    ratio_mask = df['length_ratio'].replace([np.inf, -np.inf], np.nan)
    valid_ratio = ratio_mask.dropna()
    if len(valid_ratio) > 0:
        ratio_z = detect_outliers_zscore(df, 'length_ratio')
        zscore_mask |= ratio_z

    report['outliers']['zscore'] = {
        'count': int(zscore_mask.sum()),
        'percentage': round(zscore_mask.sum() / len(df) * 100, 2),
    }

    # IQR outliers
    iqr_mask = pd.Series([False] * len(df), index=df.index)
    for col in ['en_tokens', 'ar_tokens']:
        iqr_mask |= detect_outliers_iqr(df, col)

    report['outliers']['iqr'] = {
        'count': int(iqr_mask.sum()),
        'percentage': round(iqr_mask.sum() / len(df) * 100, 2),
    }

    # Hard length-ratio filter
    ratio_outlier = (
        (df['length_ratio'] < 0.3) |
        (df['length_ratio'] > 3.0) |
        (df['length_ratio'] == float('inf'))
    )
    report['outliers']['length_ratio_filter'] = {
        'count': int(ratio_outlier.sum()),
        'percentage': round(ratio_outlier.sum() / len(df) * 100, 2),
        'threshold': '[0.3, 3.0]',
    }

    # Combined outlier mask (union of all methods)
    combined_mask = zscore_mask | iqr_mask | ratio_outlier
    report['outliers']['combined'] = {
        'count': int(combined_mask.sum()),
        'percentage': round(combined_mask.sum() / len(df) * 100, 2),
    }

    # Sample outliers
    if combined_mask.sum() > 0:
        outlier_samples = df[combined_mask].head(5)[['en', 'ar', 'en_tokens', 'ar_tokens', 'length_ratio']]
        report['outlier_samples'] = outlier_samples.to_dict(orient='records')
    else:
        report['outlier_samples'] = []

    return df, combined_mask, report


def handle_outliers(df, method='combined'):
    """
    Handle outliers: remove extreme pairs, keep within reasonable bounds.

    Returns:
        (cleaned_df, report_dict)
    """
    df_features, outlier_mask, report = detect_outliers(df)

    # Remove outliers
    df_clean = df_features[~outlier_mask].copy()

    # Also cap at 95th percentile for safety
    for col in ['en_tokens', 'ar_tokens']:
        p95 = df_clean[col].quantile(0.95)
        over_cap = (df_clean[col] > p95).sum()
        df_clean = df_clean[df_clean[col] <= p95]

    # Drop length feature columns (keep only en, ar and any original columns)
    feature_cols = ['en_tokens', 'ar_tokens', 'en_chars', 'ar_chars', 'length_ratio']
    df_final = df_clean.drop(columns=[c for c in feature_cols if c in df_clean.columns])

    removed = len(df) - len(df_final)
    report['removed_count'] = int(removed)
    report['remaining_rows'] = len(df_final)
    report['outlier_pct'] = round(removed / len(df) * 100, 2) if len(df) > 0 else 0
    report['action_taken'] = (
        f"Removed {removed} outlier pairs using Z-score + IQR + length-ratio filters. "
        f"Additionally capped at 95th percentile. "
        f"Dataset reduced from {len(df)} to {len(df_final)} rows."
    )

    return df_final.reset_index(drop=True), report
