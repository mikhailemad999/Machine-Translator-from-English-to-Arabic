"""
Step 2: Check for and Handle Duplicates.

Detect duplicates on full pair (EN+AR), EN-only, AR-only columns.
Report counts/percentages and drop duplicates.
"""
import pandas as pd


def detect_duplicates(df):
    """
    Detect duplicates in the dataset.

    Returns:
        dict with duplicate analysis results
    """
    report = {
        'total_rows': len(df),
    }

    # Duplicates on full pair (EN + AR)
    dup_full = df.duplicated(subset=['en', 'ar'], keep=False)
    dup_full_count = dup_full.sum()
    report['duplicates_full_pair'] = {
        'count': int(dup_full_count),
        'percentage': round(dup_full_count / len(df) * 100, 2) if len(df) > 0 else 0,
        'description': 'Exact same EN+AR sentence pair appears more than once',
    }

    # Duplicates on EN only
    dup_en = df.duplicated(subset=['en'], keep=False)
    dup_en_count = dup_en.sum()
    report['duplicates_en_only'] = {
        'count': int(dup_en_count),
        'percentage': round(dup_en_count / len(df) * 100, 2) if len(df) > 0 else 0,
        'description': 'Same English sentence with different Arabic translations',
    }

    # Duplicates on AR only
    dup_ar = df.duplicated(subset=['ar'], keep=False)
    dup_ar_count = dup_ar.sum()
    report['duplicates_ar_only'] = {
        'count': int(dup_ar_count),
        'percentage': round(dup_ar_count / len(df) * 100, 2) if len(df) > 0 else 0,
        'description': 'Same Arabic sentence with different English sources',
    }

    # Sample duplicates for inspection
    if dup_full_count > 0:
        dup_samples = df[df.duplicated(subset=['en', 'ar'], keep=False)].head(5)
        report['duplicate_samples'] = dup_samples[['en', 'ar']].to_dict(orient='records')
    else:
        report['duplicate_samples'] = []

    return report


def remove_duplicates(df, strategy='full_pair', keep='first'):
    """
    Remove duplicates from the dataset.

    Args:
        df: Input DataFrame
        strategy: 'full_pair' (remove if EN+AR identical),
                  'en_only' (remove if EN identical),
                  'ar_only' (remove if AR identical)
        keep: 'first' or 'last'

    Returns:
        Cleaned DataFrame, number of removed rows
    """
    original_count = len(df)

    if strategy == 'full_pair':
        df_clean = df.drop_duplicates(subset=['en', 'ar'], keep=keep)
    elif strategy == 'en_only':
        df_clean = df.drop_duplicates(subset=['en'], keep=keep)
    elif strategy == 'ar_only':
        df_clean = df.drop_duplicates(subset=['ar'], keep=keep)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    removed = original_count - len(df_clean)

    return df_clean.reset_index(drop=True), removed


def handle_duplicates(df):
    """
    Full duplicate handling pipeline.
    Detects, reports, and removes duplicates.

    Returns:
        (cleaned_df, report_dict)
    """
    # Detect
    report = detect_duplicates(df)

    # Remove full pair duplicates (the most conservative approach)
    df_clean, removed = remove_duplicates(df, strategy='full_pair', keep='first')

    report['removed_count'] = int(removed)
    report['remaining_rows'] = len(df_clean)
    report['action_taken'] = (
        f"Removed {removed} exact duplicate pairs (kept first occurrence). "
        f"Dataset reduced from {report['total_rows']} to {len(df_clean)} rows."
    )

    return df_clean, report
