"""
Step 3: Apply Techniques to Handle Missing Values.

Detect missing/empty/whitespace-only translations.
Text fields are dropped (not imputed) — imputation is only for metadata.
"""
import pandas as pd
import numpy as np


def detect_missing_values(df):
    """
    Detect missing values per column.

    Checks for: NaN, None, empty strings, whitespace-only strings.

    Returns:
        dict with missing value analysis
    """
    report = {
        'total_rows': len(df),
        'columns': {},
    }

    for col in df.columns:
        # Standard null check
        null_count = int(df[col].isna().sum())

        # Empty string check
        if df[col].dtype == object:
            empty_count = int((df[col] == '').sum())
            whitespace_count = int(df[col].astype(str).str.strip().eq('').sum()) - null_count
        else:
            empty_count = 0
            whitespace_count = 0

        total_missing = null_count + empty_count + max(0, whitespace_count)
        pct = round(total_missing / len(df) * 100, 2) if len(df) > 0 else 0

        report['columns'][col] = {
            'null_count': null_count,
            'empty_string_count': empty_count,
            'whitespace_only_count': max(0, whitespace_count),
            'total_missing': total_missing,
            'missing_percentage': pct,
        }

    # Overall missing summary
    en_missing = report['columns'].get('en', {}).get('total_missing', 0)
    ar_missing = report['columns'].get('ar', {}).get('total_missing', 0)
    report['summary'] = {
        'en_missing': en_missing,
        'ar_missing': ar_missing,
        'en_missing_pct': round(en_missing / len(df) * 100, 2) if len(df) > 0 else 0,
        'ar_missing_pct': round(ar_missing / len(df) * 100, 2) if len(df) > 0 else 0,
        'recommendation': _get_recommendation(en_missing, ar_missing, len(df)),
    }

    return report


def _get_recommendation(en_missing, ar_missing, total):
    """Generate handling recommendation based on missing percentage."""
    if total == 0:
        return 'No data to analyze'

    en_pct = en_missing / total * 100
    ar_pct = ar_missing / total * 100
    max_pct = max(en_pct, ar_pct)

    if max_pct == 0:
        return 'No missing values detected. No action needed.'
    elif max_pct < 5:
        return (
            f'Missing values are below 5% threshold ({max_pct:.1f}%). '
            'Recommended: Drop rows with missing text values (safe deletion).'
        )
    elif max_pct < 20:
        return (
            f'Missing values at {max_pct:.1f}%. '
            'Recommended: Drop missing text rows. For metadata columns, '
            'consider imputation with mode/median.'
        )
    else:
        return (
            f'High missing rate ({max_pct:.1f}%). '
            'Investigate data source quality. Drop rows with missing EN/AR text. '
            'Consider if dataset is viable.'
        )


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.

    For translation data:
    - Text fields (en, ar) are ALWAYS dropped, never imputed
    - Metadata fields can be imputed with mode/median

    Args:
        df: Input DataFrame
        strategy: 'drop' (default) — drop rows with any missing EN/AR

    Returns:
        (cleaned_df, report_dict)
    """
    report = detect_missing_values(df)
    original_count = len(df)

    # Convert empty strings and whitespace to NaN for uniform handling
    df_clean = df.copy()
    for col in ['en', 'ar']:
        if col in df_clean.columns:
            # Replace empty strings and whitespace-only with NaN
            df_clean[col] = df_clean[col].replace('', np.nan)
            df_clean[col] = df_clean[col].apply(
                lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x
            )

    # Drop rows where EN or AR is missing
    df_clean = df_clean.dropna(subset=['en', 'ar'])

    # Ensure string type
    df_clean['en'] = df_clean['en'].astype(str)
    df_clean['ar'] = df_clean['ar'].astype(str)

    removed = original_count - len(df_clean)

    report['removed_count'] = int(removed)
    report['remaining_rows'] = len(df_clean)
    report['strategy_used'] = strategy
    report['action_taken'] = (
        f"Dropped {removed} rows with missing/empty EN or AR text. "
        f"Dataset reduced from {original_count} to {len(df_clean)} rows."
    )

    return df_clean.reset_index(drop=True), report
