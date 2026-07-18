"""
Step 6: Check Imbalance and Handle It.

For translation datasets without domain labels, this step creates a synthetic
'length_category' label to demonstrate the technique. If domain labels exist,
it checks and handles their imbalance using SMOTE/undersampling.
"""
import pandas as pd
import numpy as np
from collections import Counter
from .utils import count_tokens, categorize_length


def check_imbalance(df, label_column=None):
    """
    Check class distribution for imbalance.

    If no label_column is provided, creates a synthetic 'length_category'
    to demonstrate the technique.

    Args:
        df: Input DataFrame
        label_column: Column name with class labels (optional)

    Returns:
        dict with imbalance analysis
    """
    report = {
        'total_rows': len(df),
        'is_applicable': False,
        'label_column': None,
    }

    # If no explicit label column, create synthetic length categories
    if label_column is None or label_column not in df.columns:
        df = df.copy()
        df['en_tokens'] = df['en'].apply(count_tokens)
        df['length_category'] = df['en_tokens'].apply(categorize_length)
        label_column = 'length_category'
        report['note'] = (
            'No domain labels found. Created synthetic "length_category" '
            '(very_short/short/medium/long/very_long) based on English sentence '
            'token count to demonstrate imbalance analysis technique.'
        )
        report['is_synthetic'] = True
    else:
        report['is_synthetic'] = False

    report['is_applicable'] = True
    report['label_column'] = label_column

    # Class distribution
    class_counts = df[label_column].value_counts()
    total = class_counts.sum()
    distribution = {}
    for cls, count in class_counts.items():
        distribution[str(cls)] = {
            'count': int(count),
            'percentage': round(count / total * 100, 2),
        }
    report['class_distribution'] = distribution

    # Check if imbalanced (any class < 30% when there are 2 classes,
    # or ratio of largest to smallest > 2.33 for multi-class)
    max_pct = class_counts.max() / total * 100
    min_pct = class_counts.min() / total * 100
    ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')

    report['is_imbalanced'] = ratio > 2.33
    report['imbalance_ratio'] = round(ratio, 2)
    report['majority_class'] = str(class_counts.idxmax())
    report['minority_class'] = str(class_counts.idxmin())
    report['majority_pct'] = round(max_pct, 2)
    report['minority_pct'] = round(min_pct, 2)

    return df, report


def handle_imbalance(df, label_column=None, strategy='undersample'):
    """
    Handle class imbalance.

    For translation datasets, this is typically N/A for the core task.
    This function demonstrates the technique on length categories or domain labels.

    Strategies:
    - 'undersample': Random undersampling of majority classes
    - 'oversample': Random oversampling of minority classes (no SMOTE for text)
    - 'none': Document as N/A

    Returns:
        (balanced_df, report_dict)
    """
    df_with_labels, report = check_imbalance(df, label_column)
    actual_label = report['label_column']

    report['strategy'] = strategy
    report['distribution_before'] = report['class_distribution'].copy()

    if not report['is_imbalanced']:
        report['action_taken'] = 'Dataset is balanced. No action needed.'
        report['distribution_after'] = report['class_distribution']
        # Remove synthetic columns before returning
        cols_to_drop = [c for c in ['en_tokens', 'length_category'] if c in df_with_labels.columns and c != actual_label]
        df_clean = df_with_labels.drop(columns=cols_to_drop, errors='ignore')
        return df_clean, report

    if strategy == 'none':
        report['action_taken'] = (
            'Imbalance detected but handling marked as N/A. '
            'For pure sequence-to-sequence translation, class balancing is not applicable. '
            'This applies only to auxiliary labeled sub-tasks.'
        )
        report['distribution_after'] = report['class_distribution']
        cols_to_drop = [c for c in ['en_tokens', 'length_category'] if c in df_with_labels.columns]
        df_clean = df_with_labels.drop(columns=cols_to_drop, errors='ignore')
        return df_clean, report

    elif strategy == 'undersample':
        # Random undersampling to match minority class count
        class_counts = df_with_labels[actual_label].value_counts()
        min_count = class_counts.min()

        balanced_dfs = []
        for cls in class_counts.index:
            cls_df = df_with_labels[df_with_labels[actual_label] == cls]
            sampled = cls_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled)

        df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

        report['action_taken'] = (
            f"Applied random undersampling. Each class now has {min_count} samples. "
            f"Total dataset: {len(df_balanced)} rows (from {len(df)})."
        )

    elif strategy == 'oversample':
        # Random oversampling to match majority class count
        class_counts = df_with_labels[actual_label].value_counts()
        max_count = class_counts.max()

        balanced_dfs = []
        for cls in class_counts.index:
            cls_df = df_with_labels[df_with_labels[actual_label] == cls]
            if len(cls_df) < max_count:
                sampled = cls_df.sample(n=max_count, replace=True, random_state=42)
            else:
                sampled = cls_df
            balanced_dfs.append(sampled)

        df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

        report['action_taken'] = (
            f"Applied random oversampling. Each class now has {max_count} samples. "
            f"Total dataset: {len(df_balanced)} rows (from {len(df)})."
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # After distribution
    after_counts = df_balanced[actual_label].value_counts()
    after_total = after_counts.sum()
    after_dist = {}
    for cls, count in after_counts.items():
        after_dist[str(cls)] = {
            'count': int(count),
            'percentage': round(count / after_total * 100, 2),
        }
    report['distribution_after'] = after_dist

    # Drop synthetic columns
    cols_to_drop = [c for c in ['en_tokens', 'length_category'] if c in df_balanced.columns]
    df_balanced = df_balanced.drop(columns=cols_to_drop, errors='ignore')

    return df_balanced, report
