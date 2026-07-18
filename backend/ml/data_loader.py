"""
Step 1: Load and Explore the Dataset.

Load EN-AR parallel corpus, check structure, report statistics.
Supports: CSV, TSV, JSON, HuggingFace datasets.
"""
import pandas as pd
import os
import json
import chardet
from .utils import count_tokens, count_chars, is_arabic, is_english, detect_encoding_issues


def detect_file_encoding(file_path):
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        raw = f.read(100000)  # Read first 100KB
    result = chardet.detect(raw)
    return result['encoding'], result['confidence']


def load_dataset(file_path=None, file_obj=None, file_type='csv',
                 en_column='en', ar_column='ar'):
    """
    Load an EN-AR parallel corpus.

    Args:
        file_path: Path to file on disk
        file_obj: Django uploaded file object
        file_type: 'csv', 'tsv', 'json'
        en_column: Name of the English column
        ar_column: Name of the Arabic column

    Returns:
        pd.DataFrame with columns ['en', 'ar']
    """
    if file_obj is not None:
        # Reading from Django uploaded file
        if file_type == 'csv':
            df = pd.read_csv(file_obj, encoding='utf-8')
        elif file_type == 'tsv':
            df = pd.read_csv(file_obj, sep='\t', encoding='utf-8')
        elif file_type == 'json':
            df = pd.read_json(file_obj, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    elif file_path is not None:
        # Detect encoding
        encoding, confidence = detect_file_encoding(file_path)
        if confidence < 0.5:
            encoding = 'utf-8'  # Fallback

        if file_type == 'csv':
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_type == 'tsv':
            df = pd.read_csv(file_path, sep='\t', encoding=encoding)
        elif file_type == 'json':
            df = pd.read_json(file_path, encoding=encoding)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    else:
        raise ValueError("Provide either file_path or file_obj")

    # Map columns to standard names
    target_en_col = en_column
    target_ar_col = ar_column

    # If the user specified 'en' as the English column, but there's also another column
    # like 'english' or 'en_text' in the dataset, and the 'en' column in the CSV is actually a metadata column
    # (e.g., has very low unique values like 'template_generated', or contains mostly non-English text),
    # we should check for a better candidate.
    if target_en_col == 'en' and 'en' in df.columns:
        en_col_series = df['en'].dropna()
        if len(en_col_series) > 0:
            unique_ratio = en_col_series.nunique() / len(en_col_series)
            is_metadata = en_col_series.nunique() <= 5 or unique_ratio < 0.01 or en_col_series.astype(str).str.len().mean() < 5
            if is_metadata:
                # Find a better English column candidate
                for col in df.columns:
                    if str(col).lower() in {'english', 'eng', 'en_text', 'english_sentence', 'source', 'src'} and col != 'en':
                        target_en_col = col
                        break

    if target_ar_col == 'ar' and 'ar' in df.columns:
        ar_col_series = df['ar'].dropna()
        if len(ar_col_series) > 0:
            unique_ratio = ar_col_series.nunique() / len(ar_col_series)
            is_metadata = ar_col_series.nunique() <= 5 or unique_ratio < 0.01 or ar_col_series.astype(str).str.len().mean() < 5
            if is_metadata:
                # Find a better Arabic column candidate
                for col in df.columns:
                    if str(col).lower() in {'arabic', 'ara', 'ar_text', 'arabic_sentence', 'target', 'tgt'} and col != 'ar':
                        target_ar_col = col
                        break

    if target_en_col in df.columns and target_ar_col in df.columns:
        df = df.rename(columns={target_en_col: 'en', target_ar_col: 'ar'})
    else:
        # Smart detection fallback for common column names
        en_candidates = {'en', 'english', 'eng', 'en_text', 'english_sentence', 'source', 'src'}
        ar_candidates = {'ar', 'arabic', 'ara', 'ar_text', 'arabic_sentence', 'target', 'tgt'}
        
        detected_en = None
        detected_ar = None
        
        for c in df.columns:
            c_lower = str(c).lower()
            if c_lower in en_candidates:
                # Verify it is not a metadata column
                col_series = df[c].dropna()
                if len(col_series) > 0 and col_series.nunique() > 5:
                    detected_en = c
            elif c_lower in ar_candidates:
                col_series = df[c].dropna()
                if len(col_series) > 0 and col_series.nunique() > 5:
                    detected_ar = c
                
        if detected_en and detected_ar:
            df = df.rename(columns={detected_en: 'en', detected_ar: 'ar'})
        elif 'translation' in df.columns:
            # HuggingFace format: {'en': '...', 'ar': '...'}
            df['en'] = df['translation'].apply(lambda x: x.get('en', '') if isinstance(x, dict) else '')
            df['ar'] = df['translation'].apply(lambda x: x.get('ar', '') if isinstance(x, dict) else '')
            df = df.drop(columns=['translation'])

    # Ensure we have en and ar columns
    if 'en' not in df.columns or 'ar' not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Could not find EN/AR columns. Available columns: {available}. "
            f"Tried: en_column='{en_column}', ar_column='{ar_column}'"
        )

    # Keep only en and ar (plus any metadata)
    return df


def load_from_huggingface(dataset_name='opus_books', lang_pair='en-ar', max_samples=100000):
    """
    Load dataset from HuggingFace datasets library.

    Args:
        dataset_name: HuggingFace dataset name
        lang_pair: Language pair identifier
        max_samples: Maximum number of samples to load

    Returns:
        pd.DataFrame with columns ['en', 'ar']
    """
    try:
        from datasets import load_dataset as hf_load
        dataset = hf_load(dataset_name, lang_pair, split='train')

        records = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            translation = item.get('translation', item)
            if isinstance(translation, dict):
                records.append({
                    'en': translation.get('en', ''),
                    'ar': translation.get('ar', '')
                })

        return pd.DataFrame(records)
    except Exception as e:
        raise RuntimeError(f"Failed to load from HuggingFace: {str(e)}")


def explore_dataset(df):
    """
    Step 1: Explore the loaded dataset.

    Returns a comprehensive report dictionary.
    """
    report = {}

    # Shape and basic info
    report['shape'] = {'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
    report['columns'] = list(df.columns)
    report['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Summary statistics
    report['summary'] = {}
    for col in ['en', 'ar']:
        if col in df.columns:
            non_null = int(df[col].notna().sum())
            report['summary'][col] = {
                'non_null': non_null,
                'null_count': int(df[col].isna().sum()),
                'unique': int(df[col].nunique()),
                'avg_length_chars': float(df[col].astype(str).str.len().mean()),
                'avg_length_tokens': float(df[col].astype(str).apply(count_tokens).mean()),
            }

    # Sample pairs (first 10)
    sample_pairs = []
    for _, row in df.head(10).iterrows():
        pair = {
            'en': str(row.get('en', '')),
            'ar': str(row.get('ar', '')),
        }
        sample_pairs.append(pair)
    report['sample_pairs'] = sample_pairs

    # Encoding / data type checks
    encoding_issues = []
    for col in ['en', 'ar']:
        if col in df.columns:
            # Check for non-string entries
            non_string = df[col].apply(lambda x: not isinstance(x, str) and pd.notna(x)).sum()
            if non_string > 0:
                encoding_issues.append(f"{col}: {non_string} non-string entries")

            # Sample encoding check
            for val in df[col].dropna().head(100):
                issues = detect_encoding_issues(str(val))
                if issues:
                    encoding_issues.extend([f"{col}: {issue}" for issue in issues])
                    break  # Just report first occurrence

    # Check if Arabic column actually has Arabic
    if 'ar' in df.columns:
        ar_sample = df['ar'].dropna().head(100)
        arabic_pct = ar_sample.apply(lambda x: is_arabic(str(x))).mean() * 100
        report['arabic_content_pct'] = round(arabic_pct, 1)

    # Check if English column actually has English
    if 'en' in df.columns:
        en_sample = df['en'].dropna().head(100)
        english_pct = en_sample.apply(lambda x: is_english(str(x))).mean() * 100
        report['english_content_pct'] = round(english_pct, 1)

    report['encoding_issues'] = encoding_issues if encoding_issues else ['No encoding issues detected']
    report['encoding_notes'] = (
        f"Dataset contains {report['shape']['rows']} sentence pairs. "
        f"Arabic content detected in {report.get('arabic_content_pct', 0)}% of AR column samples. "
        f"English content detected in {report.get('english_content_pct', 0)}% of EN column samples."
    )

    return report
