"""
==============================================================================
DATA LOADER MODULE — Step 1: Parallel Corpus Loading & Dataset Exploration
==============================================================================

Purpose:
  This module provides robust utility functions to:
  1. Detect file encodings (UTF-8, ISO-8859, Windows-1256, etc.) using `chardet`.
  2. Load English-Arabic (EN-AR) parallel text datasets from multiple formats:
     - CSV files
     - TSV files
     - JSON files / HuggingFace `translation` column format
     - In-memory Django uploaded files (`file_obj`)
  3. Intelligently map non-standard column names (e.g. 'source', 'target', 'english', 'arabic')
     to standard internal column names: `en` and `ar`.
  4. Perform data sanity checks, measure sentence token/char statistics, and detect encoding issues.
==============================================================================
"""

import pandas as pd
import os
import json
import chardet
from .utils import count_tokens, count_chars, is_arabic, is_english, detect_encoding_issues


def detect_file_encoding(file_path):
    """
    Detect character encoding of a file on disk using binary inspection with `chardet`.

    Args:
        file_path (str): Absolute or relative path to the target text file.

    Returns:
        tuple: (encoding_name: str, confidence_score: float)
               Example: ('utf-8', 0.99) or ('windows-1256', 0.85)
    """
    with open(file_path, 'rb') as f:
        raw = f.read(100000)  # Read initial 100 KB chunk for fast & accurate encoding detection
    result = chardet.detect(raw)
    return result['encoding'], result['confidence']


def load_dataset(file_path=None, file_obj=None, file_type='csv',
                 en_column='en', ar_column='ar'):
    """
    Load and normalize an English-Arabic parallel text corpus into a Pandas DataFrame.

    Handles both local filesystem files (`file_path`) and Django uploaded file buffers (`file_obj`).
    Automatically inspects and maps column headers to standard ['en', 'ar'] columns, ensuring
    that metadata columns (e.g., ID numbers, source flags) are not accidentally selected as text columns.

    Args:
        file_path (str, optional): Path to dataset file on local disk.
        file_obj (File, optional): File-like object (e.g., InMemoryUploadedFile from Django API).
        file_type (str): Format of the file ('csv', 'tsv', or 'json').
        en_column (str): Expected header name for the English sentence column (default: 'en').
        ar_column (str): Expected header name for the Arabic sentence column (default: 'ar').

    Returns:
        pd.DataFrame: Normalized DataFrame guaranteed to contain 'en' and 'ar' string columns.

    Raises:
        ValueError: If file parameters are missing, unsupported, or standard columns cannot be identified.
    """
    # ------------------------------------------------------------------------
    # Step A: Load Raw File into DataFrame (Django File Object vs Local File)
    # ------------------------------------------------------------------------
    if file_obj is not None:
        # Case 1: Reading directly from Django HTTP multipart file stream
        if file_type == 'csv':
            try:
                df = pd.read_csv(file_obj, encoding='utf-8', on_bad_lines='skip')
            except (UnicodeDecodeError, pd.errors.ParserError):
                file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding='utf-8-sig', on_bad_lines='skip')
        elif file_type == 'tsv':
            df = pd.read_csv(file_obj, sep='\t', encoding='utf-8', on_bad_lines='skip')
        elif file_type == 'json':
            df = pd.read_json(file_obj, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    elif file_path is not None:
        # Case 2: Reading from local disk file path with encoding auto-detection
        encoding, confidence = detect_file_encoding(file_path)
        # Fallback to UTF-8 if chardet fails or confidence score is too low (< 50%)
        if not encoding or not confidence or confidence < 0.5:
            encoding = 'utf-8'

        if file_type == 'csv':
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip')
        elif file_type == 'tsv':
            df = pd.read_csv(file_path, sep='\t', encoding=encoding, on_bad_lines='skip')
        elif file_type == 'json':
            df = pd.read_json(file_path, encoding=encoding)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    else:
        raise ValueError("Must provide either a valid 'file_path' or 'file_obj'")

    # ------------------------------------------------------------------------
    # Step B: Smart Column Detection & Disambiguation
    # ------------------------------------------------------------------------
    target_en_col = en_column
    target_ar_col = ar_column

    # Heuristic 1: If user passed 'en' as column name, but 'en' in CSV is a metadata flag (low unique count)
    # e.g., 'en' contains repetitive values like ['dataset_v1', 'dataset_v1'], check for actual text column.
    if target_en_col == 'en' and 'en' in df.columns:
        en_col_series = df['en'].dropna()
        if len(en_col_series) > 0:
            unique_ratio = en_col_series.nunique() / len(en_col_series)
            is_metadata = en_col_series.nunique() <= 5 or unique_ratio < 0.01 or en_col_series.astype(str).str.len().mean() < 5
            if is_metadata:
                # Find an alternative English sentence column header
                for col in df.columns:
                    if str(col).lower() in {'english', 'eng', 'en_text', 'english_sentence', 'source', 'src'} and col != 'en':
                        target_en_col = col
                        break

    # Heuristic 2: Apply same metadata check for Arabic column name
    if target_ar_col == 'ar' and 'ar' in df.columns:
        ar_col_series = df['ar'].dropna()
        if len(ar_col_series) > 0:
            unique_ratio = ar_col_series.nunique() / len(ar_col_series)
            is_metadata = ar_col_series.nunique() <= 5 or unique_ratio < 0.01 or ar_col_series.astype(str).str.len().mean() < 5
            if is_metadata:
                # Find an alternative Arabic sentence column header
                for col in df.columns:
                    if str(col).lower() in {'arabic', 'ara', 'ar_text', 'arabic_sentence', 'target', 'tgt'} and col != 'ar':
                        target_ar_col = col
                        break

    # Standardize selected column names to 'en' and 'ar'
    if target_en_col in df.columns and target_ar_col in df.columns:
        df = df.rename(columns={target_en_col: 'en', target_ar_col: 'ar'})
    else:
        # Fallback Heuristic: Scan header list for common synonyms
        en_candidates = {'en', 'english', 'eng', 'en_text', 'english_sentence', 'source', 'src'}
        ar_candidates = {'ar', 'arabic', 'ara', 'ar_text', 'arabic_sentence', 'target', 'tgt'}
        
        detected_en = None
        detected_ar = None
        
        for c in df.columns:
            c_lower = str(c).lower()
            if c_lower in en_candidates:
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
            # HuggingFace Dataset JSON format: {'translation': {'en': '...', 'ar': '...'}}
            df['en'] = df['translation'].apply(lambda x: x.get('en', '') if isinstance(x, dict) else '')
            df['ar'] = df['translation'].apply(lambda x: x.get('ar', '') if isinstance(x, dict) else '')
            df = df.drop(columns=['translation'])

    # ------------------------------------------------------------------------
    # Step C: Final Column Validation
    # ------------------------------------------------------------------------
    if 'en' not in df.columns or 'ar' not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Could not locate English and Arabic columns in dataset. "
            f"Found columns: {available}. Specified: en_column='{en_column}', ar_column='{ar_column}'"
        )

    return df


def load_from_huggingface(dataset_name='Helsinki-NLP/opus-100', lang_pair='ar-en', max_samples=100000):
    """
    Load English-Arabic dataset directly from HuggingFace Datasets Hub.

    Args:
        dataset_name (str): HuggingFace repository name (e.g. 'Helsinki-NLP/opus-100').
        lang_pair (str): Language code configuration (e.g. 'en-ar' or 'ar-en').
        max_samples (int): Cap maximum sentence pairs to fetch for efficient training.

    Returns:
        pd.DataFrame: DataFrame containing 'en' and 'ar' parallel sentence columns.
    """
    try:
        from datasets import load_dataset as hf_load
        try:
            dataset = hf_load(dataset_name, lang_pair, split='train')
        except Exception:
            # Try inverted language pair key if primary pair fails
            alt_pair = 'ar-en' if lang_pair == 'en-ar' else 'en-ar'
            dataset = hf_load(dataset_name, alt_pair, split='train')

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

        return pd.DataFrame(records, columns=['en', 'ar'])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch dataset from HuggingFace Hub: {str(e)}")


def explore_dataset(df):
    """
    Explore and Generate Statistical Analysis Report of the Dataset.

    Computes:
    1. Dataset shape (row count, column count, memory size in MB).
    2. Statistical summaries (min/max/avg character & token lengths, median, std, unique %).
    3. Sample translation pairs preview (first 50 rows).
    4. Quality & duplicate analysis (exact duplicates count, empty rows, language script accuracy).
    5. Database metrics & overall Dataset Health Score (0-100%).
    """
    report = {}

    total_rows = int(df.shape[0])

    # 1. Dataset Dimensions, Schema & Database Footprint
    memory_mb = float(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
    report['shape'] = {
        'rows': total_rows,
        'cols': int(df.shape[1]),
        'memory_mb': memory_mb
    }
    report['columns'] = list(df.columns)
    report['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Duplicate & Empty Rows Analysis
    exact_duplicates = int(df.duplicated(subset=['en', 'ar']).sum()) if ('en' in df.columns and 'ar' in df.columns) else 0
    duplicate_pct = float(round((exact_duplicates / total_rows * 100), 1)) if total_rows > 0 else 0.0

    empty_en = int((df['en'].isna() | (df['en'].astype(str).str.strip() == '')).sum()) if 'en' in df.columns else 0
    empty_ar = int((df['ar'].isna() | (df['ar'].astype(str).str.strip() == '')).sum()) if 'ar' in df.columns else 0

    report['database_metrics'] = {
        'total_pairs': total_rows,
        'exact_duplicates': exact_duplicates,
        'duplicate_pct': duplicate_pct,
        'empty_rows_en': empty_en,
        'empty_rows_ar': empty_ar,
        'memory_mb': memory_mb,
    }

    # 3. Column Statistical Summaries (Lengths, Tokens, Nulls, Ranges)
    report['summary'] = {}
    avg_chars_dict = {}
    avg_tokens_dict = {}

    for col in ['en', 'ar']:
        if col in df.columns:
            non_null = int(df[col].notna().sum())
            null_count = int(df[col].isna().sum())
            unique_count = int(df[col].nunique())
            unique_pct = float(round((unique_count / total_rows * 100), 1)) if total_rows > 0 else 0.0

            series_str = df[col].dropna().astype(str)
            if len(series_str) > 0:
                char_lens = series_str.str.len()
                token_lens = series_str.str.split().str.len()

                avg_chars = float(round(char_lens.mean(), 1))
                avg_tokens = float(round(token_lens.mean(), 1))
                min_chars = int(char_lens.min())
                max_chars = int(char_lens.max())
                min_tokens = int(token_lens.min())
                max_tokens = int(token_lens.max())
                median_tokens = float(round(token_lens.median(), 1))
                std_tokens = float(round(token_lens.std(), 1))
            else:
                avg_chars = avg_tokens = median_tokens = std_tokens = 0.0
                min_chars = max_chars = min_tokens = max_tokens = 0

            avg_chars_dict[col] = avg_chars
            avg_tokens_dict[col] = avg_tokens

            report['summary'][col] = {
                'non_null': non_null,
                'null_count': null_count,
                'unique': unique_count,
                'unique_pct': unique_pct,
                'avg_length_chars': avg_chars,
                'min_length_chars': min_chars,
                'max_length_chars': max_chars,
                'avg_length_tokens': avg_tokens,
                'min_length_tokens': min_tokens,
                'max_length_tokens': max_tokens,
                'median_tokens': median_tokens,
                'std_tokens': std_tokens,
            }

    # Cross-Language Length Ratios
    en_chars = avg_chars_dict.get('en', 1.0)
    ar_chars = avg_chars_dict.get('ar', 1.0)
    en_tokens = avg_tokens_dict.get('en', 1.0)
    ar_tokens = avg_tokens_dict.get('ar', 1.0)

    report['length_ratios'] = {
        'char_ratio_en_to_ar': float(round(en_chars / max(ar_chars, 0.001), 2)),
        'token_ratio_en_to_ar': float(round(en_tokens / max(ar_tokens, 0.001), 2)),
    }

    # 4. Preview 50 Sample Sentence Pairs
    sample_pairs = []
    for _, row in df.head(50).iterrows():
        pair = {
            'en': str(row.get('en', '')),
            'ar': str(row.get('ar', '')),
        }
        sample_pairs.append(pair)
    report['sample_pairs'] = sample_pairs

    # 5. Encoding Sanity Checks
    encoding_issues = []
    for col in ['en', 'ar']:
        if col in df.columns:
            non_string = df[col].apply(lambda x: not isinstance(x, str) and pd.notna(x)).sum()
            if non_string > 0:
                encoding_issues.append(f"{col}: {non_string} non-string entries detected")

            for val in df[col].dropna().head(100):
                issues = detect_encoding_issues(str(val))
                if issues:
                    encoding_issues.extend([f"{col}: {issue}" for issue in issues])
                    break

    # 6. Language Verification (Arabic & English Script Detection)
    arabic_pct = 0.0
    if 'ar' in df.columns:
        ar_sample = df['ar'].dropna().head(200)
        arabic_pct = float(round(ar_sample.apply(lambda x: is_arabic(str(x))).mean() * 100, 1)) if len(ar_sample) > 0 else 0.0
        report['arabic_content_pct'] = arabic_pct

    english_pct = 0.0
    if 'en' in df.columns:
        en_sample = df['en'].dropna().head(200)
        english_pct = float(round(en_sample.apply(lambda x: is_english(str(x))).mean() * 100, 1)) if len(en_sample) > 0 else 0.0
        report['english_content_pct'] = english_pct

    # 7. Dataset Health Score (0 - 100%)
    health_score = 100.0
    health_score -= min(30.0, duplicate_pct)
    health_score -= min(20.0, (empty_en + empty_ar) / max(total_rows * 2, 1) * 100)
    health_score -= (100.0 - arabic_pct) * 0.25
    health_score -= (100.0 - english_pct) * 0.25
    report['dataset_health_score'] = float(round(max(0.0, min(100.0, health_score)), 1))

    # 8. Final Report Notes & Summary Paragraph
    report['encoding_issues'] = encoding_issues if encoding_issues else ['No encoding issues detected']
    report['encoding_notes'] = (
        f"Dataset contains {total_rows} sentence pairs ({memory_mb} MB). "
        f"Health Score: {report['dataset_health_score']}%. "
        f"Arabic script: {arabic_pct}%, English script: {english_pct}%. "
        f"Duplicates: {exact_duplicates} ({duplicate_pct}%)."
    )

    return report

