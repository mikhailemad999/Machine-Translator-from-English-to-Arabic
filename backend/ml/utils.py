"""
ML Utilities — Arabic text normalization, tokenization helpers.
"""
import re
import string


def normalize_arabic(text):
    """
    Normalize Arabic text for NLP preprocessing.
    - Remove diacritics (tashkeel)
    - Normalize alef variants to bare alef
    - Normalize taa marbuta to haa
    - Remove tatweel (kashida)
    - Normalize digits
    """
    if not isinstance(text, str):
        return ''

    # Remove diacritics (tashkeel)
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')
    text = arabic_diacritics.sub('', text)

    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)

    # Normalize taa marbuta
    text = re.sub(r'ة', 'ه', text)

    # Remove tatweel
    text = re.sub(r'ـ', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_english(text):
    """
    Normalize English text.
    - Lowercase
    - Normalize whitespace
    - Strip leading/trailing spaces
    """
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def count_tokens(text):
    """Simple whitespace tokenization for length counting."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def count_chars(text):
    """Count non-whitespace characters."""
    if not isinstance(text, str):
        return 0
    return len(text.replace(' ', ''))


def is_arabic(text):
    """Check if text contains Arabic characters."""
    if not isinstance(text, str):
        return False
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))


def is_english(text):
    """Check if text contains English/Latin characters."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[a-zA-Z]', text))


def detect_encoding_issues(text):
    """Detect common encoding issues in text."""
    issues = []
    if not isinstance(text, str):
        return ['Not a string']

    # Check for replacement characters
    if '\ufffd' in text:
        issues.append('Contains Unicode replacement characters')

    # Check for HTML entities
    if re.search(r'&[a-z]+;', text):
        issues.append('Contains HTML entities')

    # Check for control characters
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        issues.append('Contains control characters')

    return issues


def categorize_length(token_count):
    """Categorize sentence by token count (for imbalance analysis)."""
    if token_count <= 5:
        return 'very_short'
    elif token_count <= 15:
        return 'short'
    elif token_count <= 30:
        return 'medium'
    elif token_count <= 50:
        return 'long'
    else:
        return 'very_long'
