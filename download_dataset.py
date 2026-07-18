"""
Download an EN-AR parallel corpus for training.

Downloads from HuggingFace datasets and saves as a CSV file
with 'en' and 'ar' columns in data/en_ar_dataset.csv.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'en_ar_dataset.csv')

# Maximum samples to download (adjust based on your needs)
MAX_SAMPLES = 50000


def download_dataset():
    """Download EN-AR dataset from HuggingFace."""
    print("=" * 60)
    print("  Downloading EN-AR Training Dataset")
    print("=" * 60)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if dataset already exists
    if os.path.exists(OUTPUT_FILE):
        import pandas as pd
        existing = pd.read_csv(OUTPUT_FILE)
        print(f"Dataset already exists: {OUTPUT_FILE}")
        print(f"  Rows: {len(existing)}")
        print(f"  Columns: {list(existing.columns)}")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return OUTPUT_FILE

    import pandas as pd

    print(f"Target: {MAX_SAMPLES} sentence pairs")
    print()

    records = []

    # Strategy 1: Try opus-100 (high-quality multi-domain data)
    print("[1/2] Trying Helsinki-NLP/opus-100 (ar-en)...")
    try:
        from datasets import load_dataset
        dataset = load_dataset('Helsinki-NLP/opus-100', 'ar-en', split='train', trust_remote_code=True)
        for i, item in enumerate(dataset):
            if i >= MAX_SAMPLES:
                break
            translation = item.get('translation', item)
            if isinstance(translation, dict):
                en = translation.get('en', '')
                ar = translation.get('ar', '')
                if en and ar and len(en.strip()) > 2 and len(ar.strip()) > 2:
                    records.append({'en': en.strip(), 'ar': ar.strip()})
        print(f"  Got {len(records)} pairs from opus-100")
    except Exception as e:
        print(f"  opus-100 failed: {e}")

    # Strategy 2: If we need more data, try opus_books
    if len(records) < MAX_SAMPLES:
        remaining = MAX_SAMPLES - len(records)
        print(f"[2/2] Trying opus_books for {remaining} more pairs...")
        try:
            from datasets import load_dataset
            dataset = load_dataset('opus_books', 'en-ar', split='train', trust_remote_code=True)
            count = 0
            for item in dataset:
                if count >= remaining:
                    break
                translation = item.get('translation', item)
                if isinstance(translation, dict):
                    en = translation.get('en', '')
                    ar = translation.get('ar', '')
                    if en and ar and len(en.strip()) > 2 and len(ar.strip()) > 2:
                        records.append({'en': en.strip(), 'ar': ar.strip()})
                        count += 1
            print(f"  Got {count} additional pairs from opus_books")
        except Exception as e:
            print(f"  opus_books failed: {e}")

    if not records:
        print()
        print("ERROR: Could not download any data!")
        print("Please check your internet connection.")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Remove exact duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['en', 'ar']).reset_index(drop=True)
    print(f"\nRemoved {original_len - len(df)} duplicates")

    # Save
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    print()
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Total pairs: {len(df)}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")
    print()

    # Show samples safely without breaking on console encoding
    try:
        print("Sample pairs:")
        print("-" * 60)
        for _, row in df.head(5).iterrows():
            print(f"  EN: {row['en'][:80]}")
            print(f"  AR: {row['ar'][:80].encode('ascii', 'replace').decode('ascii')}")
            print()
    except Exception:
        pass

    print("=" * 60)
    print("  Dataset download COMPLETE!")
    print("=" * 60)
    return OUTPUT_FILE


if __name__ == '__main__':
    download_dataset()
