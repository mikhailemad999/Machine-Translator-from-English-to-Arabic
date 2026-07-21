"""
MongoDB Client Wrapper for Machine Translator.

Handles all MongoDB operations:
- Store raw uploaded datasets
- Store EDA artifacts (charts metadata, exploration reports)
- Store experiment logs

Falls back to JSON file storage when MongoDB is not available
(controlled by MONGO_FALLBACK=file in .env).
"""
import os
import json
import datetime
from pathlib import Path


def _get_fallback_mode():
    """Check if we should use file-based fallback."""
    try:
        from django.conf import settings
        return settings.MONGODB_SETTINGS.get('FALLBACK', 'none') == 'file'
    except Exception:
        return os.getenv('MONGO_FALLBACK', 'none') == 'file'


# ============================================================
# File-Based Fallback (JSON storage for local dev)
# ============================================================

class FileBasedDB:
    """JSON file-based storage that mimics MongoDB operations for local dev."""

    _base_dir = None

    @classmethod
    def _get_base_dir(cls):
        """Retrieve or initialize the base directory for local JSON files storage."""
        if cls._base_dir is None:
            try:
                from django.conf import settings
                cls._base_dir = Path(settings.BASE_DIR) / 'local_mongo_data'
            except Exception:
                cls._base_dir = Path(os.getenv(
                    'DATA_DIR',
                    os.path.join(os.path.dirname(__file__), 'local_mongo_data')
                ))
            cls._base_dir.mkdir(parents=True, exist_ok=True)
        return cls._base_dir

    @classmethod
    def _get_collection_path(cls, name):
        """Get the file path corresponding to a collection JSON file."""
        path = cls._get_base_dir() / f"{name}.json"
        if not path.exists():
            path.write_text('[]', encoding='utf-8')
        return path

    @classmethod
    def _read_collection(cls, name):
        """Load and parse the records list from the specified collection file."""
        path = cls._get_collection_path(name)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    @classmethod
    def _write_collection(cls, name, data):
        """Write the records list back to the specified collection JSON file."""
        path = cls._get_collection_path(name)
        path.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2), encoding='utf-8')

    @classmethod
    def insert_many(cls, collection_name, records):
        """Write multiple records to the collection file."""
        cls._write_collection(collection_name, records)
        return len(records)

    @classmethod
    def insert_one(cls, collection_name, record):
        """Append a single record with a mock string ID to the collection file."""
        data = cls._read_collection(collection_name)
        record['_id'] = str(len(data) + 1)
        data.append(record)
        cls._write_collection(collection_name, data)
        return record['_id']

    @classmethod
    def find(cls, collection_name, query=None, limit=None):
        """Retrieve records from the collection matching the simple query filter dictionary."""
        data = cls._read_collection(collection_name)
        if query:
            data = [d for d in data if all(d.get(k) == v for k, v in query.items())]
        if limit:
            data = data[:limit]
        # Remove _id from results
        return [{k: v for k, v in d.items() if k != '_id'} for d in data]

    @classmethod
    def find_one(cls, collection_name, query=None, sort_key=None, sort_desc=True):
        """Find one record in the collection with optional sorting field."""
        data = cls._read_collection(collection_name)
        if query:
            data = [d for d in data if all(d.get(k) == v for k, v in query.items())]
        if sort_key and data:
            data.sort(key=lambda x: x.get(sort_key, ''), reverse=sort_desc)
        if data:
            return data[0]
        return None

    @classmethod
    def drop(cls, collection_name):
        """Drop the collection by clearing its JSON file contents to an empty array."""
        path = cls._get_collection_path(collection_name)
        path.write_text('[]', encoding='utf-8')

    @classmethod
    def count(cls, collection_name, query=None):
        """Count the number of items in a JSON collection file."""
        data = cls._read_collection(collection_name)
        if query:
            data = [d for d in data if all(d.get(k) == v for k, v in query.items())]
        return len(data)


# ============================================================
# MongoDB Client (with automatic fallback)
# ============================================================

class MongoDBClient:
    """Singleton-style MongoDB client for the Machine Translator project."""

    _client = None
    _db = None
    _use_fallback = None

    @classmethod
    def _should_fallback(cls):
        """Detect whether we should fall back to JSON file-based storage."""
        if cls._use_fallback is None:
            if _get_fallback_mode():
                cls._use_fallback = True
            else:
                try:
                    from pymongo import MongoClient
                    from django.conf import settings
                    client = MongoClient(
                        host=settings.MONGODB_SETTINGS['HOST'],
                        port=settings.MONGODB_SETTINGS['PORT'],
                        serverSelectionTimeoutMS=2000,
                    )
                    client.server_info()  # Test connection
                    cls._client = client
                    cls._db = client[settings.MONGODB_SETTINGS['DB_NAME']]
                    cls._use_fallback = False
                except Exception:
                    print("[MongoDBClient] MongoDB unavailable — using file-based fallback")
                    cls._use_fallback = True
        return cls._use_fallback

    @classmethod
    def get_client(cls):
        """Return the underlying pymongo MongoClient instance, or None if in fallback mode."""
        if cls._should_fallback():
            return None
        return cls._client

    @classmethod
    def get_db(cls):
        """Return the active MongoDB database object, or None if in fallback mode."""
        if cls._should_fallback():
            return None
        return cls._db

    # ============================================================
    # Raw Dataset Operations
    # ============================================================

    @classmethod
    def store_raw_dataset(cls, collection_name, records):
        """
        Store raw sentence pairs in MongoDB.

        Args:
            collection_name: Name of the collection (e.g., 'dataset_1_raw')
            records: List of dicts with 'en' and 'ar' keys
        Returns:
            Number of inserted documents
        """
        if cls._should_fallback():
            FileBasedDB.drop(collection_name)
            return FileBasedDB.insert_many(collection_name, records) if records else 0

        db = cls.get_db()
        collection = db[collection_name]
        # Clear existing data in this collection
        collection.drop()
        if records:
            result = collection.insert_many(records)
            return len(result.inserted_ids)
        return 0

    @classmethod
    def get_raw_dataset(cls, collection_name, limit=None):
        """Retrieve raw sentence pairs from MongoDB."""
        if cls._should_fallback():
            return FileBasedDB.find(collection_name, limit=limit)

        db = cls.get_db()
        collection = db[collection_name]
        cursor = collection.find({}, {'_id': 0})
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    @classmethod
    def get_raw_dataset_count(cls, collection_name):
        """Get count of documents in a raw dataset collection."""
        if cls._should_fallback():
            return FileBasedDB.count(collection_name)

        db = cls.get_db()
        return db[collection_name].count_documents({})

    # ============================================================
    # EDA Artifacts
    # ============================================================

    @classmethod
    def store_eda_report(cls, dataset_id, report_data):
        """Store EDA report data (exploration results, chart metadata)."""
        report_data['dataset_id'] = dataset_id
        report_data['created_at'] = datetime.datetime.utcnow().isoformat()

        if cls._should_fallback():
            return FileBasedDB.insert_one('eda_reports', report_data)

        db = cls.get_db()
        collection = db['eda_reports']
        report_data['created_at'] = datetime.datetime.utcnow()
        result = collection.insert_one(report_data)
        return str(result.inserted_id)

    @classmethod
    def get_eda_report(cls, dataset_id):
        """Get the latest EDA report for a dataset."""
        if cls._should_fallback():
            report = FileBasedDB.find_one(
                'eda_reports',
                query={'dataset_id': dataset_id},
                sort_key='created_at',
                sort_desc=True,
            )
            return report

        db = cls.get_db()
        collection = db['eda_reports']
        report = collection.find_one(
            {'dataset_id': dataset_id},
            sort=[('created_at', -1)]
        )
        if report:
            report['_id'] = str(report['_id'])
        return report

    # ============================================================
    # Experiment Logs
    # ============================================================

    @classmethod
    def log_experiment(cls, training_job_id, log_entry):
        """Log an experiment event (epoch completed, metric recorded, etc.)."""
        log_entry['training_job_id'] = training_job_id
        log_entry['timestamp'] = datetime.datetime.utcnow().isoformat()

        if cls._should_fallback():
            FileBasedDB.insert_one('experiment_logs', log_entry)
            return

        db = cls.get_db()
        collection = db['experiment_logs']
        log_entry['timestamp'] = datetime.datetime.utcnow()
        collection.insert_one(log_entry)

    @classmethod
    def get_experiment_logs(cls, training_job_id):
        """Get all experiment logs for a training job."""
        if cls._should_fallback():
            return FileBasedDB.find('experiment_logs', query={'training_job_id': training_job_id})

        db = cls.get_db()
        collection = db['experiment_logs']
        logs = list(collection.find(
            {'training_job_id': training_job_id},
            {'_id': 0}
        ).sort('timestamp', 1))
        return logs

    # ============================================================
    # Preprocessing Artifacts
    # ============================================================

    @classmethod
    def store_cleaned_pairs(cls, collection_name, records):
        """Store cleaned sentence pairs (intermediate step before SQL)."""
        if cls._should_fallback():
            FileBasedDB.drop(collection_name)
            return FileBasedDB.insert_many(collection_name, records) if records else 0

        db = cls.get_db()
        collection = db[collection_name]
        collection.drop()
        if records:
            result = collection.insert_many(records)
            return len(result.inserted_ids)
        return 0
