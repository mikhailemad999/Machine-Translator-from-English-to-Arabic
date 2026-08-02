"""
Django settings for English-to-Arabic Machine Translator.
Dual database: SQL Server (structured) + MongoDB (unstructured).
Local mode: SQLite + file-based MongoDB fallback.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
dotenv_path = find_dotenv(usecwd=True) or (BASE_DIR.parent / '.env')
load_dotenv(dotenv_path)

# Security
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-fallback-key')
DEBUG = os.getenv('DJANGO_DEBUG', 'True').lower() == 'true'
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', '*').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third party
    'rest_framework',
    'corsheaders',
    # Local apps
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# ============================================================
# DATABASES — Dual Database Configuration
# ============================================================
# Local mode: SQLite (zero setup)
# Docker mode: SQL Server
# ============================================================

USE_SQLITE = os.getenv('USE_SQLITE', 'False').lower() == 'true'

if USE_SQLITE:
    # SQLite for local development — no install required
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
else:
    # SQL Server for Docker / production
    DATABASES = {
        'default': {
            'ENGINE': 'mssql',
            'NAME': os.getenv('MSSQL_DB_NAME', 'MachineTranslatorEnAr'),
            'USER': 'sa',
            'PASSWORD': os.getenv('MSSQL_SA_PASSWORD', 'TranslatorDB@2024!'),
            'HOST': os.getenv('MSSQL_HOST', 'sqlserver'),
            'PORT': os.getenv('MSSQL_PORT', '1433'),
            'OPTIONS': {
                'driver': 'ODBC Driver 18 for SQL Server',
                'extra_params': 'TrustServerCertificate=yes',
            },
        }
    }

# MongoDB settings (accessed via pymongo, not Django ORM)
MONGODB_SETTINGS = {
    'HOST': os.getenv('MONGO_HOST', 'mongodb'),
    'PORT': int(os.getenv('MONGO_PORT', 27017)),
    'DB_NAME': os.getenv('MONGO_DB_NAME', 'machine_translator_en_ar'),
    'FALLBACK': os.getenv('MONGO_FALLBACK', 'none'),  # 'file' = JSON fallback
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS — allow React frontend
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
]
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# ============================================================
# ML / Data Paths
# ============================================================
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
MODEL_SAVE_DIR = Path(os.getenv('MODEL_SAVE_DIR', BASE_DIR / 'models'))
CHARTS_DIR = Path(os.getenv('CHARTS_DIR', BASE_DIR / 'charts'))

for _dir in [DATA_DIR, MODEL_SAVE_DIR, CHARTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ML Worker
ML_WORKER_URL = f"http://{os.getenv('ML_WORKER_HOST', 'ml_worker')}:{os.getenv('ML_WORKER_PORT', '8001')}"

# Model
MODEL_NAME = os.getenv('MODEL_NAME', 'Helsinki-NLP/opus-mt-en-ar')

# File upload
DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB
