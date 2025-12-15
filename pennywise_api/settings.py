# settings.py (db, celery, static bits)

import os
from urllib.parse import urlparse
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

from django.core.management.utils import get_random_secret_key

SECRET_KEY = os.getenv("SECRET_KEY") or ("dev-" + get_random_secret_key())


DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "YES", "yes")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")

WSGI_APPLICATION = "pennywise_api.wsgi.application"

# ---- Database ----
# Prefer explicit envs from docker-compose; allow DATABASE_URL to override if provided.
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "postgres"))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres"))
DB_HOST = os.getenv("DB_HOST", "db")          # <- Docker service name, not localhost
DB_PORT = int(os.getenv("DB_PORT", "5432"))

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    p = urlparse(DATABASE_URL)
    # Only override when parts are present
    DB_NAME = (p.path.lstrip("/") or DB_NAME)
    DB_USER = (p.username or DB_USER)
    DB_PASSWORD = (p.password or DB_PASSWORD)
    DB_HOST = (p.hostname or DB_HOST)
    DB_PORT = (p.port or DB_PORT)

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": DB_NAME,
        "USER": DB_USER,
        "PASSWORD": DB_PASSWORD,
        "HOST": DB_HOST,
        "PORT": DB_PORT,
    }
}

# ---- Static / Media (fine for dev; tweak for prod) ----
STATIC_URL = "/static/"
STATIC_ROOT = os.getenv("STATIC_ROOT", os.path.join(os.path.dirname(__file__), "..", "staticfiles"))
MEDIA_URL = "/media/"
MEDIA_ROOT = os.getenv("MEDIA_ROOT", os.path.join(os.path.dirname(__file__), "..", "media"))

# ---- DRF + Spectacular ----
REST_FRAMEWORK = {"DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema"}
SPECTACULAR_SETTINGS = {
    "TITLE": "PennyWise Receipt AI API",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# ---- Celery / Redis ----
# Use Docker service name "redis" by default; allow REDIS_URL/CELERY_* overrides.
DEFAULT_REDIS_URL = "redis://redis:6379/0"
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", DEFAULT_REDIS_URL))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

# ---- LLM service (keep your defaults, but avoid localhost inside containers) ----
LLM_PROVIDER_URL = os.getenv("LLM_PROVIDER_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2-vision")


ROOT_URLCONF = "pennywise_api.urls"          # path to urls.py

INSTALLED_APPS = [
    # Django core
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # 3rd-party
    "rest_framework",
    "drf_spectacular",

    # Your apps
    "apps.receipts",   # <-- add this
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",          # required by admin
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",       # required by admin
    "django.contrib.messages.middleware.MessageMiddleware",          # required by admin
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # or [] if you don't have a templates dir
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# storage
DEFAULT_FILE_STORAGE = ".backends.s3boto3.S3Boto3Storage"
AWS_S3_ENDPOINT_URL = "http://minio:9000"
AWS_ACCESS_KEY_ID = "minioaccess"
AWS_SECRET_ACCESS_KEY = "miniosecret"
AWS_STORAGE_BUCKET_NAME = "receipts"
AWS_S3_REGION_NAME = "us-east-1"
AWS_S3_SIGNATURE_VERSION = "s3v4"
AWS_S3_ADDRESSING_STYLE = "path"

# Where uploaded receipt images are stored when clients send files directly
RECEIPT_UPLOAD_DIR = os.getenv("RECEIPT_UPLOAD_DIR", str(BASE_DIR / "tmp" / "receipt_uploads"))

# LLM configuration (provider + model selection + optional prompt file)
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),  # ollama | openai
    "ollama": {
        "url": os.getenv("LLM_PROVIDER_URL", "http://ollama:11434"),
        "model": os.getenv("LLM_MODEL", "llama3.2-vision"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    },
    # Optional external prompt template; supports simple macro replacement using {{MODEL}} and {{PROVIDER}}
    "prompt_file": os.getenv("LLM_PROMPT_FILE", ""),
    "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
}
