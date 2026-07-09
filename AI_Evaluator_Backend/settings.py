

import os
from pathlib import Path
from datetime import timedelta


from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present.
load_dotenv(BASE_DIR / '.env')


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-2ll$i@i3@(g&rj&nlg@8+)=7dd^bw-^@vd6=$71k!7z_jlpurs'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party apps
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',
    'corsheaders',
    'storages',

    # Project apps
    'core',
    'authentication',
    'viva_evaluator',
    
    'code_analysis',
    'projects',

    'drf_spectacular',
    'sessions_app',
    'cloudinary',
    'cloudinary_storage',
    'agora_service',
    'cv_analysis',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Must be at the very top
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'AI_Evaluator_Backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'AI_Evaluator_Backend.wsgi.application'


# Database
# https://docs.djangoproject.com/en/6.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'neondb',
        'USER': 'neondb_owner',
        'PASSWORD': 'npg_OH6M0sYiFjAT',
        'HOST': 'ep-sweet-shape-ao5kxg2i.c-2.ap-southeast-1.aws.neon.tech',
        'PORT': '5432',
        'OPTIONS': {
            'sslmode': 'require',
        },
    }
}

# Password validation
# https://docs.djangoproject.com/en/6.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Colombo'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = 'static/'

# Media uploads
MEDIA_URL = '/uploads/'
MEDIA_ROOT = BASE_DIR / 'uploads'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# =============================================================================
# Custom User Model
# =============================================================================
AUTH_USER_MODEL = 'core.User'


# =============================================================================
# Django REST Framework
# =============================================================================
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        # Reads the access token from the Authorization header OR the HttpOnly
        # cookie set at login.
        'authentication.authentication.CookieJWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}


# =============================================================================
# Simple JWT Configuration
# =============================================================================
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
}


# =============================================================================
# CORS / CSRF Configuration
# -----------------------------------------------------------------------------
# Auth uses HttpOnly cookies, so the browser must be allowed to send
# credentials cross-subdomain (vivasense.tech -> api.vivasense.tech). That
# requires explicit origins (wildcard is not allowed with credentials).
# =============================================================================
def _split_env(name, default=''):
    return [o.strip() for o in os.getenv(name, default).split(',') if o.strip()]


CORS_ALLOWED_ORIGINS = _split_env(
    'CORS_ALLOWED_ORIGINS',
    'http://localhost:3000,http://127.0.0.1:3000',
)
CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = _split_env(
    'CSRF_TRUSTED_ORIGINS',
    'http://localhost:3000,http://127.0.0.1:3000',
)

# =============================================================================
# Auth Cookie Configuration
# -----------------------------------------------------------------------------
# Dev defaults are insecure (http, host-only) so the Next.js /api proxy makes
# everything same-origin on localhost. In production set:
#   AUTH_COOKIE_DOMAIN=.vivasense.tech
#   AUTH_COOKIE_SECURE=true
# =============================================================================
AUTH_COOKIE_ACCESS_NAME = os.getenv('AUTH_COOKIE_ACCESS_NAME', 'access_token')
AUTH_COOKIE_REFRESH_NAME = os.getenv('AUTH_COOKIE_REFRESH_NAME', 'refresh_token')
AUTH_COOKIE_DOMAIN = os.getenv('AUTH_COOKIE_DOMAIN') or None
AUTH_COOKIE_SECURE = os.getenv('AUTH_COOKIE_SECURE', 'false').lower() == 'true'
AUTH_COOKIE_SAMESITE = os.getenv('AUTH_COOKIE_SAMESITE', 'Lax')

# =============================================================================
# Cloudinary Configuration
# =============================================================================
CLOUDINARY_STORAGE = {
    'CLOUD_NAME': os.getenv('CLOUDINARY_CLOUD_NAME'),
    'API_KEY': os.getenv('CLOUDINARY_API_KEY'),
    'API_SECRET': os.getenv('CLOUDINARY_API_SECRET'),
}

STORAGES = {
    "default": {
        "BACKEND": "cloudinary_storage.storage.MediaCloudinaryStorage",
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

# =============================================================================
# Media Files
# =============================================================================
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# =============================================================================
# Gemini API
# =============================================================================

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')

GROQ_API_KEY = ''
# =============================================================================
# Code Analysis Configuration

# =============================================================================
SONAR_HOST_URL = os.getenv('SONAR_HOST_URL', 'https://sonarcloud.io')
SONAR_ORG_KEY = os.getenv('SONAR_ORG_KEY', '')
SONAR_TOKEN = os.getenv('SONAR_TOKEN', '')
SONAR_SCANNER_BIN = os.getenv('SONAR_SCANNER_BIN', 'sonar-scanner')


CODE_ANALYSIS_MAX_ZIP_MB = int(os.getenv('CODE_ANALYSIS_MAX_ZIP_MB', '100'))
CODE_ANALYSIS_MAX_PROMPT_CHARS = int(os.getenv('CODE_ANALYSIS_MAX_PROMPT_CHARS', '20000'))
CODE_ANALYSIS_ASYNC = os.getenv('CODE_ANALYSIS_ASYNC', 'true').lower() == 'true'

# D1 — run report FAISS indexing (image captioning + embeddings) in a
# background thread instead of inside the upload request/transaction.
REPORT_INDEX_ASYNC = os.getenv('REPORT_INDEX_ASYNC', 'true').lower() == 'true'
CODE_ANALYSIS_MAX_RATING = float(os.getenv('CODE_ANALYSIS_MAX_RATING', '2'))
CODE_ANALYSIS_MIN_COVERAGE = float(os.getenv('CODE_ANALYSIS_MIN_COVERAGE', '0'))
CODE_ANALYSIS_MAX_DUPLICATION = float(os.getenv('CODE_ANALYSIS_MAX_DUPLICATION', '5'))

# =============================================================================
# Azure Blob Storage (django-storages)
# =============================================================================
STORAGES = {
    "default": {
        "BACKEND": "storages.backends.azure_storage.AzureStorage",
        "OPTIONS": {
            "account_name": os.getenv("AZURE_ACCOUNT_NAME"),
            "account_key": os.getenv("AZURE_ACCOUNT_KEY"),
            "azure_container": os.getenv("AZURE_CONTAINER", "media"),
            "expiration_secs": None,
        },
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

CODE_ANALYSIS_ALLOWED_EXTENSIONS = [
    '.py',
    '.js',
    '.ts',
    '.java',
    '.cpp',
    '.c',
    '.h',
    '.cs',
    '.go',
    '.rb',
    '.php',
    '.kt',
    '.swift',
    '.rs',
    '.json',
    '.yml',
    '.yaml',
    '.toml',
    '.xml',
    '.html',
    '.css',
]

SPECTACULAR_SETTINGS = {
    'TITLE': 'AI Project Evaluator — Viva Module API',
    'DESCRIPTION': 'API for student and examiner interactions in the Viva Evaluation system.',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False,
}



# =============================================================================
# Logging — surface INFO logs (incl. per-LLM-call latency + turn timing)
# in the dev console so latency can be diagnosed in real time.
# =============================================================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
            'datefmt': '%H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        # Our viva pipeline modules
        'viva_evaluator': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# =============================================================================
# Agora RTC & STT Configuration
# =============================================================================
AGORA_APP_ID = os.getenv('AGORA_APP_ID', '')
AGORA_APP_CERTIFICATE = os.getenv('AGORA_APP_CERTIFICATE', '')
AGORA_CUSTOMER_KEY = os.getenv('AGORA_CUSTOMER_KEY', '')
AGORA_CUSTOMER_SECRET = os.getenv('AGORA_CUSTOMER_SECRET', '')
AGORA_STT_ENABLED = os.getenv('AGORA_STT_ENABLED', 'false').lower() == 'true'

# Agora Cloud Recording — server-side channel recording into Azure Blob.
# Metered add-on; must be enabled on the Agora project. AZURE region code is
# Agora's storageConfig.region enum for your storage account's region.
AGORA_CLOUD_RECORDING_ENABLED = os.getenv(
    'AGORA_CLOUD_RECORDING_ENABLED', 'false',
).lower() == 'true'
AGORA_RECORDING_AZURE_REGION = int(os.getenv('AGORA_RECORDING_AZURE_REGION', '0'))

# =============================================================================
# CV / Behavioral Analysis (exam-station-cv engine)
# =============================================================================
# Off by default: cloud deploys without the CV toolchain still store
# recordings; analysis runs where the engine (and its venv) exists.
CV_ANALYSIS_ENABLED = os.getenv('CV_ANALYSIS_ENABLED', 'false').lower() == 'true'
CV_ANALYSIS_ASYNC = os.getenv('CV_ANALYSIS_ASYNC', 'true').lower() == 'true'
# Python executable of the exam-station-cv virtualenv (heavy CV deps live
# there, not in this venv). Default assumes the in-repo module layout.
CV_ANALYSIS_PYTHON = os.getenv(
    'CV_ANALYSIS_PYTHON',
    str(BASE_DIR / 'exam-station-cv' / '.venv' / 'Scripts' / 'python.exe'),
)
CV_ANALYSIS_TIMEOUT = int(os.getenv('CV_ANALYSIS_TIMEOUT', '1800'))

# Recording storage: 'local' (on this machine — no Azure cost) or 'azure'.
CV_RECORDING_STORAGE = os.getenv('CV_RECORDING_STORAGE', 'local').lower()
CV_RECORDINGS_DIR = os.getenv(
    'CV_RECORDINGS_DIR', str(BASE_DIR / 'cv_recordings'),
)
