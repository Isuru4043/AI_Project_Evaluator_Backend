

import json
import os
from pathlib import Path
from datetime import timedelta


from corsheaders.defaults import default_headers
from dotenv import load_dotenv
from core.services.google_auth import configure_google_credentials

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present.
load_dotenv(BASE_DIR / '.env')

configure_google_credentials()
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-dev-key-change-me')
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
    'attribution',
    'physiology',
    'physical_evaluation',
    'django_q',
]

# Physical evaluation kiosk defaults. The kiosk lease is intentionally short
# lived and can only call the physical-session and shared-viva endpoints.
PHYSICAL_KIOSK_TOKEN_LIFETIME_HOURS = int(
    os.getenv('PHYSICAL_KIOSK_TOKEN_LIFETIME_HOURS', '12')
)

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Must be at the very top
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'AI_Evaluator_Backend.middleware.TrailingSlashAPIMiddleware',  # Fix slashes before CommonMiddleware
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
        'NAME': os.environ.get('NEON_DATABASE_NAME', 'neondb'),
        'USER': os.environ.get('NEON_DATABASE_USER', 'neondb_owner'),
        'PASSWORD': os.environ.get(
            'NEON_DATABASE_PASSWORD',
            os.environ.get('DATABASE_PASSWORD', ''),
        ),
        'HOST': os.environ.get(
            'NEON_DATABASE_HOST',
            'ep-fragrant-dawn-azhkavms.c-3.ap-southeast-1.aws.neon.tech',
        ),
        'PORT': os.environ.get('NEON_DATABASE_PORT', '5432'),
        'OPTIONS': {
            'sslmode': 'require',
            'channel_binding': os.environ.get('NEON_CHANNEL_BINDING', 'require'),
        },
    }
}

# Opt-in local database, for working when the shared Neon instance is
# unreachable (quota exhausted, offline, or simply to avoid writing test rows
# into a database other people are using).
#   USE_SQLITE=true python manage.py migrate
# Never set in a deployed environment; the default above stays authoritative.
if os.getenv('USE_SQLITE', '').lower() == 'true':
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'dev.sqlite3',
        }
    }

    # The core migration history has two parallel branches that both add
    # VivaQuestion.project (core.0006 and core.0007), so replaying it from
    # scratch fails with "duplicate column name: project_id". The deployed
    # database has them already applied and never re-runs them, so this only
    # bites a brand-new database.
    #
    # Rather than rewrite that history - which risks desyncing the deployed
    # database - the local one is built straight from the current models with
    #     USE_SQLITE=true python manage.py migrate --run-syncdb
    # This is scoped to the SQLite path and changes nothing for a real deploy.
    class _SkipMigrations:
        def __contains__(self, item):
            return True

        def __getitem__(self, item):
            return None

    MIGRATION_MODULES = _SkipMigrations()

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
CORS_ALLOW_HEADERS = (
    *default_headers,
    'x-physical-kiosk-token',
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
# Vertex AI Gemini (service-account / ADC authentication)
# =============================================================================

GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT', '').strip()
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'global').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite').strip()

# Optional per-million-token prices used only for telemetry cost estimates.
# Keep provider pricing in deployment configuration because it changes over
# time. Invalid or absent JSON disables monetary estimates while token metrics
# continue to work.
try:
    LLM_MODEL_PRICING = json.loads(
        os.getenv('LLM_MODEL_PRICING_JSON', '{}')
    )
except (json.JSONDecodeError, TypeError):
    LLM_MODEL_PRICING = {}
if not isinstance(LLM_MODEL_PRICING, dict):
    LLM_MODEL_PRICING = {}

# Resolve relative credential paths from the repository root so authentication
# works consistently under manage.py, Gunicorn, background workers, and scripts.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    'GOOGLE_APPLICATION_CREDENTIALS',
    '',
).strip()

if GOOGLE_APPLICATION_CREDENTIALS:
    _google_credentials_path = Path(GOOGLE_APPLICATION_CREDENTIALS).expanduser()
    if not _google_credentials_path.is_absolute():
        _google_credentials_path = BASE_DIR / _google_credentials_path
    _google_credentials_path = _google_credentials_path.resolve()

    if not _google_credentials_path.is_file():
        from django.core.exceptions import ImproperlyConfigured
        raise ImproperlyConfigured(
            'GOOGLE_APPLICATION_CREDENTIALS points to a file that does not exist: '
            f'{_google_credentials_path}'
        )

    GOOGLE_APPLICATION_CREDENTIALS = str(_google_credentials_path)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

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

# Question-validation rollout policy when the Tier-2 Critic cannot be reached:
#   degraded_tier1 — serve the Tier-1-valid generated question with an audit flag
#   safe_fallback  — replace it with a deterministic Tier-1-valid question
#   fail_closed    — return safe_question_unavailable (HTTP 503)
VIVA_QUESTION_CRITIC_UNAVAILABLE_POLICY = os.getenv(
    'VIVA_QUESTION_CRITIC_UNAVAILABLE_POLICY',
    'degraded_tier1',
).strip().lower()

# Speculative question speech. The API key remains server-side; the browser
# only requests audio for a persisted question through an authenticated route.
ELEVENLABS_TTS_ENABLED = os.getenv(
    'ELEVENLABS_TTS_ENABLED', 'false'
).lower() == 'true'
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '').strip()
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '').strip()
ELEVENLABS_MODEL_ID = os.getenv(
    'ELEVENLABS_MODEL_ID', 'eleven_flash_v2_5'
).strip()
ELEVENLABS_OUTPUT_FORMAT = os.getenv(
    'ELEVENLABS_OUTPUT_FORMAT', 'mp3_44100_128'
).strip()
ELEVENLABS_TIMEOUT_SECONDS = float(
    os.getenv('ELEVENLABS_TIMEOUT_SECONDS', '20')
)
ELEVENLABS_TTS_CACHE_MAX_JOBS = int(
    os.getenv('ELEVENLABS_TTS_CACHE_MAX_JOBS', '256')
)
ELEVENLABS_PRICE_PER_1000_CHARACTERS_USD = float(
    os.getenv('ELEVENLABS_PRICE_PER_1000_CHARACTERS_USD', '0')
)

# Student answer transcription (ElevenLabs Scribe).  Reuses ELEVENLABS_API_KEY;
# the browser posts short recorded utterances to an authenticated route so the
# key stays server-side.  Disabled means the UI falls back to browser speech.
ELEVENLABS_STT_ENABLED = os.getenv(
    'ELEVENLABS_STT_ENABLED', 'false'
).lower() == 'true'
ELEVENLABS_STT_MODEL_ID = os.getenv(
    'ELEVENLABS_STT_MODEL_ID', 'scribe_v1'
).strip()
# Empty means auto-detect; set e.g. 'eng' to pin the expected viva language.
ELEVENLABS_STT_LANGUAGE_CODE = os.getenv(
    'ELEVENLABS_STT_LANGUAGE_CODE', ''
).strip()
ELEVENLABS_STT_TIMEOUT_SECONDS = float(
    os.getenv('ELEVENLABS_STT_TIMEOUT_SECONDS', '25')
)
ELEVENLABS_STT_MAX_AUDIO_BYTES = int(
    os.getenv('ELEVENLABS_STT_MAX_AUDIO_BYTES', '12000000')
)
ELEVENLABS_STT_MIN_AUDIO_BYTES = int(
    os.getenv('ELEVENLABS_STT_MIN_AUDIO_BYTES', '1200')
)

# Offline performance gates used by question_validation_report --enforce.
# They never interrupt a live viva; tune them after collecting a representative
# baseline for the deployment region and selected provider models.
VIVA_PERF_MIN_TURNS = int(os.getenv('VIVA_PERF_MIN_TURNS', '20'))
VIVA_PERF_MAX_P95_TURN_LATENCY_MS = float(
    os.getenv('VIVA_PERF_MAX_P95_TURN_LATENCY_MS', '30000')
)
VIVA_PERF_MAX_MEAN_CALLS_PER_TURN = float(
    os.getenv('VIVA_PERF_MAX_MEAN_CALLS_PER_TURN', '5')
)
VIVA_PERF_MAX_DEGRADED_RATE = float(
    os.getenv('VIVA_PERF_MAX_DEGRADED_RATE', '0.10')
)
VIVA_PERF_MAX_FALLBACK_RATE = float(
    os.getenv('VIVA_PERF_MAX_FALLBACK_RATE', '0.05')
)
VIVA_PERF_MIN_TIER1_PASS_RATE = float(
    os.getenv('VIVA_PERF_MIN_TIER1_PASS_RATE', '0.95')
)
VIVA_PERF_MAX_MEAN_COST_PER_TURN_USD = float(
    os.getenv('VIVA_PERF_MAX_MEAN_COST_PER_TURN_USD', '0')
)

# =============================================================================
# Agora RTC & STT Configuration
# =============================================================================
AGORA_APP_ID = os.getenv('AGORA_APP_ID', '')
AGORA_APP_CERTIFICATE = os.getenv('AGORA_APP_CERTIFICATE', '')
AGORA_CUSTOMER_KEY = os.getenv('AGORA_CUSTOMER_KEY', '')
AGORA_CUSTOMER_SECRET = os.getenv('AGORA_CUSTOMER_SECRET', '')
AGORA_STT_ENABLED = os.getenv('AGORA_STT_ENABLED', 'false').lower() == 'true'

# Agora Cloud Recording — server-side channel recording into Azure Blob.
# Metered add-on; must be enabled on the Agora project. Agora requires a
# storageConfig.region field but does not use it for Microsoft Azure.
AGORA_CLOUD_RECORDING_ENABLED = os.getenv(
    'AGORA_CLOUD_RECORDING_ENABLED', 'false',
).lower() == 'true'
AGORA_RECORDING_AZURE_REGION = int(os.getenv('AGORA_RECORDING_AZURE_REGION', '0'))
# Prefer Agora's Southeast Asia REST edge for this deployment. The global
# endpoint has intermittently exceeded the recording client's timeout from
# this region. Override this for deployments hosted elsewhere.
AGORA_REST_BASE_URL = os.getenv(
    'AGORA_REST_BASE_URL',
    'https://api-ap-southeast-1.agora.io',
).rstrip('/')

# =============================================================================
# CV / Behavioral Analysis (exam-station-cv engine)
# =============================================================================
# Off by default: cloud deploys without the CV toolchain still store
# recordings; analysis runs where the engine (and its venv) exists.
CV_ANALYSIS_ENABLED = os.getenv('CV_ANALYSIS_ENABLED', 'false').lower() == 'true'
CV_ANALYSIS_ASYNC = os.getenv('CV_ANALYSIS_ASYNC', 'true').lower() == 'true'

# Where the engine runs:
#   'modal'      — Modal CPU containers (cloud default; see cv_analyze_modal.py)
#   'subprocess' — the exam-station-cv venv on this machine (local dev)
CV_ANALYSIS_BACKEND = os.getenv('CV_ANALYSIS_BACKEND', 'modal').lower()

# Modal endpoints from `modal deploy cv_analyze_modal.py`. Analysis is
# submit/poll: a 20-min viva takes ~5-10 min to process.
MODAL_CV_SUBMIT_URL = os.getenv('MODAL_CV_SUBMIT_URL', '')
MODAL_CV_RESULT_URL = os.getenv('MODAL_CV_RESULT_URL', '')
MODAL_CV_TOKEN = os.getenv('MODAL_CV_TOKEN', '')

# =============================================================================
# SPEAKER ATTRIBUTION — who answered, in a group viva
# =============================================================================
# On by default and safe: with no evidence providers reporting, every answer
# resolves to 'group', which is exactly the behaviour before this app existed.
ATTRIBUTION_ENABLED = os.getenv('ATTRIBUTION_ENABLED', 'true').lower() == 'true'

# How far each evidence source counts in the fusion vote. Tune against a
# labelled pilot session — the defaults are defensible priors, not
# measurements. Anything omitted falls back to
# attribution.services.resolver.DEFAULT_WEIGHTS.
ATTRIBUTION_SOURCE_WEIGHTS = {
    'manual': 1.00,        # examiner/kiosk choice — short-circuits the vote
    'agora_stt': 0.90,     # per-UID stream, timestamped, carries the words
    'agora_volume': 0.85,  # per-UID stream, noisier
    'posthoc_cv': 0.80,    # full CV engine: ArcFace + iris gaze
    'live_cv': 0.70,       # same logic under real-time constraints
    'submitter': 0.50,     # weak prior: who pressed submit
}

# Where physical seat binding (face -> student) runs:
#   'modal' — the Modal CV app (needs MODAL_CV_BIND_URL)
#   'local' — the CV engine in this process (dev; needs its heavy deps)
ATTRIBUTION_BINDING_BACKEND = os.getenv(
    'ATTRIBUTION_BINDING_BACKEND', 'modal',
).lower()
MODAL_CV_BIND_URL = os.getenv('MODAL_CV_BIND_URL', '')
MODAL_CV_ENROLL_URL = os.getenv('MODAL_CV_ENROLL_URL', '')
ATTRIBUTION_IDENTITY_MIN_CONFIDENCE = float(os.getenv(
    'ATTRIBUTION_IDENTITY_MIN_CONFIDENCE', '0.42',
))
ATTRIBUTION_IDENTITY_MIN_MARGIN = float(os.getenv(
    'ATTRIBUTION_IDENTITY_MIN_MARGIN', '0.05',
))
ATTRIBUTION_IDENTITY_MIN_VOTES = int(os.getenv(
    'ATTRIBUTION_IDENTITY_MIN_VOTES', '2',
))

# Shared secret for a physical exam station running the CV engine as its own
# process (exam-station-cv --backend-url). It has no browser session, so it
# authenticates with this instead. Empty = no station may connect, which is
# the right default for a cloud deploy with no physical stations.
# It only permits ADDING evidence — never reading data or changing a score.
EXAM_STATION_TOKEN = os.getenv('EXAM_STATION_TOKEN', '')

# Python executable of the exam-station-cv virtualenv (heavy CV deps live
# there, not in this venv). Only used by the 'subprocess' backend.
CV_ANALYSIS_PYTHON = os.getenv(
    'CV_ANALYSIS_PYTHON',
    str(BASE_DIR / 'exam-station-cv' / '.venv' / 'Scripts' / 'python.exe'),
)
CV_ANALYSIS_TIMEOUT = int(os.getenv('CV_ANALYSIS_TIMEOUT', '3600'))

# Recording storage for the legacy client-upload endpoint: 'azure' or 'local'.
# The live path no longer uses it — Agora Cloud Recording writes the session
# recording straight to blob (see agora_service/cloud_recording.py).
CV_RECORDING_STORAGE = os.getenv('CV_RECORDING_STORAGE', 'azure').lower()
CV_RECORDINGS_DIR = os.getenv(
    'CV_RECORDINGS_DIR', str(BASE_DIR / 'cv_recordings'),
)

# =============================================================================
# Modal Serverless GPU Endpoints
# =============================================================================
MODAL_CANARY_URL = os.getenv('MODAL_CANARY_URL', '')
MODAL_QWEN_VL_URL = os.getenv('MODAL_QWEN_VL_URL', '')

# =============================================================================
# Django-Q2 Background Task Queue
# =============================================================================
Q_CLUSTER = {
    'name': 'viva_queue',
    'workers': 4,
    'timeout': 120,
    'retry': 150,
    'orm': 'default',
}
