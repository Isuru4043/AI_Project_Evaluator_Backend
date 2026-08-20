"""Fast, isolated settings used by the repository's local Django tests."""

from .settings import *  # noqa: F401,F403


TESTING = True
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    },
}
PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']
Q_CLUSTER = {**Q_CLUSTER, 'sync': True}  # noqa: F405

# The repository contains historical parallel core migration branches that do
# not replay cleanly on SQLite. Tests build these local app tables directly
# from the current models; production continues to use the real migrations.
MIGRATION_MODULES = {
    'core': None,
    'viva_evaluator': None,
    'cv_analysis': None,
    'physical_evaluation': None,
}
