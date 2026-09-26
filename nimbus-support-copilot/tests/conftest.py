"""
Pytest loads conftest.py before collecting any test modules, so setting
these here — before any `from app...` import anywhere in the test suite
triggers Settings() to load — means tests never depend on a real .env file
or real secrets existing. This is what makes the suite CI-safe: a CI runner
has none of your local secrets, and shouldn't need them just to run tests
against fake providers.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-for-production")
os.environ.setdefault("GROQ_API_KEY", "test-key-unused-in-mocked-tests")
