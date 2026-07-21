"""
Tests — test_environment.py
Tests for environment variable reading and priority (system > .env).
"""

import os
import sys
import unittest
from unittest import mock


class TestEnvironmentConfig(unittest.TestCase):
    """Test that config reads env vars correctly."""

    def _import_config_fresh(self):
        """Re-import config module to pick up patched env vars."""
        # Remove cached module to force re-import
        mods_to_remove = [k for k in sys.modules if "ComfyUI_LocalAssetFactory" in k or "config" in k]
        for m in mods_to_remove:
            del sys.modules[m]
        # We need a clean import of config helpers
        # Since config uses module-level code, we test the helper functions directly
        from importlib import import_module
        return import_module("config")

    def test_bool_env_true_values(self):
        """Various truthy values should parse correctly."""
        # Import the helper directly
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _bool_env

        for val in ("1", "true", "True", "TRUE", "yes", "on"):
            with mock.patch.dict(os.environ, {"TEST_BOOL": val}):
                self.assertTrue(_bool_env("TEST_BOOL", False))

    def test_bool_env_false_values(self):
        """Various falsy values should parse correctly."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _bool_env

        for val in ("0", "false", "False", "FALSE", "no", "off"):
            with mock.patch.dict(os.environ, {"TEST_BOOL": val}):
                self.assertFalse(_bool_env("TEST_BOOL", True))

    def test_bool_env_default(self):
        """Missing env var should return default."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _bool_env

        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove the key if it exists
            os.environ.pop("TEST_NONEXISTENT", None)
            self.assertFalse(_bool_env("TEST_NONEXISTENT", False))
            self.assertTrue(_bool_env("TEST_NONEXISTENT", True))

    def test_int_env_valid(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _int_env

        with mock.patch.dict(os.environ, {"TEST_INT": "42"}):
            self.assertEqual(_int_env("TEST_INT", 0), 42)

    def test_int_env_invalid(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _int_env

        with mock.patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            self.assertEqual(_int_env("TEST_INT", 99), 99)

    def test_str_env(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _str_env

        with mock.patch.dict(os.environ, {"TEST_STR": "  hello  "}):
            self.assertEqual(_str_env("TEST_STR", ""), "hello")

    def test_system_env_takes_priority(self):
        """System env var should win over .env defaults."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import _str_env

        with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:9999"}):
            self.assertEqual(_str_env("OLLAMA_BASE_URL", "fallback"), "http://custom:9999")


if __name__ == "__main__":
    unittest.main()
