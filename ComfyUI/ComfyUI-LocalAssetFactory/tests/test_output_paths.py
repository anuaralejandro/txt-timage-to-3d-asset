"""
Tests — test_output_paths.py
Tests for safe path construction and directory management.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utilities.file_safety import (
    sanitize_filename,
    ensure_within,
    safe_join,
    safe_asset_dir,
)


class TestSanitizeFilename(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(sanitize_filename("hello_world.txt"), "hello_world.txt")

    def test_special_chars(self):
        result = sanitize_filename("file/with:bad*chars?")
        self.assertNotIn("/", result)
        self.assertNotIn(":", result)
        self.assertNotIn("*", result)

    def test_empty(self):
        self.assertEqual(sanitize_filename(""), "unnamed")

    def test_max_length(self):
        long = "a" * 300
        result = sanitize_filename(long, max_length=50)
        self.assertLessEqual(len(result), 50)


class TestEnsureWithin(unittest.TestCase):
    def test_valid_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inner = os.path.join(tmpdir, "sub", "file.txt")
            os.makedirs(os.path.join(tmpdir, "sub"), exist_ok=True)
            result = ensure_within(inner, tmpdir)
            self.assertTrue(result.startswith(os.path.realpath(tmpdir)))

    def test_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evil = os.path.join(tmpdir, "..", "..", "etc", "passwd")
            with self.assertRaises(ValueError):
                ensure_within(evil, tmpdir)


class TestSafeJoin(unittest.TestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_join(tmpdir, "sub", "file.txt")
            self.assertIn("sub", result)
            self.assertIn("file.txt", result)

    def test_sanitizes_parts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_join(tmpdir, "bad/dir", "evil:file")
            # Should not contain raw special chars
            basename = os.path.basename(result)
            self.assertNotIn(":", basename)


class TestSafeAssetDir(unittest.TestCase):
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_asset_dir(tmpdir, "sword_001", "textures")
            self.assertTrue(os.path.isdir(result))
            self.assertIn("sword_001", result)
            self.assertIn("textures", result)

    def test_within_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_asset_dir(tmpdir, "test_asset")
            resolved = os.path.realpath(result)
            root_resolved = os.path.realpath(tmpdir)
            self.assertTrue(resolved.startswith(root_resolved))


if __name__ == "__main__":
    unittest.main()
