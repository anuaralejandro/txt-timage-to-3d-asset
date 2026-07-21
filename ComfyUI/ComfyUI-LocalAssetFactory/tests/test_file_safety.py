"""
Tests — test_file_safety.py
Tests for file safety, sanitisation, and path traversal prevention.
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
    ensure_directory,
    safe_asset_dir,
)


class TestSanitizeFilename(unittest.TestCase):
    def test_removes_slashes(self):
        self.assertNotIn("/", sanitize_filename("a/b/c"))
        self.assertNotIn("\\", sanitize_filename("a\\b\\c"))

    def test_removes_colons(self):
        self.assertNotIn(":", sanitize_filename("file:name"))

    def test_removes_dots_kept(self):
        # Dots should be kept for extensions
        result = sanitize_filename("file.txt")
        self.assertIn(".", result)

    def test_spaces_replaced(self):
        result = sanitize_filename("my file name")
        self.assertNotIn(" ", result)

    def test_unicode(self):
        result = sanitize_filename("espada_épica")
        # Should not crash, non-ASCII replaced
        self.assertTrue(len(result) > 0)


class TestPathTraversal(unittest.TestCase):
    def test_simple_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                ensure_within(os.path.join(tmpdir, "..", "outside"), tmpdir)

    def test_double_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                ensure_within(os.path.join(tmpdir, "..", "..", "etc"), tmpdir)

    def test_valid_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            os.makedirs(nested, exist_ok=True)
            result = ensure_within(nested, tmpdir)
            self.assertTrue(os.path.isabs(result))


class TestSafeJoin(unittest.TestCase):
    def test_traversal_in_parts_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # safe_join sanitises individual parts, but ".." contains only valid chars
            # (dots are allowed for extensions). The traversal is caught by ensure_within.
            with self.assertRaises(ValueError):
                safe_join(tmpdir, "..", "etc")


class TestEnsureDirectory(unittest.TestCase):
    def test_creates_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "a", "b", "c")
            result = ensure_directory(target)
            self.assertTrue(os.path.isdir(result))


class TestSafeAssetDir(unittest.TestCase):
    def test_creates_asset_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_asset_dir(tmpdir, "hero_sword", "textures", "source")
            self.assertTrue(os.path.isdir(result))
            self.assertIn("hero_sword", result)
            self.assertIn("textures", result)
            self.assertIn("source", result)


if __name__ == "__main__":
    unittest.main()
