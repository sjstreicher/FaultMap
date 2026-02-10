"""Unit tests for config_setup module."""

import tempfile
from pathlib import Path

import pytest

from faultmap.config_setup import ensure_existence


class TestEnsureExistence:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir, "newdir")
            result = ensure_existence(new_dir, make=True)
            assert new_dir.is_dir()
            assert result == new_dir

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_existence(Path(tmpdir), make=True)
            assert result == Path(tmpdir)

    def test_raises_on_missing_no_make(self):
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            ensure_existence(Path("/nonexistent/path/12345"), make=False)

    def test_nested_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir, "a", "b", "c")
            result = ensure_existence(nested, make=True)
            assert nested.is_dir()
            assert result == nested
