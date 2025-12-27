"""
Tests for File Tools

Tests for read_file, search_files, and list_dir tools.
"""

import tempfile
from pathlib import Path

import pytest

from server.tools.file_tools import ListDirTool, ReadFileTool, SearchFilesTool


class TestReadFileTool:
    """Tests for ReadFileTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = ReadFileTool()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        """Should read an existing file successfully."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\nprint('world')")

        result = await self.tool.execute(path=str(test_file))

        assert result.success is True
        assert "hello" in result.result["content"]
        assert result.result["lines"] == 2

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        """Should return error for nonexistent file."""
        result = await self.tool.execute(path="/nonexistent/file.py")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_sensitive_env_file(self, tmp_path):
        """Should block reading .env files."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value")

        result = await self.tool.execute(path=str(env_file))

        assert result.success is False
        assert "sensitive" in result.error.lower() or "denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_max_lines(self, tmp_path):
        """Should truncate output based on max_lines."""
        test_file = tmp_path / "long.py"
        test_file.write_text("\n".join([f"line {i}" for i in range(200)]))

        result = await self.tool.execute(path=str(test_file), max_lines=10)

        assert result.success is True
        assert result.result["lines"] == 10


class TestSearchFilesTool:
    """Tests for SearchFilesTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = SearchFilesTool()

    @pytest.mark.asyncio
    async def test_search_pattern_found(self, tmp_path):
        """Should find matching patterns."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello_world():\n    pass")

        result = await self.tool.execute(
            pattern="hello_world",
            directory=str(tmp_path),
        )

        assert result.success is True
        assert result.result["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_no_match(self, tmp_path):
        """Should return empty when no match found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass")

        result = await self.tool.execute(
            pattern="nonexistent_pattern_xyz",
            directory=str(tmp_path),
        )

        assert result.success is True
        assert result.result["count"] == 0

    @pytest.mark.asyncio
    async def test_search_invalid_regex(self):
        """Should handle invalid regex gracefully."""
        result = await self.tool.execute(
            pattern="[invalid",
            directory=".",
        )

        assert result.success is False
        assert "regex" in result.error.lower() or "invalid" in result.error.lower()


class TestListDirTool:
    """Tests for ListDirTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = ListDirTool()

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        """Should list directory contents."""
        (tmp_path / "file1.py").write_text("test")
        (tmp_path / "file2.py").write_text("test")
        (tmp_path / "subdir").mkdir()

        result = await self.tool.execute(path=str(tmp_path))

        assert result.success is True
        assert result.result["count"] >= 3

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self):
        """Should return error for nonexistent directory."""
        result = await self.tool.execute(path="/nonexistent/directory")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_file_instead_of_dir(self, tmp_path):
        """Should return error when path is a file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("test")

        result = await self.tool.execute(path=str(test_file))

        assert result.success is False
        assert "not a directory" in result.error.lower()
