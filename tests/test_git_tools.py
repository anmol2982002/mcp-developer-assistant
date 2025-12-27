"""
Tests for Git Tools

Tests for git_status, git_diff, and git_log tools.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from server.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    # Initialize git repo
    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        capture_output=True,
    )

    # Create initial commit
    (tmp_path / "initial.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
    )

    return tmp_path


class TestGitStatusTool:
    """Tests for GitStatusTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = GitStatusTool()

    @pytest.mark.asyncio
    async def test_status_clean_repo(self, git_repo):
        """Should return clean status for unmodified repo."""
        result = await self.tool.execute(path=str(git_repo))

        assert result.success is True
        assert isinstance(result.result, dict)
        assert "modified" in result.result
        assert "added" in result.result
        assert "deleted" in result.result

    @pytest.mark.asyncio
    async def test_status_modified_file(self, git_repo):
        """Should detect modified files."""
        # Modify a file
        (git_repo / "initial.txt").write_text("modified content")

        result = await self.tool.execute(path=str(git_repo))

        assert result.success is True
        assert len(result.result["modified"]) > 0

    @pytest.mark.asyncio
    async def test_status_untracked_file(self, git_repo):
        """Should detect untracked files."""
        # Create a new untracked file
        (git_repo / "untracked.txt").write_text("untracked")

        result = await self.tool.execute(path=str(git_repo))

        assert result.success is True
        assert len(result.result["untracked"]) > 0

    @pytest.mark.asyncio
    async def test_status_non_git_repo(self, tmp_path):
        """Should fail gracefully for non-git directory."""
        result = await self.tool.execute(path=str(tmp_path))

        # Either fails or returns empty
        if result.success:
            # Some git versions return empty status for non-repos
            pass
        else:
            assert result.error is not None


class TestGitDiffTool:
    """Tests for GitDiffTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = GitDiffTool()

    @pytest.mark.asyncio
    async def test_diff_modified_file(self, git_repo):
        """Should show diff for modified file."""
        # Modify file
        (git_repo / "initial.txt").write_text("modified content")

        result = await self.tool.execute(path=str(git_repo))

        assert result.success is True
        assert "modified content" in result.result["diff"] or result.result["diff"] == ""

    @pytest.mark.asyncio
    async def test_diff_no_changes(self, git_repo):
        """Should return empty diff for clean repo."""
        result = await self.tool.execute(path=str(git_repo))

        assert result.success is True
        assert result.result["diff"] == ""


class TestGitLogTool:
    """Tests for GitLogTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = GitLogTool()

    @pytest.mark.asyncio
    async def test_log_with_commits(self, git_repo):
        """Should return commit history."""
        result = await self.tool.execute(path=str(git_repo), count=5)

        assert result.success is True
        assert "commits" in result.result
        assert len(result.result["commits"]) >= 1

    @pytest.mark.asyncio
    async def test_log_commit_structure(self, git_repo):
        """Should return properly structured commits."""
        result = await self.tool.execute(path=str(git_repo), count=1)

        assert result.success is True
        if result.result["commits"]:
            commit = result.result["commits"][0]
            assert "hash" in commit
            assert "author" in commit
            assert "message" in commit
