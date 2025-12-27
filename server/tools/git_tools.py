"""
Git Tools

Git integration tools: git_status, git_diff, git_log.
"""

import subprocess
from typing import Optional

from observability.logging_config import get_logger
from server.tools.base import BaseTool, ToolParameter, ToolResult

logger = get_logger(__name__)


def run_git_command(args: list, cwd: str = ".") -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except Exception as e:
        return False, str(e)


class GitStatusTool(BaseTool):
    """Get git repository status."""

    name = "git_status"
    description = "Get the current git repository status"
    parameters = [
        ToolParameter(name="path", description="Repository path", type="string", required=False, default="."),
    ]

    async def execute(self, path: str = ".") -> ToolResult:
        """Get git status with branch information."""
        success, output = run_git_command(["status", "--porcelain"], cwd=path)

        if not success:
            logger.error("git_status_failed", error=output)
            return ToolResult(success=False, result=None, error=output)

        # Parse status
        files = {"modified": [], "added": [], "deleted": [], "untracked": []}

        for line in output.splitlines():
            if not line.strip():
                continue
            status = line[:2]
            filename = line[3:]

            if "M" in status:
                files["modified"].append(filename)
            elif "A" in status:
                files["added"].append(filename)
            elif "D" in status:
                files["deleted"].append(filename)
            elif "?" in status:
                files["untracked"].append(filename)

        # Get branch information
        branch_info = {"current": None, "tracking": None, "ahead": 0, "behind": 0}
        
        # Get current branch
        success, branch_output = run_git_command(["branch", "--show-current"], cwd=path)
        if success:
            branch_info["current"] = branch_output.strip()

        # Get tracking info
        success, tracking_output = run_git_command(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=path
        )
        if success:
            branch_info["tracking"] = tracking_output.strip()

            # Get ahead/behind counts
            success, ahead_behind = run_git_command(
                ["rev-list", "--left-right", "--count", "HEAD...@{u}"], cwd=path
            )
            if success and ahead_behind.strip():
                parts = ahead_behind.strip().split()
                if len(parts) >= 2:
                    branch_info["ahead"] = int(parts[0])
                    branch_info["behind"] = int(parts[1])

        result = {
            "files": files,
            "branch": branch_info,
            "is_clean": all(len(v) == 0 for v in files.values()),
        }

        logger.info("git_status", path=path, branch=branch_info["current"])
        return ToolResult(success=True, result=result)


class GitDiffTool(BaseTool):
    """Get git diff."""

    name = "git_diff"
    description = "Get diff of changes"
    parameters = [
        ToolParameter(name="ref", description="Git ref (commit, HEAD~1, etc)", type="string", required=False, default="HEAD"),
        ToolParameter(name="path", description="Repository path", type="string", required=False, default="."),
        ToolParameter(name="file", description="Specific file to diff", type="string", required=False),
    ]

    async def execute(self, ref: str = "HEAD", path: str = ".", file: Optional[str] = None) -> ToolResult:
        """Get git diff."""
        args = ["diff", ref]
        if file:
            args.extend(["--", file])

        success, output = run_git_command(args, cwd=path)

        if not success:
            logger.error("git_diff_failed", error=output)
            return ToolResult(success=False, result=None, error=output)

        # Truncate large diffs
        if len(output) > 50000:
            output = output[:50000] + "\n... (diff truncated)"

        logger.info("git_diff", ref=ref, path=path)
        return ToolResult(success=True, result={"ref": ref, "diff": output})


class GitLogTool(BaseTool):
    """Get git commit log."""

    name = "git_log"
    description = "Get commit history"
    parameters = [
        ToolParameter(name="count", description="Number of commits", type="integer", required=False, default=10),
        ToolParameter(name="path", description="Repository path", type="string", required=False, default="."),
    ]

    async def execute(self, count: int = 10, path: str = ".") -> ToolResult:
        """Get git log."""
        args = ["log", f"-{count}", "--pretty=format:%H|%an|%ae|%s|%ci"]

        success, output = run_git_command(args, cwd=path)

        if not success:
            logger.error("git_log_failed", error=output)
            return ToolResult(success=False, result=None, error=output)

        commits = []
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                commits.append({
                    "hash": parts[0],
                    "author": parts[1],
                    "email": parts[2],
                    "message": parts[3],
                    "date": parts[4],
                })

        logger.info("git_log", path=path, commits=len(commits))
        return ToolResult(success=True, result={"commits": commits})
