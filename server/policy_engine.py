"""
Policy Engine

BAML-based policy enforcement for tool access control.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyResult:
    """Result of policy evaluation."""

    allowed: bool
    reason: str
    matched_rule: Optional[str] = None


class PolicyEngine:
    """
    BAML-based policy engine.

    Evaluates access control policies for tool calls.
    """

    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self._load_default_policies()

    def _load_default_policies(self):
        """Load default policies."""
        self.policies = {
            "read_file": {
                "allowed_patterns": ["*.py", "*.js", "*.ts", "*.md", "*.txt", "*.json", "*.yml", "*.yaml"],
                "denied_patterns": [".env*", "*.pem", "*.key", "secrets/*", "credentials/*"],
                "max_file_size_kb": 1000,
            },
            "search_files": {
                "allowed_directories": ["."],
                "denied_patterns": [".git/*", "node_modules/*"],
                "max_results": 100,
            },
            "git_diff": {
                "allowed_refs": ["HEAD", "HEAD~*", "main", "master", "develop"],
                "max_diff_size_kb": 500,
            },
        }

    def evaluate(
        self,
        tool_name: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> PolicyResult:
        """
        Evaluate policy for a tool call.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            user_id: User identifier (for user-specific policies)

        Returns:
            PolicyResult with decision
        """
        policy = self.policies.get(tool_name)

        if not policy:
            # No policy defined - allow by default
            return PolicyResult(allowed=True, reason="No policy defined")

        # Check denied patterns
        path = params.get("path", "") or params.get("file", "") or ""
        denied_patterns = policy.get("denied_patterns", [])

        import fnmatch
        for pattern in denied_patterns:
            if fnmatch.fnmatch(path, pattern):
                logger.warning("policy_denied", tool=tool_name, path=path, pattern=pattern)
                return PolicyResult(
                    allowed=False,
                    reason=f"Path matches denied pattern: {pattern}",
                    matched_rule="denied_patterns",
                )

        # Check allowed patterns (if defined)
        allowed_patterns = policy.get("allowed_patterns", [])
        if allowed_patterns:
            matched = any(fnmatch.fnmatch(path, p) for p in allowed_patterns)
            if not matched:
                return PolicyResult(
                    allowed=False,
                    reason=f"Path doesn't match allowed patterns",
                    matched_rule="allowed_patterns",
                )

        # Additional checks based on tool
        if tool_name == "read_file":
            max_size = policy.get("max_file_size_kb", 1000)
            # Size check would happen during execution

        logger.debug("policy_allowed", tool=tool_name, path=path)
        return PolicyResult(allowed=True, reason="Policy check passed")

    def add_policy(self, tool_name: str, policy: Dict[str, Any]):
        """Add or update a policy."""
        self.policies[tool_name] = policy
        logger.info("policy_added", tool=tool_name)


# Singleton
policy_engine = PolicyEngine()
