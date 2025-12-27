"""
BAML Policy Engine

Pydantic-based policy enforcement for tool requests.
Validates tool inputs and outputs against schemas with type safety.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field, ValidationError

from observability.logging_config import get_logger

logger = get_logger(__name__)


class PolicyDecision(Enum):
    """Policy evaluation result."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"


@dataclass
class PolicyResult:
    """Result of policy evaluation."""

    decision: PolicyDecision
    reason: str
    violations: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.warnings is None:
            self.warnings = []


class ToolPolicy(BaseModel):
    """Policy definition for a tool."""

    tool_name: str
    description: str = ""
    
    # Path restrictions
    allowed_path_patterns: List[str] = Field(default_factory=list)
    blocked_path_patterns: List[str] = Field(default_factory=lambda: [
        r"\.env($|\.)",
        r"\.pem$",
        r"\.key$",
        r"secrets/",
        r"credentials/",
        r"\.git/config",
    ])
    
    # Parameter restrictions
    max_param_length: int = 10000
    required_params: List[str] = Field(default_factory=list)
    
    # Rate limiting
    max_calls_per_minute: int = 60
    
    # Output restrictions
    max_output_size_bytes: int = 100 * 1024  # 100KB
    sanitize_output: bool = True


class PolicyEngine:
    """
    Policy enforcement engine.

    Validates tool requests against defined policies.
    Uses Pydantic for schema validation.
    """

    # Default policies for built-in tools
    DEFAULT_POLICIES = {
        "read_file": ToolPolicy(
            tool_name="read_file",
            description="Read file contents",
            blocked_path_patterns=[
                r"\.env($|\.)",
                r"\.pem$",
                r"\.key$",
                r"secrets/",
                r"credentials/",
                r"\.git/config",
                r"\.ssh/",
                r"\.aws/",
            ],
            required_params=["path"],
        ),
        "search_files": ToolPolicy(
            tool_name="search_files",
            description="Search for patterns in files",
            required_params=["pattern"],
            max_param_length=1000,  # Limit regex length
        ),
        "list_dir": ToolPolicy(
            tool_name="list_dir",
            description="List directory contents",
        ),
        "git_status": ToolPolicy(
            tool_name="git_status",
            description="Get git repository status",
        ),
        "git_diff": ToolPolicy(
            tool_name="git_diff",
            description="Get git diff",
            max_output_size_bytes=500 * 1024,  # Larger for diffs
        ),
        "git_log": ToolPolicy(
            tool_name="git_log",
            description="Get git commit log",
        ),
        "extract_functions": ToolPolicy(
            tool_name="extract_functions",
            description="Extract function definitions",
            required_params=["path"],
        ),
        "extract_classes": ToolPolicy(
            tool_name="extract_classes",
            description="Extract class definitions",
            required_params=["path"],
        ),
        "analyze_imports": ToolPolicy(
            tool_name="analyze_imports",
            description="Analyze imports in a file",
            required_params=["path"],
        ),
    }

    def __init__(self, custom_policies: Optional[Dict[str, ToolPolicy]] = None):
        """
        Initialize policy engine.

        Args:
            custom_policies: Additional or override policies
        """
        self.policies = dict(self.DEFAULT_POLICIES)
        if custom_policies:
            self.policies.update(custom_policies)

        # Compile regex patterns for performance
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for name, policy in self.policies.items():
            self._compiled_patterns[name] = [
                re.compile(p, re.IGNORECASE) for p in policy.blocked_path_patterns
            ]

    def evaluate(
        self,
        tool_name: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        """
        Evaluate a tool request against policies.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            user_id: User making the request
            context: Additional context

        Returns:
            PolicyResult with decision
        """
        violations = []
        warnings = []

        # Get policy for tool
        policy = self.policies.get(tool_name)
        if not policy:
            # No policy defined - allow with warning
            logger.debug("no_policy_defined", tool=tool_name)
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="No policy defined for tool",
                warnings=["No explicit policy defined"],
            )

        # Check required parameters
        for required in policy.required_params:
            if required not in params or params[required] is None:
                violations.append(f"Missing required parameter: {required}")

        # Check parameter length
        for key, value in params.items():
            if isinstance(value, str) and len(value) > policy.max_param_length:
                violations.append(
                    f"Parameter '{key}' exceeds max length ({len(value)} > {policy.max_param_length})"
                )

        # Check path-based restrictions
        path = params.get("path") or params.get("file_path") or params.get("directory")
        if path:
            path_violations = self._check_path_policy(tool_name, path, policy)
            violations.extend(path_violations)

        # Return result
        if violations:
            logger.warning(
                "policy_violation",
                tool=tool_name,
                violations=violations,
                user_id=user_id,
            )
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Policy violations: {', '.join(violations)}",
                violations=violations,
                warnings=warnings,
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="All policies passed",
            violations=[],
            warnings=warnings,
        )

    def _check_path_policy(
        self,
        tool_name: str,
        path: str,
        policy: ToolPolicy,
    ) -> List[str]:
        """Check path against blocked patterns."""
        violations = []
        patterns = self._compiled_patterns.get(tool_name, [])

        for pattern in patterns:
            if pattern.search(path):
                violations.append(f"Path '{path}' matches blocked pattern")
                break

        return violations

    def validate_output(
        self,
        tool_name: str,
        output: Any,
    ) -> PolicyResult:
        """
        Validate tool output against policies.

        Args:
            tool_name: Name of the tool
            output: Tool output to validate

        Returns:
            PolicyResult with decision
        """
        policy = self.policies.get(tool_name)
        if not policy:
            return PolicyResult(decision=PolicyDecision.ALLOW, reason="No output policy")

        warnings = []

        # Check output size
        output_str = str(output)
        if len(output_str.encode()) > policy.max_output_size_bytes:
            warnings.append(
                f"Output exceeds max size ({len(output_str.encode())} > {policy.max_output_size_bytes})"
            )

        # Sanitize if needed
        if policy.sanitize_output:
            # Check for potential secrets in output
            secret_patterns = [
                r"api[_-]?key\s*[:=]\s*['\"]?[\w-]+",
                r"password\s*[:=]\s*['\"]?[\w-]+",
                r"secret\s*[:=]\s*['\"]?[\w-]+",
                r"token\s*[:=]\s*['\"]?[\w-]+",
            ]
            for pattern in secret_patterns:
                if re.search(pattern, output_str, re.IGNORECASE):
                    warnings.append("Potential secret detected in output")
                    break

        if warnings:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason="Output warnings",
                warnings=warnings,
            )

        return PolicyResult(decision=PolicyDecision.ALLOW, reason="Output validated")

    def add_policy(self, policy: ToolPolicy) -> None:
        """Add or update a policy."""
        self.policies[policy.tool_name] = policy
        self._compiled_patterns[policy.tool_name] = [
            re.compile(p, re.IGNORECASE) for p in policy.blocked_path_patterns
        ]
        logger.info("policy_added", tool=policy.tool_name)


# Singleton instance
policy_engine = PolicyEngine()


def evaluate_policy(
    tool_name: str,
    params: Dict[str, Any],
    user_id: Optional[str] = None,
) -> PolicyResult:
    """Convenience function for policy evaluation."""
    return policy_engine.evaluate(tool_name, params, user_id)
