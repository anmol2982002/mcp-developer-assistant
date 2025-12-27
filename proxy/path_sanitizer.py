"""
Path Sanitizer

Centralized path validation and sanitization to prevent directory traversal,
symlink attacks, and access to sensitive files.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from observability.logging_config import get_logger
from server.config import get_server_config

logger = get_logger(__name__)


class PathSanitizer:
    """
    Path sanitization and validation.

    Security features:
    - Directory traversal prevention
    - Symlink resolution
    - Allowlist/blocklist enforcement
    - Sensitive file detection
    """

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        excluded_patterns: Optional[List[str]] = None,
        sensitive_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize path sanitizer.

        Args:
            allowed_paths: List of allowed base directories
            excluded_patterns: Patterns to exclude (e.g., .git, node_modules)
            sensitive_patterns: Patterns for sensitive files (e.g., .env, *.pem)
        """
        config = get_server_config()
        self.allowed_paths = allowed_paths or config.allowed_paths
        self.excluded_patterns = excluded_patterns or config.excluded_patterns
        self.sensitive_patterns = sensitive_patterns or config.sensitive_patterns

    def sanitize_path(self, path: str, base_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Sanitize and validate a path.

        Args:
            path: Path to sanitize
            base_dir: Optional base directory for relative paths

        Returns:
            Tuple of (is_valid, sanitized_path, error_message)
        """
        try:
            # Handle relative paths
            if base_dir:
                full_path = Path(base_dir) / path
            else:
                full_path = Path(path)

            # Resolve to absolute path (handles .., symlinks, etc.)
            resolved_path = full_path.resolve()

            # Check if path is within allowed directories
            if not self.is_allowed_path(str(resolved_path)):
                return False, str(resolved_path), "Path not in allowed directories"

            # Check for excluded patterns
            if self.is_excluded_path(str(resolved_path)):
                return False, str(resolved_path), "Path matches excluded pattern"

            # Check for sensitive patterns
            if self.is_sensitive_path(str(resolved_path)):
                return False, str(resolved_path), "Path matches sensitive pattern"

            return True, str(resolved_path), None

        except Exception as e:
            logger.error("path_sanitize_error", path=path, error=str(e))
            return False, path, f"Path validation error: {e}"

    def is_allowed_path(self, path: str) -> bool:
        """
        Check if path is within allowed directories.

        Args:
            path: Absolute path to check

        Returns:
            True if path is allowed
        """
        try:
            resolved = Path(path).resolve()

            for allowed in self.allowed_paths:
                allowed_resolved = Path(allowed).resolve()
                try:
                    resolved.relative_to(allowed_resolved)
                    return True
                except ValueError:
                    continue

            return False

        except Exception as e:
            logger.warning("is_allowed_path_error", path=path, error=str(e))
            return False

    def is_excluded_path(self, path: str) -> bool:
        """
        Check if path matches excluded patterns.

        Args:
            path: Path to check

        Returns:
            True if path should be excluded
        """
        path_lower = path.lower()

        for pattern in self.excluded_patterns:
            pattern_lower = pattern.lower()

            # Check if pattern appears anywhere in path
            if pattern_lower in path_lower:
                return True

            # Check path components
            path_parts = Path(path).parts
            for part in path_parts:
                if part.lower() == pattern_lower:
                    return True

        return False

    def is_sensitive_path(self, path: str) -> bool:
        """
        Check if path matches sensitive file patterns.

        Args:
            path: Path to check

        Returns:
            True if path is sensitive
        """
        import fnmatch

        path_lower = path.lower()
        filename = Path(path).name.lower()

        for pattern in self.sensitive_patterns:
            pattern_lower = pattern.lower()

            # Match against filename
            if fnmatch.fnmatch(filename, pattern_lower):
                logger.debug("sensitive_path_match", path=path, pattern=pattern)
                return True

            # Match against full path
            if fnmatch.fnmatch(path_lower, f"*{pattern_lower}*"):
                logger.debug("sensitive_path_match", path=path, pattern=pattern)
                return True

        return False

    def normalize_path(self, path: str) -> str:
        """
        Normalize path for consistent handling.

        Args:
            path: Path to normalize

        Returns:
            Normalized path string
        """
        return str(Path(path).resolve())

    def get_safe_path(self, path: str, base_dir: Optional[str] = None) -> Optional[str]:
        """
        Get a safe, sanitized path or None if invalid.

        Args:
            path: Path to sanitize
            base_dir: Optional base directory

        Returns:
            Sanitized path or None if invalid
        """
        is_valid, sanitized, error = self.sanitize_path(path, base_dir)

        if not is_valid:
            logger.warning("safe_path_denied", path=path, reason=error)
            return None

        return sanitized


# Singleton instance
path_sanitizer = PathSanitizer()


def sanitize_path(path: str, base_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """Convenience function for path sanitization."""
    return path_sanitizer.sanitize_path(path, base_dir)


def is_sensitive_path(path: str) -> bool:
    """Convenience function to check if path is sensitive."""
    return path_sanitizer.is_sensitive_path(path)


def get_safe_path(path: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Convenience function to get safe path."""
    return path_sanitizer.get_safe_path(path, base_dir)
