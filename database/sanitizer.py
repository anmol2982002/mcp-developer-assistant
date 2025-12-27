"""
Output Sanitizer

Sanitizes sensitive data in audit logs including:
- PII detection and hashing (emails, IPs, etc.)
- Secret pattern detection (API keys, passwords)
- Configurable sanitization rules
"""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Union

from pydantic import BaseModel

from observability.logging_config import get_logger

logger = get_logger(__name__)


class SanitizationType(str, Enum):
    """Types of sanitization actions."""
    
    HASH = "hash"  # One-way hash
    REDACT = "redact"  # Replace with [REDACTED]
    MASK = "mask"  # Partial mask (e.g., ****@email.com)
    REMOVE = "remove"  # Remove entirely


@dataclass
class SanitizationRule:
    """A rule for sanitizing sensitive data."""
    
    name: str
    pattern: Pattern
    action: SanitizationType
    replacement: str = "[REDACTED]"
    
    def apply(self, value: str, salt: str = "") -> str:
        """Apply this rule to a string value."""
        if not isinstance(value, str):
            return value
        
        def replace_match(match: re.Match) -> str:
            matched = match.group(0)
            
            if self.action == SanitizationType.HASH:
                # One-way hash with optional salt
                hash_input = f"{salt}{matched}".encode()
                return f"[HASH:{hashlib.sha256(hash_input).hexdigest()[:12]}]"
            
            elif self.action == SanitizationType.REDACT:
                return self.replacement
            
            elif self.action == SanitizationType.MASK:
                # Keep first and last chars, mask middle
                if len(matched) <= 4:
                    return "*" * len(matched)
                return matched[0] + "*" * (len(matched) - 2) + matched[-1]
            
            elif self.action == SanitizationType.REMOVE:
                return ""
            
            return matched
        
        return self.pattern.sub(replace_match, value)


# Default sanitization rules
DEFAULT_RULES = [
    # Email addresses
    SanitizationRule(
        name="email",
        pattern=re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        action=SanitizationType.HASH,
    ),
    
    # IP addresses (IPv4)
    SanitizationRule(
        name="ipv4",
        pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        action=SanitizationType.HASH,
    ),
    
    # IPv6 addresses
    SanitizationRule(
        name="ipv6",
        pattern=re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
        action=SanitizationType.HASH,
    ),
    
    # API keys (common patterns)
    SanitizationRule(
        name="api_key",
        pattern=re.compile(r"(?:api[_-]?key|apikey|api_secret)[=:\s]+['\"]?([a-zA-Z0-9_-]{20,})['\"]?", re.IGNORECASE),
        action=SanitizationType.REDACT,
        replacement="[API_KEY_REDACTED]",
    ),
    
    # Bearer tokens
    SanitizationRule(
        name="bearer_token",
        pattern=re.compile(r"Bearer\s+[a-zA-Z0-9._-]+"),
        action=SanitizationType.REDACT,
        replacement="Bearer [TOKEN_REDACTED]",
    ),
    
    # AWS credentials
    SanitizationRule(
        name="aws_key",
        pattern=re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}"),
        action=SanitizationType.REDACT,
        replacement="[AWS_KEY_REDACTED]",
    ),
    
    # AWS secret
    SanitizationRule(
        name="aws_secret",
        pattern=re.compile(r"(?:aws_secret|secret_key)[=:\s]+['\"]?([a-zA-Z0-9/+=]{40})['\"]?", re.IGNORECASE),
        action=SanitizationType.REDACT,
        replacement="[AWS_SECRET_REDACTED]",
    ),
    
    # Password fields
    SanitizationRule(
        name="password",
        pattern=re.compile(r"(?:password|passwd|pwd|secret)[=:\s]+['\"]?([^\s'\"]+)['\"]?", re.IGNORECASE),
        action=SanitizationType.REDACT,
        replacement="[PASSWORD_REDACTED]",
    ),
    
    # Credit card numbers
    SanitizationRule(
        name="credit_card",
        pattern=re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
        action=SanitizationType.MASK,
    ),
    
    # SSN (US)
    SanitizationRule(
        name="ssn",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        action=SanitizationType.REDACT,
        replacement="[SSN_REDACTED]",
    ),
    
    # Phone numbers
    SanitizationRule(
        name="phone",
        pattern=re.compile(r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b"),
        action=SanitizationType.HASH,
    ),
    
    # Private keys
    SanitizationRule(
        name="private_key",
        pattern=re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----"),
        action=SanitizationType.REDACT,
        replacement="[PRIVATE_KEY_REDACTED]",
    ),
    
    # GitHub tokens
    SanitizationRule(
        name="github_token",
        pattern=re.compile(r"gh[pousr]_[a-zA-Z0-9]{36,}"),
        action=SanitizationType.REDACT,
        replacement="[GITHUB_TOKEN_REDACTED]",
    ),
    
    # JWT tokens (simple pattern)
    SanitizationRule(
        name="jwt",
        pattern=re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),
        action=SanitizationType.REDACT,
        replacement="[JWT_REDACTED]",
    ),
]


class OutputSanitizer:
    """
    Sanitizes output data to remove or hash sensitive information.
    
    Features:
    - Configurable sanitization rules
    - PII detection and consistent hashing
    - Secret pattern detection
    - Recursive object traversal
    """
    
    def __init__(
        self,
        rules: Optional[List[SanitizationRule]] = None,
        salt: str = "",
        max_depth: int = 10,
    ):
        """
        Initialize the sanitizer.
        
        Args:
            rules: Sanitization rules (defaults to DEFAULT_RULES)
            salt: Salt for hashing (should be consistent for tracking)
            max_depth: Maximum recursion depth for nested objects
        """
        self.rules = rules or DEFAULT_RULES
        self.salt = salt
        self.max_depth = max_depth
    
    def sanitize(self, data: Any, depth: int = 0) -> Any:
        """
        Sanitize data by applying all rules.
        
        Args:
            data: Data to sanitize (string, dict, list, etc.)
            depth: Current recursion depth
            
        Returns:
            Sanitized data
        """
        if depth > self.max_depth:
            return "[MAX_DEPTH_EXCEEDED]"
        
        if isinstance(data, str):
            return self._sanitize_string(data)
        
        elif isinstance(data, dict):
            return {
                key: self.sanitize(value, depth + 1)
                for key, value in data.items()
            }
        
        elif isinstance(data, list):
            return [self.sanitize(item, depth + 1) for item in data]
        
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        
        else:
            # Convert to string and sanitize
            return self._sanitize_string(str(data))
    
    def _sanitize_string(self, value: str) -> str:
        """Apply all rules to a string."""
        result = value
        
        for rule in self.rules:
            try:
                result = rule.apply(result, self.salt)
            except Exception as e:
                logger.warning(
                    "sanitization_rule_error",
                    rule=rule.name,
                    error=str(e),
                )
        
        return result
    
    def hash_pii(self, value: str) -> str:
        """
        One-way hash for PII while preserving consistency.
        
        Same input always produces same hash (with same salt),
        allowing for analysis without exposing actual values.
        """
        if not value:
            return value
        
        hash_input = f"{self.salt}{value}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def add_rule(self, rule: SanitizationRule):
        """Add a new sanitization rule."""
        self.rules.append(rule)
    
    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != name]
    
    def detect_secrets(self, data: str) -> List[str]:
        """
        Detect potential secrets in data without sanitizing.
        
        Returns list of rule names that matched.
        """
        matches = []
        
        for rule in self.rules:
            if rule.pattern.search(data):
                matches.append(rule.name)
        
        return matches


# Default sanitizer instance
default_sanitizer = OutputSanitizer()


def sanitize_output(data: Any) -> Any:
    """Convenience function for sanitizing data."""
    return default_sanitizer.sanitize(data)


def hash_pii(value: str) -> str:
    """Convenience function for hashing PII."""
    return default_sanitizer.hash_pii(value)


def detect_secrets(data: str) -> List[str]:
    """Convenience function for detecting secrets."""
    return default_sanitizer.detect_secrets(data)
