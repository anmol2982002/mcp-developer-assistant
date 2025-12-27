"""
Output Validator

Pydantic-based validation for LLM outputs with secret detection.
Ensures structured, safe, and schema-compliant responses.

Features:
- Pydantic models for all LLM output types
- Secret detection (API keys, passwords, tokens)
- PII detection (emails, phone numbers)
- Schema validation with helpful error messages
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from observability.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# BAML-style Output Schemas (Pydantic implementation)
# =============================================================================


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Severity(str, Enum):
    """Issue severity enumeration."""
    
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ReviewIssueCategory(str, Enum):
    """Review issue category enumeration."""
    
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BUG = "bug"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class SourceCitation(BaseModel):
    """Grounded source citation with file location."""
    
    file_path: str = Field(..., description="Path to source file")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    snippet: Optional[str] = Field(None, max_length=500, description="Code snippet")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="Relevance score")
    
    @model_validator(mode="after")
    def validate_lines(self) -> "SourceCitation":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class CodeIssue(BaseModel):
    """Code review issue."""
    
    file: str = Field(..., description="File path")
    line: Optional[int] = Field(None, ge=1, description="Line number")
    severity: Severity = Field(..., description="Issue severity")
    category: Optional[str] = Field(None, description="Issue category (security, style, etc)")
    message: str = Field(..., min_length=5, max_length=500, description="Issue description")
    suggestion: Optional[str] = Field(None, max_length=500, description="Fix suggestion")


class CodeReviewOutput(BaseModel):
    """Validated code review output schema with ML enhancements."""
    
    summary: str = Field(..., min_length=10, max_length=500, description="One-line summary")
    issues: List[CodeIssue] = Field(default_factory=list, max_length=50, description="Found issues")
    test_suggestions: List[str] = Field(default_factory=list, max_length=20, description="Suggested tests")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    estimated_review_time_minutes: Optional[int] = Field(None, ge=1, le=480, description="Estimated time")
    security_concerns: List[str] = Field(default_factory=list, max_length=10, description="Security issues")
    
    # ML-enhanced fields (Phase 5)
    ml_risk_score: Optional[float] = Field(None, ge=0, le=1, description="ML-predicted risk score")
    ml_confidence: Optional[float] = Field(None, ge=0, le=1, description="ML model confidence")
    risk_factors: List[str] = Field(default_factory=list, max_length=20, description="Contributing risk factors")
    model_id: Optional[str] = Field(None, description="ML model version used")
    
    # Routing information
    review_priority: Optional[str] = Field(None, description="Review priority: critical, high, normal, low")
    requires_security_review: bool = Field(False, description="Requires security team review")
    requires_senior_review: bool = Field(False, description="Requires senior developer review")
    suggested_reviewers: List[str] = Field(default_factory=list, max_length=5, description="Suggested reviewers")
    
    @field_validator("issues", mode="before")
    @classmethod
    def parse_issues(cls, v):
        if isinstance(v, list):
            return [CodeIssue(**item) if isinstance(item, dict) else item for item in v]
        return v
    
    @model_validator(mode='after')
    def validate_security_issues_severity(self) -> "CodeReviewOutput":
        """Validate that security issues are flagged appropriately."""
        # Ensure security issues are tracked if security concerns exist
        if self.security_concerns and not self.requires_security_review:
            # Auto-flag for security review if there are security concerns
            if any(concern.lower() in ['critical', 'high', 'vulnerability', 'injection'] 
                   for concern in self.security_concerns):
                object.__setattr__(self, 'requires_security_review', True)
        return self


class CodeSearchResult(BaseModel):
    """Validated code search result schema."""
    
    answer: str = Field(..., min_length=10, description="Answer to the query")
    confidence: float = Field(..., ge=0, le=1, description="Answer confidence")
    sources: List[SourceCitation] = Field(default_factory=list, max_length=20, description="Source citations")
    related_terms: List[str] = Field(default_factory=list, max_length=10, description="Related search terms")
    
    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, list):
            return [SourceCitation(**item) if isinstance(item, dict) else item for item in v]
        return v


class DiffSummary(BaseModel):
    """Validated diff summary output."""
    
    title: str = Field(..., min_length=5, max_length=100, description="PR title")
    description: str = Field(..., min_length=20, max_length=2000, description="Detailed description")
    changes: List[str] = Field(..., min_length=1, max_length=20, description="List of changes")
    breaking_changes: List[str] = Field(default_factory=list, max_length=10, description="Breaking changes")
    files_affected: int = Field(..., ge=0, description="Number of files affected")
    lines_added: int = Field(0, ge=0, description="Lines added")
    lines_removed: int = Field(0, ge=0, description="Lines removed")


class RepoSummary(BaseModel):
    """Validated repository summary output."""
    
    purpose: str = Field(..., min_length=20, max_length=500, description="Project purpose")
    technologies: List[str] = Field(..., min_length=1, max_length=20, description="Technologies used")
    key_components: List[str] = Field(..., min_length=1, max_length=20, description="Main components")
    getting_started: str = Field(..., min_length=20, max_length=1000, description="Setup instructions")
    architecture: Optional[str] = Field(None, max_length=1000, description="Architecture overview")


class IntentCheckResult(BaseModel):
    """Validated intent check result."""
    
    is_valid: bool = Field(..., description="Whether intent is valid")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reason: str = Field(..., min_length=5, max_length=500, description="Explanation")
    suggested_action: Optional[str] = Field(None, max_length=200, description="Suggested next action")


# =============================================================================
# Secret Detection
# =============================================================================


@dataclass
class SecretPattern:
    """Pattern for detecting secrets."""
    
    name: str
    pattern: str
    replacement: str = "[REDACTED]"


class SecretScanner:
    """
    Scan text for accidentally exposed secrets.
    
    Detects:
    - API keys (AWS, Stripe, OpenAI, etc.)
    - Private keys
    - Passwords in common formats
    - Tokens (JWT, Bearer)
    - Connection strings
    """
    
    PATTERNS: List[SecretPattern] = [
        # API Keys
        SecretPattern("aws_access_key", r"AKIA[0-9A-Z]{16}", "[AWS_KEY_REDACTED]"),
        SecretPattern("aws_secret_key", r"(?i)aws[_-]?secret[_-]?access[_-]?key['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})", "[AWS_SECRET_REDACTED]"),
        SecretPattern("stripe_key", r"sk_live_[0-9a-zA-Z]{24}", "[STRIPE_KEY_REDACTED]"),
        SecretPattern("stripe_test_key", r"sk_test_[0-9a-zA-Z]{24}", "[STRIPE_TEST_KEY_REDACTED]"),
        SecretPattern("openai_key", r"sk-[a-zA-Z0-9]{48}", "[OPENAI_KEY_REDACTED]"),
        SecretPattern("anthropic_key", r"sk-ant-[a-zA-Z0-9-]{40,}", "[ANTHROPIC_KEY_REDACTED]"),
        SecretPattern("github_token", r"ghp_[a-zA-Z0-9]{36}", "[GITHUB_TOKEN_REDACTED]"),
        SecretPattern("gitlab_token", r"glpat-[a-zA-Z0-9-]{20}", "[GITLAB_TOKEN_REDACTED]"),
        SecretPattern("slack_token", r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}", "[SLACK_TOKEN_REDACTED]"),
        SecretPattern("google_api_key", r"AIza[0-9A-Za-z-_]{35}", "[GOOGLE_API_KEY_REDACTED]"),
        
        # Private Keys
        SecretPattern("private_key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "[PRIVATE_KEY_REDACTED]"),
        SecretPattern("ssh_private", r"-----BEGIN OPENSSH PRIVATE KEY-----", "[SSH_KEY_REDACTED]"),
        
        # Passwords
        SecretPattern("password_assignment", r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{4,}['\"]", "[PASSWORD_REDACTED]"),
        SecretPattern("db_password", r"(?i)(db_password|database_password|mysql_password)\s*[:=]\s*['\"][^'\"]+['\"]", "[DB_PASSWORD_REDACTED]"),
        
        # Tokens
        SecretPattern("jwt_token", r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_.+/=]+", "[JWT_REDACTED]"),
        SecretPattern("bearer_token", r"(?i)bearer\s+[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_.+/=]+", "[BEARER_TOKEN_REDACTED]"),
        
        # Connection Strings
        SecretPattern("connection_string", r"(?i)(mongodb|mysql|postgres|redis)://[^\s'\"]+", "[CONNECTION_STRING_REDACTED]"),
        SecretPattern("database_url", r"(?i)database_url\s*=\s*['\"][^'\"]+['\"]", "[DATABASE_URL_REDACTED]"),
        
        # Generic Secrets
        SecretPattern("generic_secret", r"(?i)(secret|api_key|apikey|auth_token)\s*[:=]\s*['\"][^'\"]{8,}['\"]", "[SECRET_REDACTED]"),
    ]
    
    PII_PATTERNS: List[SecretPattern] = [
        SecretPattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
        SecretPattern("phone", r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b", "[PHONE_REDACTED]"),
        SecretPattern("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
        SecretPattern("credit_card", r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CC_REDACTED]"),
        SecretPattern("ip_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]"),
    ]
    
    def __init__(self, scan_pii: bool = True):
        """
        Initialize scanner.
        
        Args:
            scan_pii: Whether to also scan for PII
        """
        self.scan_pii = scan_pii
        self._compiled_patterns = [
            (p.name, re.compile(p.pattern), p.replacement)
            for p in self.PATTERNS
        ]
        if scan_pii:
            self._compiled_patterns.extend([
                (p.name, re.compile(p.pattern), p.replacement)
                for p in self.PII_PATTERNS
            ])
    
    def scan(self, text: str) -> List[Dict[str, Any]]:
        """
        Scan text for secrets.
        
        Args:
            text: Text to scan
            
        Returns:
            List of detected secrets with type and location
        """
        findings = []
        
        for name, pattern, _ in self._compiled_patterns:
            for match in pattern.finditer(text):
                findings.append({
                    "type": name,
                    "start": match.start(),
                    "end": match.end(),
                    "preview": text[max(0, match.start() - 10):match.start()] + "***" + text[match.end():min(len(text), match.end() + 10)],
                })
        
        return findings
    
    def redact(self, text: str) -> str:
        """
        Redact all secrets from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Redacted text
        """
        result = text
        
        for name, pattern, replacement in self._compiled_patterns:
            result = pattern.sub(replacement, result)
        
        return result
    
    def has_secrets(self, text: str) -> bool:
        """Check if text contains secrets."""
        return len(self.scan(text)) > 0


# =============================================================================
# Validation Functions
# =============================================================================


def validate_output(
    response: Union[str, Dict[str, Any]],
    schema: Type[T],
    redact_secrets: bool = True,
) -> T:
    """
    Validate LLM output against schema.
    
    Args:
        response: Raw LLM response (JSON string or dict)
        schema: Pydantic model class to validate against
        redact_secrets: Whether to redact secrets before validation
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If validation fails
    """
    import json
    
    # Parse JSON if string
    if isinstance(response, str):
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", response)
        if json_match:
            response = json_match.group(1)
        
        # Clean up common LLM formatting issues
        response = response.strip()
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("json_parse_failed", error=str(e))
            raise ValidationError.from_exception_data(
                "JSONDecodeError",
                [{"type": "value_error", "msg": f"Invalid JSON: {e}"}],
            )
    else:
        data = response
    
    # Redact secrets from string fields
    if redact_secrets:
        scanner = SecretScanner(scan_pii=True)
        data = _redact_dict_secrets(data, scanner)
    
    # Validate against schema
    return schema.model_validate(data)


def _redact_dict_secrets(data: Any, scanner: SecretScanner) -> Any:
    """Recursively redact secrets from dictionary values."""
    if isinstance(data, dict):
        return {k: _redact_dict_secrets(v, scanner) for k, v in data.items()}
    elif isinstance(data, list):
        return [_redact_dict_secrets(item, scanner) for item in data]
    elif isinstance(data, str):
        return scanner.redact(data)
    return data


def safe_parse_llm_response(
    response: str,
    schema: Type[T],
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Safely parse LLM response with fallback.
    
    Args:
        response: Raw LLM response
        schema: Pydantic model class
        default: Default value if parsing fails
        
    Returns:
        Validated model or default
    """
    try:
        return validate_output(response, schema)
    except (ValidationError, Exception) as e:
        logger.warning("llm_response_parse_failed", schema=schema.__name__, error=str(e))
        return default


def create_validation_prompt(schema: Type[BaseModel]) -> str:
    """
    Create a prompt snippet describing the expected output format.
    
    Args:
        schema: Pydantic model class
        
    Returns:
        Prompt text describing the schema
    """
    schema_json = schema.model_json_schema()
    
    return f"""
Return your response as valid JSON matching this schema:

```json
{schema.model_json_schema()}
```

Important:
- Use the exact field names shown
- Follow the type constraints (string lengths, value ranges)
- Do not include any secrets, API keys, or passwords in your response
"""


# =============================================================================
# Metrics Integration
# =============================================================================


class ValidationMetrics:
    """Track validation metrics."""
    
    def __init__(self):
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.secrets_detected = 0
        self.pii_detected = 0
    
    def record_validation(self, success: bool, secrets_found: int = 0, pii_found: int = 0):
        """Record a validation result."""
        self.total_validations += 1
        
        if success:
            self.successful_validations += 1
        else:
            self.failed_validations += 1
        
        self.secrets_detected += secrets_found
        self.pii_detected += pii_found
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total": self.total_validations,
            "successful": self.successful_validations,
            "failed": self.failed_validations,
            "success_rate": self.successful_validations / max(1, self.total_validations),
            "secrets_detected": self.secrets_detected,
            "pii_detected": self.pii_detected,
        }


validation_metrics = ValidationMetrics()
