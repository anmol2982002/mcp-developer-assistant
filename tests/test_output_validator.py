"""
Tests for Output Validator

Tests Pydantic schema validation and secret detection for LLM outputs.
"""

import pytest

from ai.output_validator import (
    CodeReviewOutput,
    CodeSearchResult,
    DiffSummary,
    CodeIssue,
    SourceCitation,
    SecretScanner,
    validate_output,
    safe_parse_llm_response,
    RiskLevel,
    Severity,
)


class TestCodeReviewOutput:
    """Tests for CodeReviewOutput schema."""

    def test_valid_review(self):
        """Should validate a correct review output."""
        data = {
            "summary": "This is a valid summary of the code changes made.",
            "issues": [
                {
                    "file": "test.py",
                    "line": 42,
                    "severity": "HIGH",
                    "category": "security",
                    "message": "SQL injection vulnerability detected.",
                }
            ],
            "test_suggestions": ["Test authentication flow", "Test edge cases"],
            "risk_level": "HIGH",
            "estimated_review_time_minutes": 30,
            "security_concerns": ["SQL injection in user input"],
        }
        
        review = CodeReviewOutput.model_validate(data)
        
        assert review.summary == data["summary"]
        assert review.risk_level == RiskLevel.HIGH
        assert len(review.issues) == 1
        assert review.issues[0].severity == Severity.HIGH

    def test_minimal_valid_review(self):
        """Should validate minimal required fields."""
        data = {
            "summary": "Minimal valid review summary here.",
            "risk_level": "LOW",
        }
        
        review = CodeReviewOutput.model_validate(data)
        
        assert review.risk_level == RiskLevel.LOW
        assert review.issues == []

    def test_invalid_risk_level(self):
        """Should reject invalid risk level."""
        data = {
            "summary": "Valid summary here with enough chars.",
            "risk_level": "INVALID",
        }
        
        with pytest.raises(Exception):
            CodeReviewOutput.model_validate(data)

    def test_summary_too_short(self):
        """Should reject too-short summary."""
        data = {
            "summary": "Short",  # Less than 10 chars
            "risk_level": "LOW",
        }
        
        with pytest.raises(Exception):
            CodeReviewOutput.model_validate(data)


class TestCodeSearchResult:
    """Tests for CodeSearchResult schema."""

    def test_valid_search_result(self):
        """Should validate a correct search result."""
        data = {
            "answer": "This is the answer to your question about the code.",
            "confidence": 0.85,
            "sources": [
                {
                    "file_path": "auth.py",
                    "start_line": 10,
                    "end_line": 25,
                    "snippet": "def authenticate()...",
                }
            ],
        }
        
        result = CodeSearchResult.model_validate(data)
        
        assert result.confidence == 0.85
        assert len(result.sources) == 1

    def test_invalid_confidence(self):
        """Should reject confidence outside 0-1 range."""
        data = {
            "answer": "Valid answer text.",
            "confidence": 1.5,  # Invalid
        }
        
        with pytest.raises(Exception):
            CodeSearchResult.model_validate(data)


class TestSourceCitation:
    """Tests for SourceCitation schema."""

    def test_valid_citation(self):
        """Should validate correct citation."""
        data = {
            "file_path": "src/utils.py",
            "start_line": 10,
            "end_line": 20,
        }
        
        citation = SourceCitation.model_validate(data)
        
        assert citation.file_path == "src/utils.py"
        assert citation.start_line == 10

    def test_invalid_line_range(self):
        """Should reject end_line < start_line."""
        data = {
            "file_path": "test.py",
            "start_line": 20,
            "end_line": 10,  # Invalid
        }
        
        with pytest.raises(Exception):
            SourceCitation.model_validate(data)


class TestSecretScanner:
    """Tests for secret detection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.scanner = SecretScanner(scan_pii=True)

    def test_detect_aws_key(self):
        """Should detect AWS access keys."""
        text = "The key is AKIAIOSFODNN7EXAMPLE for the account."
        
        findings = self.scanner.scan(text)
        
        assert len(findings) > 0
        assert any(f["type"] == "aws_access_key" for f in findings)

    def test_detect_openai_key(self):
        """Should detect OpenAI API keys."""
        text = "export OPENAI_API_KEY=sk-1234567890abcdefghijklmnopqrstuvwxyz123456789012"
        
        findings = self.scanner.scan(text)
        
        assert len(findings) > 0
        assert any(f["type"] == "openai_key" for f in findings)

    def test_detect_jwt(self):
        """Should detect JWT tokens."""
        text = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        
        findings = self.scanner.scan(text)
        
        assert len(findings) > 0
        assert any("jwt" in f["type"].lower() for f in findings)

    def test_detect_password(self):
        """Should detect password assignments."""
        text = 'password = "super_secret_123"'
        
        findings = self.scanner.scan(text)
        
        assert len(findings) > 0

    def test_detect_email(self):
        """Should detect email addresses."""
        text = "Contact me at user@example.com for more info."
        
        findings = self.scanner.scan(text)
        
        assert len(findings) > 0
        assert any(f["type"] == "email" for f in findings)

    def test_redact_secrets(self):
        """Should redact all secrets from text."""
        text = "The API key is AKIAIOSFODNN7EXAMPLE and email is user@example.com"
        
        redacted = self.scanner.redact(text)
        
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[AWS_KEY_REDACTED]" in redacted
        assert "user@example.com" not in redacted

    def test_no_secrets(self):
        """Should return empty for clean text."""
        text = "This is just regular code with no secrets."
        
        findings = self.scanner.scan(text)
        
        assert len(findings) == 0

    def test_has_secrets(self):
        """Should correctly report presence of secrets."""
        clean = "Regular text"
        dirty = "password = 'secret123'"
        
        assert not self.scanner.has_secrets(clean)
        assert self.scanner.has_secrets(dirty)


class TestValidateOutput:
    """Tests for validate_output function."""

    def test_validate_json_string(self):
        """Should parse and validate JSON string."""
        json_str = '{"summary": "Valid summary text here.", "risk_level": "LOW"}'
        
        result = validate_output(json_str, CodeReviewOutput)
        
        assert result.risk_level == RiskLevel.LOW

    def test_validate_dict(self):
        """Should validate dict directly."""
        data = {"summary": "Valid summary text here.", "risk_level": "MEDIUM"}
        
        result = validate_output(data, CodeReviewOutput)
        
        assert result.risk_level == RiskLevel.MEDIUM

    def test_validate_with_markdown_code_block(self):
        """Should extract JSON from markdown code blocks."""
        response = '''
Here is the review:

```json
{"summary": "Valid summary in code block.", "risk_level": "HIGH"}
```
        '''
        
        result = validate_output(response, CodeReviewOutput)
        
        assert result.risk_level == RiskLevel.HIGH

    def test_redact_secrets_in_validation(self):
        """Should redact secrets during validation."""
        data = {
            "summary": "Found password = 'secret123' in the code.",
            "risk_level": "HIGH",
        }
        
        result = validate_output(data, CodeReviewOutput, redact_secrets=True)
        
        assert "secret123" not in result.summary
        assert "[PASSWORD_REDACTED]" in result.summary

    def test_invalid_json(self):
        """Should raise on invalid JSON."""
        invalid = "not valid json {"
        
        with pytest.raises(Exception):
            validate_output(invalid, CodeReviewOutput)


class TestSafeParseResponse:
    """Tests for safe_parse_llm_response function."""

    def test_returns_default_on_error(self):
        """Should return default on parse error."""
        invalid = "invalid json"
        
        result = safe_parse_llm_response(invalid, CodeReviewOutput, default=None)
        
        assert result is None

    def test_returns_parsed_on_success(self):
        """Should return parsed result on success."""
        valid = '{"summary": "Valid summary text.", "risk_level": "LOW"}'
        
        result = safe_parse_llm_response(valid, CodeReviewOutput)
        
        assert result is not None
        assert result.risk_level == RiskLevel.LOW


class TestDiffSummary:
    """Tests for DiffSummary schema."""

    def test_valid_diff_summary(self):
        """Should validate correct diff summary."""
        data = {
            "title": "Add user authentication",
            "description": "This PR adds JWT-based authentication to the API endpoints.",
            "changes": ["Added auth middleware", "Created JWT utilities"],
            "breaking_changes": [],
            "files_affected": 5,
            "lines_added": 150,
            "lines_removed": 10,
        }
        
        summary = DiffSummary.model_validate(data)
        
        assert summary.files_affected == 5
        assert len(summary.changes) == 2
