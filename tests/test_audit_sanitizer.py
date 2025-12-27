"""
Tests for Output Sanitizer

Tests for:
- PII hashing
- Secret detection
- Sanitization rules
"""

import pytest
import re

from database.sanitizer import (
    OutputSanitizer,
    SanitizationRule,
    SanitizationType,
    DEFAULT_RULES,
    sanitize_output,
    hash_pii,
    detect_secrets,
)


@pytest.fixture
def sanitizer():
    """Create sanitizer with default rules."""
    return OutputSanitizer()


@pytest.fixture
def sanitizer_with_salt():
    """Create sanitizer with salt for consistent hashing."""
    return OutputSanitizer(salt="test-salt-123")


class TestPIIHashing:
    """Tests for PII hashing."""
    
    def test_hash_pii_email(self, sanitizer):
        """Should hash email addresses."""
        data = "Contact us at user@example.com"
        result = sanitizer.sanitize(data)
        
        assert "user@example.com" not in result
        assert "[HASH:" in result
    
    def test_hash_pii_ipv4(self, sanitizer):
        """Should hash IPv4 addresses."""
        data = "Client IP: 192.168.1.100"
        result = sanitizer.sanitize(data)
        
        assert "192.168.1.100" not in result
        assert "[HASH:" in result
    
    def test_hash_pii_phone(self, sanitizer):
        """Should hash phone numbers."""
        data = "Call me at 555-123-4567"
        result = sanitizer.sanitize(data)
        
        assert "555-123-4567" not in result
        assert "[HASH:" in result
    
    def test_consistent_hashing(self, sanitizer_with_salt):
        """Same input should produce same hash."""
        email = "test@example.com"
        
        hash1 = sanitizer_with_salt.hash_pii(email)
        hash2 = sanitizer_with_salt.hash_pii(email)
        
        assert hash1 == hash2
    
    def test_different_inputs_different_hashes(self, sanitizer_with_salt):
        """Different inputs should produce different hashes."""
        hash1 = sanitizer_with_salt.hash_pii("email1@test.com")
        hash2 = sanitizer_with_salt.hash_pii("email2@test.com")
        
        assert hash1 != hash2


class TestSecretDetection:
    """Tests for secret pattern detection."""
    
    def test_detect_api_key(self, sanitizer):
        """Should detect API keys."""
        data = "api_key=test_fake_key_not_real_abcdefghij123456"
        result = sanitizer.sanitize(data)
        
        assert "test_fake_key" not in result
        assert "[API_KEY_REDACTED]" in result
    
    def test_detect_bearer_token(self, sanitizer):
        """Should detect Bearer tokens."""
        data = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        result = sanitizer.sanitize(data)
        
        assert "eyJ" not in result
        assert "[TOKEN_REDACTED]" in result or "[JWT_REDACTED]" in result
    
    def test_detect_aws_key(self, sanitizer):
        """Should detect AWS access keys."""
        data = "AWS Key: AKIAIOSFODNN7EXAMPLE"
        result = sanitizer.sanitize(data)
        
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[AWS_KEY_REDACTED]" in result
    
    def test_detect_password(self, sanitizer):
        """Should detect password fields."""
        data = "password=mysecretpassword123"
        result = sanitizer.sanitize(data)
        
        assert "mysecretpassword123" not in result
        assert "[PASSWORD_REDACTED]" in result
    
    def test_detect_private_key(self, sanitizer):
        """Should detect private keys."""
        data = """-----BEGIN RSA PRIVATE KEY-----
        MIIEowIBAAKCAQEA...
        -----END RSA PRIVATE KEY-----"""
        result = sanitizer.sanitize(data)
        
        assert "BEGIN RSA PRIVATE KEY" not in result
        assert "[PRIVATE_KEY_REDACTED]" in result
    
    def test_detect_github_token(self, sanitizer):
        """Should detect GitHub tokens."""
        data = "token: ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        result = sanitizer.sanitize(data)
        
        assert "ghp_" not in result
        assert "[GITHUB_TOKEN_REDACTED]" in result
    
    def test_detect_ssn(self, sanitizer):
        """Should detect SSN."""
        data = "SSN: 123-45-6789"
        result = sanitizer.sanitize(data)
        
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result


class TestCreditCardMasking:
    """Tests for credit card masking."""
    
    def test_mask_credit_card(self, sanitizer):
        """Should mask credit card numbers."""
        data = "Card: 4111-1111-1111-1111"
        result = sanitizer.sanitize(data)
        
        # Should be masked, not fully visible
        assert "4111-1111-1111-1111" not in result
    
    def test_mask_credit_card_spaces(self, sanitizer):
        """Should mask credit cards with spaces."""
        data = "Card: 4111 1111 1111 1111"
        result = sanitizer.sanitize(data)
        
        assert "4111 1111 1111 1111" not in result


class TestNestedDataSanitization:
    """Tests for nested data structure sanitization."""
    
    def test_sanitize_dict(self, sanitizer):
        """Should sanitize dictionary values."""
        data = {
            "email": "user@example.com",
            "ip": "192.168.1.1",
            "safe": "normal data",
        }
        result = sanitizer.sanitize(data)
        
        assert "user@example.com" not in str(result)
        assert "192.168.1.1" not in str(result)
        assert result["safe"] == "normal data"
    
    def test_sanitize_list(self, sanitizer):
        """Should sanitize list items."""
        data = ["user@example.com", "normal", "192.168.1.1"]
        result = sanitizer.sanitize(data)
        
        assert "user@example.com" not in str(result)
        assert "normal" in str(result)
    
    def test_sanitize_nested_structure(self, sanitizer):
        """Should sanitize deeply nested structures."""
        data = {
            "user": {
                "contact": {
                    "email": "secret@example.com",
                    "phone": "555-123-4567",
                }
            }
        }
        result = sanitizer.sanitize(data)
        
        assert "secret@example.com" not in str(result)
        assert "555-123-4567" not in str(result)
    
    def test_max_depth_protection(self, sanitizer):
        """Should protect against excessive nesting."""
        # Create deeply nested structure
        data = {"level": 1}
        current = data
        for i in range(15):
            current["nested"] = {"level": i + 2}
            current = current["nested"]
        
        result = sanitizer.sanitize(data)
        
        # Should not crash and should handle depth limit
        assert result is not None


class TestCustomRules:
    """Tests for custom sanitization rules."""
    
    def test_add_custom_rule(self):
        """Should allow adding custom rules."""
        sanitizer = OutputSanitizer(rules=[])
        
        custom_rule = SanitizationRule(
            name="custom_id",
            pattern=re.compile(r"CUSTOM-\d{6}"),
            action=SanitizationType.REDACT,
            replacement="[CUSTOM_REDACTED]",
        )
        sanitizer.add_rule(custom_rule)
        
        result = sanitizer.sanitize("ID: CUSTOM-123456")
        
        assert "CUSTOM-123456" not in result
        assert "[CUSTOM_REDACTED]" in result
    
    def test_remove_rule(self):
        """Should allow removing rules."""
        sanitizer = OutputSanitizer()
        
        # Remove email rule
        sanitizer.remove_rule("email")
        
        result = sanitizer.sanitize("Email: user@example.com")
        
        # Email should not be sanitized now
        assert "user@example.com" in result


class TestDetectSecrets:
    """Tests for secret detection without sanitization."""
    
    def test_detect_multiple_secrets(self, sanitizer):
        """Should detect multiple secret types."""
        data = """
        API Key: api_key=test_fake_key_not_real_abcdefghij
        Email: admin@company.com
        Password: password=secret123
        """
        
        matches = sanitizer.detect_secrets(data)
        
        assert "api_key" in matches
        assert "email" in matches
        assert "password" in matches
    
    def test_detect_no_secrets(self, sanitizer):
        """Should return empty list for clean data."""
        matches = sanitizer.detect_secrets("This is clean data without secrets")
        
        assert len(matches) == 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_sanitize_output_function(self):
        """Should work as convenience function."""
        result = sanitize_output({"email": "test@example.com"})
        
        assert "test@example.com" not in str(result)
    
    def test_hash_pii_function(self):
        """Should hash PII values."""
        result = hash_pii("sensitive-value")
        
        assert result != "sensitive-value"
        assert len(result) == 16  # SHA256 truncated to 16 chars
    
    def test_detect_secrets_function(self):
        """Should detect secrets."""
        matches = detect_secrets("password=secret")
        
        assert "password" in matches
