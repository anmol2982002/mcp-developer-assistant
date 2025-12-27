"""
Generate Test Tokens Script

Create JWT tokens for testing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proxy.oauth_validator import OAuthValidator


def main():
    """Generate test tokens."""
    validator = OAuthValidator()

    # Create test tokens
    tokens = {
        "admin": validator.create_token(
            user_id="admin",
            client_id="test-client",
            scopes=["read", "write", "admin"],
        ),
        "user": validator.create_token(
            user_id="user_123",
            client_id="claude-desktop",
            scopes=["read"],
        ),
        "developer": validator.create_token(
            user_id="dev_456",
            client_id="vscode-extension",
            scopes=["read", "write"],
        ),
    }

    print("Generated test tokens:\n")
    for name, token in tokens.items():
        print(f"{name}:")
        print(f"  {token[:50]}...\n")

    print("Use with: Authorization: Bearer <token>")


if __name__ == "__main__":
    main()
