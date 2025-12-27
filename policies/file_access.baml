# BAML File Access Policy
# Defines access control for file operations

class FileAccessPolicy {
  # Allowed file patterns for reading
  allowed_read_patterns: list<string> = [
    "*.py",
    "*.js",
    "*.ts",
    "*.json",
    "*.yml",
    "*.yaml",
    "*.md",
    "*.txt",
    "*.html",
    "*.css",
    "*.toml",
    "*.cfg",
    "*.ini",
    "Makefile",
    "Dockerfile",
    "*.sh"
  ]

  # Denied patterns (always blocked)
  denied_patterns: list<string> = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "secrets/*",
    "credentials/*",
    ".git/config",
    "*.secret",
    "id_rsa*"
  ]

  # Maximum file size in KB
  max_file_size_kb: int = 1000

  # Maximum lines to read at once
  max_lines: int = 500
}

function check_file_access(path: string, policy: FileAccessPolicy) -> bool {
  # Check denied patterns first (highest priority)
  for pattern in policy.denied_patterns {
    if glob_match(path, pattern) {
      return false
    }
  }

  # Check allowed patterns
  for pattern in policy.allowed_read_patterns {
    if glob_match(path, pattern) {
      return true
    }
  }

  # Default deny
  return false
}
