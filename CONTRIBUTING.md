# Contributing to MCP Developer Assistant

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/mcp-developer-assistant.git
   cd mcp-developer-assistant
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style
- Add type hints to all functions
- Write docstrings for public APIs
- Add tests for new functionality

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_file_tools.py -v
```

### 4. Run Linting

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

### 5. Validate Project

```bash
python scripts/validate_project.py
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "feat: add amazing feature"
```

Use conventional commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## Code Style

- **Python**: Follow PEP 8, use Black formatter
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google style docstrings
- **Max Line Length**: 100 characters

Example:
```python
def process_request(
    tool_name: str,
    params: Dict[str, Any],
    user_id: Optional[str] = None,
) -> ToolResult:
    """
    Process a tool execution request.

    Args:
        tool_name: Name of the tool to execute
        params: Tool parameters
        user_id: Optional user identifier

    Returns:
        ToolResult with execution outcome

    Raises:
        ToolNotFoundError: If tool doesn't exist
    """
    ...
```

## Testing Guidelines

- Write tests for all new functionality
- Maintain minimum 60% code coverage
- Use pytest fixtures for common setup
- Mock external services (LLM APIs, databases)

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs

## Questions?

Feel free to open an issue for questions or discussions.
