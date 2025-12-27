"""
Pytest Configuration and Fixtures
"""

import asyncio
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_tool_request():
    """Sample tool request for testing."""
    return {
        "tool_name": "read_file",
        "params": {"path": "test.py", "max_lines": 100},
        "user_stated_intent": "Review the test file",
        "client_id": "test-client",
    }


@pytest.fixture
def sample_user_history():
    """Sample user history for anomaly detection tests."""
    from datetime import datetime, timedelta

    base_time = datetime.utcnow()
    return [
        {"tool": "read_file", "timestamp": base_time - timedelta(hours=1), "ip": "192.168.1.1"},
        {"tool": "git_status", "timestamp": base_time - timedelta(minutes=30), "ip": "192.168.1.1"},
        {"tool": "search_files", "timestamp": base_time - timedelta(minutes=10), "ip": "192.168.1.1"},
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM client for testing."""

    class MockLLM:
        async def generate(self, prompt: str) -> str:
            return '{"is_valid": true, "confidence": 0.95, "reason": "Mock response"}'

    return MockLLM()


@pytest.fixture
def test_file_content():
    """Sample Python file content for testing."""
    return '''
"""Test module."""

import os
from typing import List

def hello_world():
    """Say hello."""
    return "Hello, World!"

class Greeter:
    """Greeter class."""

    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"
'''
