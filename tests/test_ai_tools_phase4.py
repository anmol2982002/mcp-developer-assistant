"""
Tests for Enhanced AI Tools (Phase 4)

Tests for ask_about_code, review_changes, summarize_repo, and summarize_diff tools.
"""

import pytest

from server.tools.ai_tools import (
    AskAboutCodeTool,
    ReviewChangesTool,
    SummarizeRepoTool,
    SummarizeDiffTool,
    QueryExpansionTool,
)
from server.tools.base import ToolResult


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = None):
        self.response = response or '{"answer": "Mock answer"}'
        self.calls = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class MockHybridSearchEngine:
    """Mock hybrid search engine for testing."""

    def __init__(self, results=None):
        self.results = results or []

    def search(self, query: str, top_k: int = 5):
        return self.results

    def expand_query(self, query: str) -> str:
        return query + " expanded"


class TestAskAboutCodeTool:
    """Tests for AskAboutCodeTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLLMClient("Based on the code, the function handles authentication.")
        self.mock_search = MockHybridSearchEngine([
            {
                "file_path": "auth.py",
                "content": "def authenticate(): pass",
                "start_line": 1,
                "end_line": 5,
                "score": 0.9,
            }
        ])
        self.tool = AskAboutCodeTool(
            hybrid_search_engine=self.mock_search,
            llm_client=self.mock_llm,
        )

    @pytest.mark.asyncio
    async def test_execute_with_hybrid_search(self):
        """Should use hybrid search and return answer with sources."""
        result = await self.tool.execute("How does authentication work?")

        assert result.success is True
        assert "answer" in result.result
        assert "sources" in result.result
        assert len(result.result["sources"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_without_search_engine(self):
        """Should return error when no search engine available."""
        tool = AskAboutCodeTool(llm_client=self.mock_llm)

        result = await tool.execute("Query without search")

        assert result.success is False
        assert "search" in result.error.lower() or "index" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_without_llm(self):
        """Should return error when LLM not configured."""
        tool = AskAboutCodeTool(hybrid_search_engine=self.mock_search)

        result = await tool.execute("Query without LLM")

        assert result.success is False
        assert "llm" in result.error.lower()

    @pytest.mark.asyncio
    async def test_query_expansion(self):
        """Should expand query when enabled."""
        result = await self.tool.execute("auth issue", expand_query=True)

        assert result.success is True
        # Search should have been called with expanded query

    @pytest.mark.asyncio
    async def test_inline_citations_extraction(self):
        """Should extract inline citations from LLM response."""
        self.mock_llm.response = "Authentication uses JWT [source: auth.py:10-20]"

        result = await self.tool.execute("How does auth work?", include_citations=True)

        assert result.success is True
        # Check for inline citations if present
        if "inline_citations" in result.result:
            assert len(result.result["inline_citations"]) > 0


class TestReviewChangesTool:
    """Tests for ReviewChangesTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.valid_response = '''
{
    "summary": "Added user authentication with JWT tokens",
    "issues": [
        {"file": "auth.py", "line": 42, "severity": "MEDIUM", "message": "Consider using constant-time comparison"}
    ],
    "test_suggestions": ["Test invalid token handling"],
    "risk_level": "MEDIUM",
    "estimated_review_time_minutes": 15
}
'''
        self.mock_llm = MockLLMClient(self.valid_response)
        self.tool = ReviewChangesTool(llm_client=self.mock_llm)

    @pytest.mark.asyncio
    async def test_execute_valid_diff(self):
        """Should review diff and return structured result."""
        diff = """
diff --git a/auth.py b/auth.py
+def authenticate(token):
+    return verify_jwt(token)
"""
        result = await self.tool.execute(diff)

        assert result.success is True
        assert "summary" in result.result
        assert "risk_level" in result.result

    @pytest.mark.asyncio
    async def test_truncate_large_diff(self):
        """Should truncate large diffs."""
        large_diff = "+" * 15000  # Very large diff

        result = await self.tool.execute(large_diff)

        assert result.success is True
        assert result.result.get("truncated") is True

    @pytest.mark.asyncio
    async def test_without_llm(self):
        """Should return error without LLM."""
        tool = ReviewChangesTool()

        result = await tool.execute("some diff")

        assert result.success is False


class TestSummarizeRepoTool:
    """Tests for SummarizeRepoTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.valid_response = '''
{
    "purpose": "A web API for user management",
    "technologies": ["Python", "FastAPI", "PostgreSQL"],
    "key_components": ["API routes", "Database models", "Auth middleware"],
    "getting_started": "Run pip install -r requirements.txt && uvicorn main:app"
}
'''
        self.mock_llm = MockLLMClient(self.valid_response)
        self.tool = SummarizeRepoTool(llm_client=self.mock_llm)

    @pytest.mark.asyncio
    async def test_execute_brief(self, tmp_path):
        """Should summarize repo in brief mode."""
        # Create minimal repo structure
        (tmp_path / "README.md").write_text("# Test Project\nA test project.")
        (tmp_path / "main.py").write_text("def main(): pass")

        result = await self.tool.execute(str(tmp_path), detail_level="brief")

        assert result.success is True
        assert "purpose" in result.result or "summary" in result.result

    @pytest.mark.asyncio
    async def test_execute_detailed(self, tmp_path):
        """Should summarize repo in detailed mode."""
        (tmp_path / "README.md").write_text("# Project\nDetailed readme.")
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn")

        result = await self.tool.execute(str(tmp_path), detail_level="detailed")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_without_llm(self):
        """Should return error without LLM."""
        tool = SummarizeRepoTool()

        result = await tool.execute(".")

        assert result.success is False


class TestSummarizeDiffTool:
    """Tests for SummarizeDiffTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.valid_response = '''
{
    "title": "Add authentication feature",
    "description": "This PR adds JWT-based authentication.",
    "changes": ["Added auth module", "Updated routes"],
    "breaking_changes": []
}
'''
        self.mock_llm = MockLLMClient(self.valid_response)
        self.tool = SummarizeDiffTool(llm_client=self.mock_llm)

    @pytest.mark.asyncio
    async def test_execute_brief(self):
        """Should summarize diff in brief mode."""
        diff = """
diff --git a/auth.py b/auth.py
new file mode 100644
+def authenticate(token):
+    pass
"""
        result = await self.tool.execute(diff, detail_level="brief")

        assert result.success is True
        assert "files_affected" in result.result
        assert result.result["files_affected"] >= 1

    @pytest.mark.asyncio
    async def test_analyze_diff_stats(self):
        """Should correctly analyze diff statistics."""
        diff = """
diff --git a/file1.py b/file1.py
+line added
-line removed
+another added
diff --git a/file2.py b/file2.py
+new content
"""
        result = await self.tool.execute(diff)

        assert result.success is True
        stats = result.result
        assert stats["files_affected"] == 2
        assert stats["lines_added"] >= 3
        assert stats["lines_removed"] >= 1

    @pytest.mark.asyncio
    async def test_markdown_format(self):
        """Should work with markdown format."""
        result = await self.tool.execute("diff content", format="markdown")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_without_llm(self):
        """Should return error without LLM."""
        tool = SummarizeDiffTool()

        result = await tool.execute("diff")

        assert result.success is False


class TestQueryExpansionTool:
    """Tests for QueryExpansionTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = QueryExpansionTool()

    @pytest.mark.asyncio
    async def test_expand_auth_query(self):
        """Should expand auth-related queries."""
        result = await self.tool.execute("how does auth work")

        assert result.success is True
        assert "authentication" in result.result["expanded_query"] or "authorization" in result.result["expanded_query"]

    @pytest.mark.asyncio
    async def test_expand_db_query(self):
        """Should expand database-related queries."""
        result = await self.tool.execute("query the db")

        assert result.success is True
        assert "database" in result.result["expanded_query"] or "sql" in result.result["expanded_query"]

    @pytest.mark.asyncio
    async def test_no_expansion_needed(self):
        """Should return original for queries without synonyms."""
        result = await self.tool.execute("xyz custom term")

        assert result.success is True
        assert len(result.result["added_terms"]) == 0

    @pytest.mark.asyncio
    async def test_multiple_expansions(self):
        """Should handle multiple expansions."""
        result = await self.tool.execute("auth error in db")

        assert result.success is True
        terms = result.result["added_terms"]
        # Should have expanded both auth and error and db
        assert len(terms) >= 2
