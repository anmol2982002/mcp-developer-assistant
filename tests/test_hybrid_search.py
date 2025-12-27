"""
Tests for Hybrid Search Engine

Tests the BM25 + FAISS hybrid search with code-aware chunking
and reciprocal rank fusion.
"""

import tempfile
from pathlib import Path

import pytest


class TestCodeChunker:
    """Tests for AST-aware code chunking."""

    def setup_method(self):
        """Setup test fixtures."""
        from ai.hybrid_search import CodeChunker
        self.chunker = CodeChunker(max_chunk_size=1000)

    def test_chunk_python_functions(self):
        """Should create separate chunks for each function."""
        content = '''
def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2

def function_three():
    """Third function."""
    return 3
'''
        chunks = self.chunker.chunk_file("test.py", content)
        
        assert len(chunks) >= 3
        # Each function should be in its own chunk or combined in module
        function_names = [c.name for c in chunks if c.name]
        assert "function_one" in function_names or any("function_one" in c.content for c in chunks)

    def test_chunk_python_class(self):
        """Should chunk classes with their methods."""
        content = '''
class MyClass:
    """A test class."""
    
    def __init__(self):
        self.value = 0
    
    def method_one(self):
        return self.value
    
    def method_two(self):
        return self.value * 2
'''
        chunks = self.chunker.chunk_file("test.py", content)
        
        # Class should be chunked
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1
        assert class_chunks[0].name == "MyClass"

    def test_chunk_generic_file(self):
        """Should fall back to line-based chunking for non-Python files."""
        content = "line 1\n" * 100
        
        chunks = self.chunker.chunk_file("test.txt", content)
        
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "block"

    def test_chunk_empty_file(self):
        """Should handle empty files."""
        chunks = self.chunker.chunk_file("empty.py", "")
        
        # Should return one empty chunk or empty list
        assert len(chunks) <= 1

    def test_chunk_syntax_error(self):
        """Should fall back to generic chunking on syntax errors."""
        content = "def broken(\n  invalid python"
        
        chunks = self.chunker.chunk_file("broken.py", content)
        
        # Should not raise, falls back to generic
        assert len(chunks) >= 1


class TestBM25Index:
    """Tests for BM25 keyword search."""

    def setup_method(self):
        """Setup test fixtures."""
        from ai.hybrid_search import BM25Index, CodeChunk
        self.index = BM25Index()
        self.CodeChunk = CodeChunk

    def test_tokenize_camel_case(self):
        """Should split camelCase tokens."""
        tokens = self.index.tokenize("getUserName")
        
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_tokenize_snake_case(self):
        """Should split snake_case tokens."""
        tokens = self.index.tokenize("get_user_name")
        
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_index_and_search(self):
        """Should index chunks and search."""
        chunks = [
            self.CodeChunk(
                file_path="auth.py",
                content="def authenticate_user(username, password):\n    pass",
                start_line=1,
                end_line=2,
                chunk_type="function",
                name="authenticate_user",
            ),
            self.CodeChunk(
                file_path="db.py",
                content="def connect_database(host, port):\n    pass",
                start_line=1,
                end_line=2,
                chunk_type="function",
                name="connect_database",
            ),
        ]
        
        self.index.index(chunks)
        results = self.index.search("authentication user")
        
        assert len(results) > 0
        # Auth chunk should rank higher
        assert results[0][0] == "auth.py:1-2"

    def test_search_no_results(self):
        """Should return empty for non-matching queries."""
        chunks = [
            self.CodeChunk(
                file_path="test.py",
                content="def foo(): pass",
                start_line=1,
                end_line=1,
                chunk_type="function",
            ),
        ]
        
        self.index.index(chunks)
        results = self.index.search("nonexistent_xyz_term")
        
        # Should return empty or low-scoring
        assert len(results) == 0 or results[0][1] == 0


class TestHybridSearchEngine:
    """Tests for the hybrid search engine."""

    def test_reciprocal_rank_fusion(self):
        """Should correctly fuse BM25 and semantic rankings."""
        from ai.hybrid_search import HybridSearchEngine
        
        engine = HybridSearchEngine(bm25_weight=0.5, semantic_weight=0.5, rrf_k=60)
        
        # Mock results
        bm25_results = [("doc_a", 0.9), ("doc_b", 0.7), ("doc_c", 0.5)]
        semantic_results = [("doc_b", 0.95), ("doc_a", 0.8), ("doc_d", 0.6)]
        
        fused = engine._reciprocal_rank_fusion(bm25_results, semantic_results, top_k=5)
        
        # Both doc_a and doc_b appear in both lists
        doc_ids = [r[0] for r in fused]
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids
        
        # RRF scores should be calculated correctly
        for doc_id, rrf_score, bm25_score, semantic_score in fused:
            assert rrf_score > 0

    def test_query_expansion(self):
        """Should expand queries with synonyms."""
        from ai.hybrid_search import HybridSearchEngine
        
        engine = HybridSearchEngine()
        
        # Test auth expansion
        expanded = engine.expand_query("check auth status")
        assert "authentication" in expanded or "authorization" in expanded
        
        # Test db expansion
        expanded = engine.expand_query("query db")
        assert "database" in expanded or "sql" in expanded

    @pytest.mark.skipif(True, reason="Requires sentence-transformers")
    def test_index_codebase(self, tmp_path):
        """Should index a codebase directory."""
        from ai.hybrid_search import HybridSearchEngine
        
        # Create test files
        (tmp_path / "test1.py").write_text("def hello(): return 'hello'")
        (tmp_path / "test2.py").write_text("def world(): return 'world'")
        
        engine = HybridSearchEngine()
        count = engine.index_codebase(str(tmp_path))
        
        assert count >= 2


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_chunk_id(self):
        """Should generate unique ID."""
        from ai.hybrid_search import CodeChunk
        
        chunk = CodeChunk(
            file_path="test.py",
            content="def foo(): pass",
            start_line=10,
            end_line=20,
            chunk_type="function",
        )
        
        assert chunk.id == "test.py:10-20"

    def test_searchable_text(self):
        """Should include metadata in searchable text."""
        from ai.hybrid_search import CodeChunk
        
        chunk = CodeChunk(
            file_path="test.py",
            content="def foo(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            name="foo",
            docstring="A test function",
        )
        
        text = chunk.to_searchable_text()
        
        assert "foo" in text
        assert "function" in text
        assert "A test function" in text
