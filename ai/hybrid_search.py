"""
Hybrid Search Engine

Combines BM25 keyword search with FAISS semantic search using
Reciprocal Rank Fusion (RRF) for optimal code retrieval.

Features:
- Code-aware AST chunking (function/class boundaries)
- BM25 keyword matching for exact terms
- FAISS semantic search for conceptual similarity
- RRF score fusion for best results
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from observability.logging_config import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "module", "block"
    name: Optional[str] = None
    docstring: Optional[str] = None
    
    @property
    def id(self) -> str:
        """Unique identifier for the chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
    
    def to_searchable_text(self) -> str:
        """Convert to searchable text with metadata."""
        parts = [self.content]
        if self.name:
            parts.insert(0, f"# {self.chunk_type}: {self.name}")
        if self.docstring:
            parts.insert(1, f'"""{self.docstring}"""')
        return "\n".join(parts)


class CodeChunker:
    """
    AST-aware code chunking.
    
    Splits code on semantic boundaries (functions, classes) rather than
    arbitrary character/token counts for better retrieval.
    """
    
    def __init__(self, max_chunk_size: int = 1500, overlap_lines: int = 3):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap_lines: Lines of overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_lines = overlap_lines
    
    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """
        Chunk a file using AST for Python, fallback for others.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of CodeChunk objects
        """
        if file_path.endswith(".py"):
            return self._chunk_python(file_path, content)
        elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
            return self._chunk_javascript(file_path, content)
        else:
            return self._chunk_generic(file_path, content)
    
    def _chunk_python(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk Python file using AST."""
        chunks = []
        lines = content.split("\n")
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning("python_parse_failed", path=file_path)
            return self._chunk_generic(file_path, content)
        
        # Find top-level definitions
        definitions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append({
                    "node": node,
                    "start": node.lineno - 1,
                    "end": node.end_lineno,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                })
        
        if not definitions:
            # No functions/classes, chunk the whole file
            return self._chunk_generic(file_path, content)
        
        # Sort by line number
        definitions.sort(key=lambda x: x["start"])
        
        # Create chunks for module-level code before first definition
        if definitions[0]["start"] > 0:
            module_content = "\n".join(lines[:definitions[0]["start"]])
            if module_content.strip():
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=module_content,
                    start_line=1,
                    end_line=definitions[0]["start"],
                    chunk_type="module",
                    name="imports",
                ))
        
        # Create chunks for each definition
        for defn in definitions:
            chunk_content = "\n".join(lines[defn["start"]:defn["end"]])
            
            # If too large, split further
            if len(chunk_content) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(
                    file_path, chunk_content, defn["start"] + 1, defn["type"], defn["name"]
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=defn["start"] + 1,
                    end_line=defn["end"],
                    chunk_type=defn["type"],
                    name=defn["name"],
                    docstring=defn["docstring"],
                ))
        
        return chunks
    
    def _chunk_javascript(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript using regex patterns."""
        chunks = []
        lines = content.split("\n")
        
        # Patterns for JS/TS functions and classes
        patterns = [
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            (r"(?:export\s+)?class\s+(\w+)", "class"),
            (r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),
            (r"(?:export\s+)?const\s+(\w+)\s*=\s*\{", "object"),
        ]
        
        definitions = []
        for pattern, chunk_type in patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n")
                definitions.append({
                    "start": line_num,
                    "name": match.group(1),
                    "type": chunk_type,
                })
        
        if not definitions:
            return self._chunk_generic(file_path, content)
        
        definitions.sort(key=lambda x: x["start"])
        
        # Estimate end lines (next definition or end of file)
        for i, defn in enumerate(definitions):
            if i + 1 < len(definitions):
                defn["end"] = definitions[i + 1]["start"]
            else:
                defn["end"] = len(lines)
        
        for defn in definitions:
            chunk_content = "\n".join(lines[defn["start"]:defn["end"]])
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=defn["start"] + 1,
                end_line=defn["end"],
                chunk_type=defn["type"],
                name=defn["name"],
            ))
        
        return chunks if chunks else self._chunk_generic(file_path, content)
    
    def _chunk_generic(self, file_path: str, content: str) -> List[CodeChunk]:
        """Fallback chunking by line count."""
        lines = content.split("\n")
        chunks = []
        
        # Chunk size in lines (estimate ~50 chars per line)
        lines_per_chunk = max(10, self.max_chunk_size // 50)
        
        for i in range(0, len(lines), lines_per_chunk - self.overlap_lines):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_content = "\n".join(chunk_lines)
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=i + 1,
                end_line=min(i + lines_per_chunk, len(lines)),
                chunk_type="block",
            ))
        
        return chunks
    
    def _split_large_chunk(
        self,
        file_path: str,
        content: str,
        start_line: int,
        chunk_type: str,
        name: str,
    ) -> List[CodeChunk]:
        """Split a large chunk into smaller pieces."""
        lines = content.split("\n")
        chunks = []
        lines_per_chunk = max(20, self.max_chunk_size // 50)
        
        for i in range(0, len(lines), lines_per_chunk - self.overlap_lines):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_content = "\n".join(chunk_lines)
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=start_line + i,
                end_line=start_line + min(i + lines_per_chunk, len(lines)) - 1,
                chunk_type=chunk_type,
                name=f"{name}_part{len(chunks) + 1}" if name else None,
            ))
        
        return chunks


class BM25Index:
    """
    BM25 (Okapi) keyword search index.
    
    Optimized for code search with specialized tokenization.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.N = 0
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for code search.
        
        Handles camelCase, snake_case, and preserves important symbols.
        """
        # Split camelCase and PascalCase
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Split on underscores
        text = text.replace("_", " ")
        # Remove special characters but keep important ones
        text = re.sub(r"[^\w\s\.]", " ", text)
        # Lowercase and split
        tokens = text.lower().split()
        # Filter very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        return tokens
    
    def index(self, chunks: List[CodeChunk]):
        """
        Build BM25 index from code chunks.
        
        Args:
            chunks: List of CodeChunk objects
        """
        self.corpus = []
        self.doc_ids = []
        self.doc_lengths = []
        
        for chunk in chunks:
            tokens = self.tokenize(chunk.to_searchable_text())
            self.corpus.append(tokens)
            self.doc_ids.append(chunk.id)
            self.doc_lengths.append(len(tokens))
        
        self.N = len(self.corpus)
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Compute document frequencies
        self.doc_freqs = {}
        for doc in self.corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)
        
        # Compute IDF
        self.idf = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
        
        logger.info("bm25_index_built", documents=self.N, vocab_size=len(self.idf))
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search index with query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.N)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            idf = self.idf[token]
            
            for i, doc in enumerate(self.corpus):
                tf = doc.count(token)
                if tf == 0:
                    continue
                
                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf * numerator / denominator
        
        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(self.doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return results


class HybridSearchEngine:
    """
    Hybrid search combining BM25 and FAISS with Reciprocal Rank Fusion.
    
    Provides the best of both worlds:
    - BM25 for exact keyword matching (API names, function names)
    - FAISS for semantic similarity (concepts, documentation)
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            embedding_model_name: Sentence transformer model
            bm25_weight: Weight for BM25 results in fusion
            semantic_weight: Weight for semantic results in fusion
            rrf_k: RRF constant (higher = more weight to lower ranks)
        """
        self.embedding_model_name = embedding_model_name
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k
        
        self.chunker = CodeChunker()
        self.bm25_index = BM25Index()
        self.chunks: List[CodeChunk] = []
        self.chunk_map: Dict[str, CodeChunk] = {}
        
        # Lazy load
        self._model = None
        self._faiss_index = None
    
    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
                logger.info("embedding_model_loaded", model=self.embedding_model_name)
            except ImportError:
                logger.error("sentence_transformers_not_installed")
                raise
        return self._model
    
    def index_codebase(
        self,
        project_root: str,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ) -> int:
        """
        Index a codebase for hybrid search.
        
        Args:
            project_root: Root directory
            file_patterns: Glob patterns to include (default: common code files)
            exclude_patterns: Patterns to exclude
            
        Returns:
            Number of chunks indexed
        """
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.go"]
        
        if exclude_patterns is None:
            exclude_patterns = [
                ".git", "__pycache__", "node_modules", ".venv", "venv",
                "dist", "build", ".next", "*.min.js", "*.min.css"
            ]
        
        root = Path(project_root)
        self.chunks = []
        
        for pattern in file_patterns:
            for file_path in root.rglob(pattern):
                # Skip excluded patterns
                if any(excl in str(file_path) for excl in exclude_patterns):
                    continue
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    file_chunks = self.chunker.chunk_file(str(file_path), content)
                    self.chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning("chunk_file_failed", path=str(file_path), error=str(e))
        
        if not self.chunks:
            logger.warning("no_chunks_created")
            return 0
        
        # Build chunk map
        self.chunk_map = {chunk.id: chunk for chunk in self.chunks}
        
        # Build BM25 index
        self.bm25_index.index(self.chunks)
        
        # Build FAISS index
        self._build_faiss_index()
        
        logger.info(
            "hybrid_index_built",
            chunks=len(self.chunks),
            files=len(set(c.file_path for c in self.chunks)),
        )
        
        return len(self.chunks)
    
    def _build_faiss_index(self):
        """Build FAISS index from chunks."""
        try:
            import faiss
        except ImportError:
            logger.error("faiss_not_installed")
            return
        
        # Embed all chunks
        texts = [chunk.to_searchable_text() for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_candidates: int = 50,
        semantic_candidates: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: BM25 candidate pool size
            semantic_candidates: Semantic candidate pool size
            
        Returns:
            List of search results with scores and metadata
        """
        import time
        start_time = time.time()
        
        # BM25 search
        bm25_results = self.bm25_index.search(query, top_k=bm25_candidates)
        
        # Semantic search
        semantic_results = self._semantic_search(query, top_k=semantic_candidates)
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            top_k=top_k,
        )
        
        # Build result objects
        results = []
        for doc_id, rrf_score, bm25_score, semantic_score in fused_results:
            chunk = self.chunk_map.get(doc_id)
            if chunk:
                results.append({
                    "id": doc_id,
                    "file_path": chunk.file_path,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.name,
                    "score": rrf_score,
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                })
        
        latency = time.time() - start_time
        logger.info(
            "hybrid_search_completed",
            query_len=len(query),
            results=len(results),
            latency_ms=round(latency * 1000, 2),
        )
        
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Perform FAISS semantic search."""
        if self._faiss_index is None:
            return []
        
        try:
            import faiss
            
            # Embed query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype("float32")
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self._faiss_index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append((self.chunks[idx].id, float(score)))
            
            return results
        
        except Exception as e:
            logger.error("semantic_search_failed", error=str(e))
            return []
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        semantic_results: List[Tuple[str, float]],
        top_k: int,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        
        Returns:
            List of (doc_id, rrf_score, bm25_score, semantic_score)
        """
        # Create rank maps
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_results)}
        semantic_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(semantic_results)}
        
        # Create score maps
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        semantic_scores = {doc_id: score for doc_id, score in semantic_results}
        
        # Collect all doc_ids
        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
        
        # Compute RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0
            
            if doc_id in bm25_ranks:
                rrf_score += self.bm25_weight / (self.rrf_k + bm25_ranks[doc_id])
            
            if doc_id in semantic_ranks:
                rrf_score += self.semantic_weight / (self.rrf_k + semantic_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k with all scores
        return [
            (
                doc_id,
                rrf_score,
                bm25_scores.get(doc_id, 0),
                semantic_scores.get(doc_id, 0),
            )
            for doc_id, rrf_score in sorted_results[:top_k]
        ]
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # Common code synonyms
        synonyms = {
            "function": ["func", "method", "def"],
            "class": ["type", "struct", "interface"],
            "error": ["exception", "error", "fail", "bug"],
            "test": ["test", "spec", "unit"],
            "config": ["configuration", "settings", "options"],
            "auth": ["authentication", "authorization", "login"],
            "db": ["database", "sql", "query"],
            "api": ["endpoint", "route", "handler"],
        }
        
        expanded = query
        for term, syns in synonyms.items():
            if term in query.lower():
                expanded += " " + " ".join(syns)
        
        return expanded
    
    def save_index(self, path: str):
        """Save index to disk."""
        import json
        import pickle
        
        index_dir = Path(path)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save BM25 state
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump({
                "corpus": self.bm25_index.corpus,
                "doc_ids": self.bm25_index.doc_ids,
                "doc_lengths": self.bm25_index.doc_lengths,
                "avgdl": self.bm25_index.avgdl,
                "doc_freqs": self.bm25_index.doc_freqs,
                "idf": self.bm25_index.idf,
                "N": self.bm25_index.N,
            }, f)
        
        # Save FAISS index
        if self._faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self._faiss_index, str(index_dir / "faiss.index"))
            except Exception as e:
                logger.error("faiss_save_failed", error=str(e))
        
        logger.info("hybrid_index_saved", path=path)
    
    def load_index(self, path: str) -> bool:
        """Load index from disk."""
        import pickle
        
        index_dir = Path(path)
        
        if not index_dir.exists():
            return False
        
        try:
            # Load chunks
            with open(index_dir / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            self.chunk_map = {chunk.id: chunk for chunk in self.chunks}
            
            # Load BM25 state
            with open(index_dir / "bm25.pkl", "rb") as f:
                bm25_state = pickle.load(f)
            
            self.bm25_index.corpus = bm25_state["corpus"]
            self.bm25_index.doc_ids = bm25_state["doc_ids"]
            self.bm25_index.doc_lengths = bm25_state["doc_lengths"]
            self.bm25_index.avgdl = bm25_state["avgdl"]
            self.bm25_index.doc_freqs = bm25_state["doc_freqs"]
            self.bm25_index.idf = bm25_state["idf"]
            self.bm25_index.N = bm25_state["N"]
            
            # Load FAISS index
            faiss_path = index_dir / "faiss.index"
            if faiss_path.exists():
                try:
                    import faiss
                    self._faiss_index = faiss.read_index(str(faiss_path))
                except Exception as e:
                    logger.error("faiss_load_failed", error=str(e))
            
            logger.info("hybrid_index_loaded", path=path, chunks=len(self.chunks))
            return True
        
        except Exception as e:
            logger.error("index_load_failed", error=str(e))
            return False
