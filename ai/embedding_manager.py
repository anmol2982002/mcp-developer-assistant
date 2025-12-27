"""
Embedding Manager

Compute and store code embeddings for semantic search.
Uses sentence-transformers for speed, FAISS for efficient search.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from observability.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """
    Manage code embeddings for semantic search.

    Features:
    - Local embedding computation (no API calls)
    - FAISS for efficient similarity search
    - Incremental indexing support
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
    ):
        """
        Initialize embedding manager.

        Args:
            model_name: Sentence-transformers model
            index_path: Path to FAISS index (optional)
        """
        self.model_name = model_name
        self.index_path = index_path
        self.model = None
        self.faiss_index = None
        self.file_registry: Dict[str, Dict] = {}  # id -> {path, chunk_idx, preview}

        self._load_model()

    def _load_model(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)
        except ImportError:
            logger.warning("sentence_transformers_not_installed")
        except Exception as e:
            logger.error("embedding_model_load_failed", error=str(e))

    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.

        Args:
            text: Input text

        Returns:
            Numpy array embedding
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

    def index_codebase(
        self,
        project_root: str,
        file_patterns: List[str] = None,
        chunk_size: int = 500,
    ):
        """
        Pre-compute embeddings for all files in project.

        Args:
            project_root: Root directory
            file_patterns: Glob patterns (default: ["*.py"])
            chunk_size: Tokens per chunk
        """
        if file_patterns is None:
            file_patterns = ["*.py"]

        try:
            import faiss
        except ImportError:
            logger.error("faiss_not_installed")
            return

        embeddings = []
        file_ids = []
        root = Path(project_root)

        for pattern in file_patterns:
            for file_path in root.rglob(pattern):
                if self._should_skip(file_path):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    chunks = self._chunk_code(content, chunk_size=chunk_size)

                    for chunk_idx, chunk in enumerate(chunks):
                        embedding = self.embed(chunk)
                        embeddings.append(embedding)

                        file_id = f"{file_path}#{chunk_idx}"
                        file_ids.append(file_id)
                        self.file_registry[file_id] = {
                            "path": str(file_path),
                            "chunk_idx": chunk_idx,
                            "preview": chunk[:100],
                        }
                except Exception as e:
                    logger.warning("embedding_file_failed", path=str(file_path), error=str(e))

        if not embeddings:
            logger.warning("no_files_indexed")
            return

        # Build FAISS index
        embeddings_array = np.array(embeddings).astype("float32")
        dimension = embeddings_array.shape[1]

        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)

        logger.info(
            "codebase_indexed",
            chunks=len(embeddings),
            files=len(set(r["path"] for r in self.file_registry.values())),
        )

        # Save index if path provided
        if self.index_path:
            self._save_index()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[str]:
        """
        Find most similar code chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of file paths
        """
        if self.faiss_index is None:
            logger.warning("faiss_index_not_loaded")
            return []

        try:
            import faiss

            query = np.array([query_embedding]).astype("float32")
            distances, indices = self.faiss_index.search(query, top_k)

            results = []
            file_ids = list(self.file_registry.keys())

            for idx in indices[0]:
                if idx < len(file_ids):
                    file_id = file_ids[idx]
                    results.append(self.file_registry[file_id]["path"])

            return list(dict.fromkeys(results))  # Unique, preserve order

        except Exception as e:
            logger.error("search_failed", error=str(e))
            return []

    def _chunk_code(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split code into semantic chunks.

        Tries to split on function/class boundaries when possible.
        """
        # Simple token-based chunking for now
        # TODO: Implement AST-aware chunking for better semantics
        tokens = text.split()
        chunks = []
        current_chunk = []

        for token in tokens:
            current_chunk.append(token)
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _should_skip(self, path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [".git", "__pycache__", "node_modules", ".venv", "venv"]
        return any(p in str(path) for p in skip_patterns)

    def _save_index(self):
        """Save FAISS index to disk."""
        if self.faiss_index and self.index_path:
            try:
                import faiss
                import json

                Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.faiss_index, self.index_path)

                # Save registry
                registry_path = self.index_path + ".registry.json"
                with open(registry_path, "w") as f:
                    json.dump(self.file_registry, f)

                logger.info("index_saved", path=self.index_path)

            except Exception as e:
                logger.error("index_save_failed", error=str(e))

    def _load_index(self):
        """Load FAISS index from disk."""
        if self.index_path and Path(self.index_path).exists():
            try:
                import faiss
                import json

                self.faiss_index = faiss.read_index(self.index_path)

                registry_path = self.index_path + ".registry.json"
                if Path(registry_path).exists():
                    with open(registry_path) as f:
                        self.file_registry = json.load(f)

                logger.info("index_loaded", path=self.index_path)

            except Exception as e:
                logger.error("index_load_failed", error=str(e))
