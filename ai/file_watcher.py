"""
File Watcher for Incremental Indexing

Watch for file changes and incrementally update the search index.
Uses watchdog for cross-platform file system events.

Features:
- Real-time file change detection
- Debounced updates to avoid excessive re-indexing
- Index versioning with model compatibility checks
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IndexVersion:
    """Track index version and model compatibility."""
    
    version: str
    model_name: str
    model_version: str
    created_at: str
    updated_at: str
    file_count: int
    chunk_count: int
    file_hashes: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "file_count": self.file_count,
            "chunk_count": self.chunk_count,
            "file_hashes": self.file_hashes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            model_name=data["model_name"],
            model_version=data.get("model_version", "unknown"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            file_count=data["file_count"],
            chunk_count=data["chunk_count"],
            file_hashes=data.get("file_hashes", {}),
        )
    
    def is_compatible(self, model_name: str) -> bool:
        """Check if index is compatible with given model."""
        return self.model_name == model_name
    
    def save(self, path: str):
        """Save version to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> Optional["IndexVersion"]:
        """Load version from file."""
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return None


class FileHasher:
    """Compute and cache file hashes for change detection."""
    
    def __init__(self):
        self.hashes: Dict[str, str] = {}
    
    @staticmethod
    def hash_file(path: str) -> str:
        """Compute MD5 hash of file content."""
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except (FileNotFoundError, PermissionError):
            return ""
    
    def update(self, path: str) -> bool:
        """
        Update hash for file.
        
        Returns:
            True if file changed, False otherwise
        """
        new_hash = self.hash_file(path)
        old_hash = self.hashes.get(path)
        
        if new_hash != old_hash:
            self.hashes[path] = new_hash
            return True
        return False
    
    def remove(self, path: str):
        """Remove file from hash cache."""
        self.hashes.pop(path, None)
    
    def get_changed_files(self, paths: List[str]) -> List[str]:
        """Get list of files that have changed."""
        return [p for p in paths if self.update(p)]


class IncrementalIndexer:
    """
    Incremental indexing manager.
    
    Tracks file changes and updates only modified portions of the index.
    """
    
    def __init__(
        self,
        hybrid_search_engine,
        index_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize incremental indexer.
        
        Args:
            hybrid_search_engine: HybridSearchEngine instance
            index_path: Path to store index files
            model_name: Embedding model name for versioning
        """
        self.engine = hybrid_search_engine
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.hasher = FileHasher()
        self.version: Optional[IndexVersion] = None
        
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self, project_root: str, file_patterns: List[str] = None) -> bool:
        """
        Initialize or load existing index.
        
        Args:
            project_root: Project root directory
            file_patterns: File patterns to index
            
        Returns:
            True if index was loaded, False if created new
        """
        version_path = self.index_path / "version.json"
        self.version = IndexVersion.load(str(version_path))
        
        if self.version and self.version.is_compatible(self.model_name):
            # Try to load existing index
            if self.engine.load_index(str(self.index_path)):
                self.hasher.hashes = self.version.file_hashes.copy()
                logger.info("loaded_existing_index", version=self.version.version)
                return True
        
        # Create new index
        self._create_new_index(project_root, file_patterns)
        return False
    
    def _create_new_index(self, project_root: str, file_patterns: List[str] = None):
        """Create a new index from scratch."""
        chunk_count = self.engine.index_codebase(project_root, file_patterns)
        
        # Update hashes for all indexed files
        for chunk in self.engine.chunks:
            if chunk.file_path not in self.hasher.hashes:
                self.hasher.update(chunk.file_path)
        
        # Create version
        self.version = IndexVersion(
            version="1.0",
            model_name=self.model_name,
            model_version="1.0",
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            file_count=len(set(c.file_path for c in self.engine.chunks)),
            chunk_count=chunk_count,
            file_hashes=self.hasher.hashes.copy(),
        )
        
        # Save index and version
        self.engine.save_index(str(self.index_path))
        self.version.save(str(self.index_path / "version.json"))
        
        logger.info("created_new_index", chunks=chunk_count)
    
    def update_file(self, file_path: str) -> bool:
        """
        Update index for a single file.
        
        Args:
            file_path: Path to changed file
            
        Returns:
            True if index was updated
        """
        if not self.hasher.update(file_path):
            return False  # File hasn't changed
        
        logger.info("updating_file_index", path=file_path)
        
        # Remove old chunks for this file
        self.engine.chunks = [c for c in self.engine.chunks if c.file_path != file_path]
        
        # Add new chunks
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            new_chunks = self.engine.chunker.chunk_file(file_path, content)
            self.engine.chunks.extend(new_chunks)
            
            # Rebuild chunk map
            self.engine.chunk_map = {chunk.id: chunk for chunk in self.engine.chunks}
            
            # Rebuild indices
            self.engine.bm25_index.index(self.engine.chunks)
            self.engine._build_faiss_index()
            
            # Update version
            if self.version:
                self.version.updated_at = datetime.utcnow().isoformat()
                self.version.file_hashes = self.hasher.hashes.copy()
                self.version.chunk_count = len(self.engine.chunks)
                self.version.save(str(self.index_path / "version.json"))
            
            return True
        
        except Exception as e:
            logger.error("file_update_failed", path=file_path, error=str(e))
            return False
    
    def remove_file(self, file_path: str) -> bool:
        """
        Remove file from index.
        
        Args:
            file_path: Path to removed file
            
        Returns:
            True if file was removed
        """
        self.hasher.remove(file_path)
        
        old_count = len(self.engine.chunks)
        self.engine.chunks = [c for c in self.engine.chunks if c.file_path != file_path]
        
        if len(self.engine.chunks) < old_count:
            # Rebuild indices
            self.engine.chunk_map = {chunk.id: chunk for chunk in self.engine.chunks}
            self.engine.bm25_index.index(self.engine.chunks)
            self.engine._build_faiss_index()
            
            logger.info("removed_file_from_index", path=file_path)
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "version": self.version.version if self.version else None,
            "model": self.model_name,
            "file_count": len(set(c.file_path for c in self.engine.chunks)),
            "chunk_count": len(self.engine.chunks),
            "created_at": self.version.created_at if self.version else None,
            "updated_at": self.version.updated_at if self.version else None,
        }


class FileWatcher:
    """
    Watch for file changes using watchdog.
    
    Provides debounced callbacks for file create/modify/delete events.
    """
    
    def __init__(
        self,
        project_root: str,
        on_change: Callable[[str, str], None],
        file_patterns: List[str] = None,
        debounce_seconds: float = 1.0,
    ):
        """
        Initialize file watcher.
        
        Args:
            project_root: Directory to watch
            on_change: Callback(path, event_type) where event_type is "created", "modified", "deleted"
            file_patterns: File extensions to watch (e.g., [".py", ".js"])
            debounce_seconds: Debounce interval
        """
        self.project_root = Path(project_root)
        self.on_change = on_change
        self.file_patterns = file_patterns or [".py", ".js", ".ts", ".jsx", ".tsx"]
        self.debounce_seconds = debounce_seconds
        
        self._observer = None
        self._pending_events: Dict[str, Tuple[str, float]] = {}
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._running = False
    
    def start(self):
        """Start watching for file changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.error("watchdog_not_installed")
            return
        
        class Handler(FileSystemEventHandler):
            def __init__(handler_self, watcher):
                handler_self.watcher = watcher
            
            def on_created(handler_self, event):
                if not event.is_directory:
                    handler_self.watcher._queue_event(event.src_path, "created")
            
            def on_modified(handler_self, event):
                if not event.is_directory:
                    handler_self.watcher._queue_event(event.src_path, "modified")
            
            def on_deleted(handler_self, event):
                if not event.is_directory:
                    handler_self.watcher._queue_event(event.src_path, "deleted")
        
        self._observer = Observer()
        self._observer.schedule(Handler(self), str(self.project_root), recursive=True)
        self._observer.start()
        self._running = True
        
        logger.info("file_watcher_started", path=str(self.project_root))
    
    def stop(self):
        """Stop watching for file changes."""
        self._running = False
        
        if self._debounce_timer:
            self._debounce_timer.cancel()
        
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        logger.info("file_watcher_stopped")
    
    def _queue_event(self, path: str, event_type: str):
        """Queue an event with debouncing."""
        # Check if file matches patterns
        if not any(path.endswith(ext) for ext in self.file_patterns):
            return
        
        # Skip excluded directories
        excluded = [".git", "__pycache__", "node_modules", ".venv", "venv"]
        if any(excl in path for excl in excluded):
            return
        
        with self._lock:
            self._pending_events[path] = (event_type, time.time())
            
            # Reset debounce timer
            if self._debounce_timer:
                self._debounce_timer.cancel()
            
            self._debounce_timer = threading.Timer(
                self.debounce_seconds,
                self._process_pending_events,
            )
            self._debounce_timer.start()
    
    def _process_pending_events(self):
        """Process pending events after debounce."""
        with self._lock:
            events = self._pending_events.copy()
            self._pending_events.clear()
        
        for path, (event_type, _) in events.items():
            try:
                self.on_change(path, event_type)
            except Exception as e:
                logger.error("file_change_callback_error", path=path, error=str(e))


class WatchedHybridSearch:
    """
    Hybrid search with automatic file watching.
    
    Convenience class that combines HybridSearchEngine with FileWatcher
    for automatic index updates.
    """
    
    def __init__(
        self,
        project_root: str,
        index_path: str,
        file_patterns: List[str] = None,
        auto_watch: bool = True,
    ):
        """
        Initialize watched hybrid search.
        
        Args:
            project_root: Project root directory
            index_path: Path to store index
            file_patterns: File patterns to index
            auto_watch: Whether to start watching immediately
        """
        from ai.hybrid_search import HybridSearchEngine
        
        self.project_root = project_root
        self.engine = HybridSearchEngine()
        self.indexer = IncrementalIndexer(self.engine, index_path)
        self.watcher: Optional[FileWatcher] = None
        
        # Initialize index
        self.indexer.initialize(project_root, file_patterns)
        
        # Start watching
        if auto_watch:
            self._start_watching(file_patterns)
    
    def _start_watching(self, file_patterns: List[str] = None):
        """Start file watcher."""
        extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go"]
        if file_patterns:
            extensions = [f".{p.lstrip('*.')}" for p in file_patterns]
        
        self.watcher = FileWatcher(
            self.project_root,
            self._on_file_change,
            file_patterns=extensions,
        )
        self.watcher.start()
    
    def _on_file_change(self, path: str, event_type: str):
        """Handle file change event."""
        if event_type in ("created", "modified"):
            self.indexer.update_file(path)
        elif event_type == "deleted":
            self.indexer.remove_file(path)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the codebase."""
        return self.engine.search(query, top_k=top_k)
    
    def stop(self):
        """Stop file watcher."""
        if self.watcher:
            self.watcher.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self.indexer.get_stats()
