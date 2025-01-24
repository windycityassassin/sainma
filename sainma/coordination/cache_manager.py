"""Cache management module."""

from typing import Dict, Any, Optional, List
import json
import hashlib
import time
from pathlib import Path
import threading
from dataclasses import dataclass
import diskcache
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CacheConfig:
    cache_dir: str = ".cache/sainma"
    max_size_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB
    ttl: int = 24 * 60 * 60  # 24 hours
    query_similarity_threshold: float = 0.85  # Threshold for semantic similarity

class CacheManager:
    """Manages caching of search results and processed clips."""
    
    def __init__(self, cache_dir: Optional[str] = None, config: Optional[CacheConfig] = None):
        """Initialize the cache manager."""
        if cache_dir:
            self.config = CacheConfig(cache_dir=cache_dir)
        else:
            self.config = config or CacheConfig()
            
        self._lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize disk cache
        self.cache = diskcache.Cache(
            directory=str(cache_path),
            size_limit=self.config.max_size_bytes
        )
        
        # Create namespaces for different types of data
        self._create_namespaces()
    
    def _create_namespaces(self):
        """Create cache namespaces."""
        # Create a dictionary to store namespaces
        self._namespaces = {}
        
        # Create separate caches for different types of data
        for ns in ['queries', 'clips', 'metadata']:
            self._namespaces[ns] = diskcache.Cache(
                directory=str(Path(self.config.cache_dir) / ns),
                size_limit=self.config.max_size_bytes // 3  # Split size among namespaces
            )
        
        # Assign namespaces to instance variables for backward compatibility
        self.query_cache = self._namespaces['queries']
        self.clip_cache = self._namespaces['clips']
        self.metadata_cache = self._namespaces['metadata']
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate a unique cache key for the data."""
        if isinstance(data, str):
            # For simple string queries
            return hashlib.sha256(data.encode()).hexdigest()
        else:
            # For complex data structures
            return hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
    
    def _is_semantically_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are semantically similar."""
        # TODO: Implement semantic similarity using sentence embeddings
        # For now, use simple string similarity
        from difflib import SequenceMatcher
        return SequenceMatcher(None, query1.lower(), query2.lower()).ratio() > \
            self.config.query_similarity_threshold
    
    def get_query_results(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached results for a query."""
        with self._lock:
            # Check exact match first
            key = self._generate_cache_key(query)
            if key in self.query_cache:
                result = self.query_cache[key]
                if self._is_cache_valid(result):
                    logger.info(f"Cache hit for query: {query}")
                    return result['data']
            
            # Check for semantically similar queries
            for cached_key in self.query_cache.iterkeys():
                cached_data = self.query_cache[cached_key]
                if self._is_semantically_similar(query, cached_data['original_query']):
                    if self._is_cache_valid(cached_data):
                        logger.info(f"Semantic cache hit for query: {query}")
                        return cached_data['data']
            
            return None
    
    def cache_query_results(self, query: str, results: Dict[str, Any]):
        """Cache the results of a query."""
        with self._lock:
            key = self._generate_cache_key(query)
            cache_data = {
                'data': results,
                'timestamp': time.time(),
                'original_query': query
            }
            self.query_cache[key] = cache_data
            logger.info(f"Cached results for query: {query}")
    
    def get_clip_data(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get cached clip data."""
        with self._lock:
            if clip_id in self.clip_cache:
                clip_data = self.clip_cache[clip_id]
                if self._is_cache_valid(clip_data):
                    return clip_data['data']
            return None
    
    def cache_clip_data(self, clip_id: str, clip_data: Dict[str, Any]):
        """Cache processed clip data."""
        with self._lock:
            cache_data = {
                'data': clip_data,
                'timestamp': time.time()
            }
            self.clip_cache[clip_id] = cache_data
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        age = time.time() - cache_data['timestamp']
        return age < self.config.ttl
    
    def invalidate_query(self, query: str):
        """Invalidate cached results for a query."""
        with self._lock:
            key = self._generate_cache_key(query)
            if key in self.query_cache:
                del self.query_cache[key]
    
    def invalidate_clip(self, clip_id: str):
        """Invalidate cached clip data."""
        with self._lock:
            if clip_id in self.clip_cache:
                del self.clip_cache[clip_id]
    
    def clear_expired(self):
        """Clear all expired cache entries."""
        with self._lock:
            current_time = time.time()
            
            # Clear expired queries
            for key in list(self.query_cache.iterkeys()):
                data = self.query_cache[key]
                if not self._is_cache_valid(data):
                    del self.query_cache[key]
            
            # Clear expired clips
            for key in list(self.clip_cache.iterkeys()):
                data = self.clip_cache[key]
                if not self._is_cache_valid(data):
                    del self.clip_cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'query_cache_size': len(self.query_cache),
            'clip_cache_size': len(self.clip_cache),
            'total_size_bytes': self.cache.size,
            'max_size_bytes': self.config.max_size_bytes,
            'ttl_seconds': self.config.ttl
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cache.close()
