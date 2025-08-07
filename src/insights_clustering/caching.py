"""
Generation 3 Multi-Layer Intelligent Caching System
Provides feature vector caching, cluster model caching, and result caching
with intelligent eviction, compression, and distributed cache support
"""

import asyncio
import hashlib
import json
import logging
import time
import zlib
import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading
import numpy as np
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecureSerializer:
    """Secure JSON-based serialization to replace pickle for cache data"""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Securely serialize object to JSON bytes"""
        try:
            if isinstance(obj, np.ndarray):
                # Convert numpy arrays to lists with metadata
                serializable = {
                    '_type': 'numpy.ndarray',
                    'data': obj.tolist(),
                    'shape': obj.shape,
                    'dtype': str(obj.dtype)
                }
            elif isinstance(obj, pd.DataFrame):
                # Convert dataframes to JSON-serializable format
                serializable = {
                    '_type': 'pandas.DataFrame',
                    'data': obj.to_dict('records'),
                    'columns': obj.columns.tolist(),
                    'index': obj.index.tolist()
                }
            elif isinstance(obj, dict):
                # Handle nested dictionaries recursively
                serializable = SecureSerializer._serialize_dict(obj)
            elif hasattr(obj, '__dict__'):
                # Handle objects with __dict__
                serializable = {
                    '_type': 'object',
                    'class': obj.__class__.__name__,
                    'data': SecureSerializer._serialize_dict(obj.__dict__)
                }
            else:
                # Basic types
                serializable = obj
            
            json_str = json.dumps(serializable, separators=(',', ':'))
            return json_str.encode('utf-8')
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise ValueError(f"Cannot serialize object: {e}")
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Securely deserialize object from JSON bytes"""
        try:
            json_str = data.decode('utf-8')
            obj = json.loads(json_str)
            
            return SecureSerializer._deserialize_object(obj)
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise ValueError(f"Cannot deserialize data: {e}")
    
    @staticmethod
    def _serialize_dict(d: dict) -> dict:
        """Recursively serialize dictionary"""
        result = {}
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                result[key] = {
                    '_type': 'numpy.ndarray',
                    'data': value.tolist(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, pd.DataFrame):
                result[key] = {
                    '_type': 'pandas.DataFrame',
                    'data': value.to_dict('records'),
                    'columns': value.columns.tolist(),
                    'index': value.index.tolist()
                }
            elif isinstance(value, dict):
                result[key] = SecureSerializer._serialize_dict(value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _deserialize_object(obj: Any) -> Any:
        """Recursively deserialize object"""
        if isinstance(obj, dict) and '_type' in obj:
            if obj['_type'] == 'numpy.ndarray':
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif obj['_type'] == 'pandas.DataFrame':
                df = pd.DataFrame(obj['data'], columns=obj['columns'])
                df.index = obj['index']
                return df
            elif obj['_type'] == 'object':
                # For security, we only deserialize known safe objects
                return obj['data']  # Return the data dict instead of reconstructing the object
        elif isinstance(obj, dict):
            return {key: SecureSerializer._deserialize_object(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [SecureSerializer._deserialize_object(item) for item in obj]
        else:
            return obj


@dataclass
class CacheStats:
    """Cache statistics for monitoring and optimization"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    compression_ratio: float = 1.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    compressed: bool = False
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1=low, 2=medium, 3=high
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def access_frequency(self) -> float:
        """Access frequency (accesses per second)"""
        age = max(self.age, 1.0)  # Avoid division by zero
        return self.access_count / age


class CompressionManager:
    """Handles data compression for cache optimization"""
    
    @staticmethod
    def compress(data: Any, algorithm: str = 'auto') -> Tuple[bytes, str, float]:
        """
        Compress data using specified algorithm
        Returns: (compressed_data, algorithm_used, compression_ratio)
        """
        serialized = SecureSerializer.serialize(data)
        original_size = len(serialized)
        
        if algorithm == 'auto':
            # Choose best compression based on data characteristics
            if original_size < 1024:  # < 1KB
                return serialized, 'none', 1.0
            elif LZ4_AVAILABLE and original_size > 100000:  # > 100KB
                algorithm = 'lz4'
            else:
                algorithm = 'zlib'
        
        if algorithm == 'lz4' and LZ4_AVAILABLE:
            compressed = lz4.frame.compress(serialized)
        elif algorithm == 'zlib':
            compressed = zlib.compress(serialized, level=6)
        else:
            compressed = serialized
            algorithm = 'none'
        
        compression_ratio = len(compressed) / original_size if original_size > 0 else 1.0
        return compressed, algorithm, compression_ratio
    
    @staticmethod
    def decompress(compressed_data: bytes, algorithm: str) -> Any:
        """Decompress data using specified algorithm"""
        if algorithm == 'none':
            return SecureSerializer.deserialize(compressed_data)
        elif algorithm == 'lz4' and LZ4_AVAILABLE:
            decompressed = lz4.frame.decompress(compressed_data)
        elif algorithm == 'zlib':
            decompressed = zlib.decompress(compressed_data)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
        
        return SecureSerializer.deserialize(decompressed)


class IntelligentLRUCache:
    """Intelligent LRU cache with adaptive eviction and compression"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 500.0,
                 default_ttl: Optional[float] = 3600, enable_compression: bool = True):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.compression_manager = CompressionManager()
        
        # Adaptive parameters
        self.compression_threshold = 10240  # 10KB
        self.eviction_batch_size = max(10, max_size // 100)
    
    def _generate_key(self, key_data: Any) -> str:
        """Generate cache key from arbitrary data"""
        if isinstance(key_data, str):
            return key_data
        elif isinstance(key_data, dict):
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        else:
            key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(SecureSerializer.serialize(obj))
        except:
            return 1024  # Fallback estimate
    
    def _should_compress(self, size_bytes: int, priority: int) -> bool:
        """Determine if object should be compressed"""
        if not self.enable_compression:
            return False
        return size_bytes > self.compression_threshold or priority <= 1
    
    def _evict_entries(self):
        """Intelligent cache eviction based on multiple factors"""
        if len(self.cache) <= self.max_size and self._get_memory_usage() <= self.max_memory_bytes:
            return
        
        # Calculate eviction scores for each entry
        eviction_candidates = []
        current_time = time.time()
        
        for key, entry in self.cache.items():
            if entry.is_expired:
                eviction_candidates.append((key, float('inf')))  # Expired entries first
                continue
            
            # Eviction score based on multiple factors
            age_factor = (current_time - entry.last_accessed) / 3600  # Hours since access
            frequency_factor = 1.0 / (entry.access_frequency + 0.01)  # Lower frequency = higher score
            size_factor = entry.size_bytes / (1024 * 1024)  # Size in MB
            priority_factor = 1.0 / entry.priority  # Lower priority = higher score
            
            score = age_factor * frequency_factor * size_factor * priority_factor
            eviction_candidates.append((key, score))
        
        # Sort by eviction score (highest first)
        eviction_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evict entries in batches
        evicted_count = 0
        target_evictions = min(self.eviction_batch_size, len(eviction_candidates))
        
        for key, score in eviction_candidates[:target_evictions]:
            if key in self.cache:
                del self.cache[key]
                evicted_count += 1
                self.stats.evictions += 1
                
                # Stop if we've freed enough space
                if (len(self.cache) <= self.max_size * 0.8 and 
                    self._get_memory_usage() <= self.max_memory_bytes * 0.8):
                    break
        
        if evicted_count > 0:
            logger.debug(f"Evicted {evicted_count} cache entries")
    
    def _get_memory_usage(self) -> int:
        """Calculate current memory usage"""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def get(self, key: Union[str, Any], tags: Optional[List[str]] = None) -> Optional[Any]:
        """Get value from cache with intelligent access tracking"""
        cache_key = self._generate_key(key)
        start_time = time.time()
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                if entry.is_expired:
                    del self.cache[cache_key]
                    self.stats.misses += 1
                    return None
                
                # Update access statistics
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                
                # Decompress if needed
                try:
                    if entry.compressed:
                        value = self.compression_manager.decompress(
                            entry.value, getattr(entry, 'compression_algorithm', 'zlib')
                        )
                    else:
                        value = entry.value
                    
                    self.stats.hits += 1
                    access_time = (time.time() - start_time) * 1000
                    self.stats.avg_access_time_ms = (
                        (self.stats.avg_access_time_ms * (self.stats.hits - 1) + access_time) / self.stats.hits
                    )
                    
                    return value
                    
                except Exception as e:
                    logger.warning(f"Failed to decompress cache entry {cache_key}: {e}")
                    del self.cache[cache_key]
            
            self.stats.misses += 1
            return None
    
    def put(self, key: Union[str, Any], value: Any, ttl: Optional[float] = None,
            tags: Optional[List[str]] = None, priority: int = 2) -> bool:
        """Put value in cache with intelligent storage optimization"""
        cache_key = self._generate_key(key)
        
        with self.lock:
            # Evict entries if needed
            self._evict_entries()
            
            current_time = time.time()
            size_bytes = self._estimate_size(value)
            
            # Determine compression
            should_compress = self._should_compress(size_bytes, priority)
            
            if should_compress:
                try:
                    compressed_data, algorithm, compression_ratio = self.compression_manager.compress(value)
                    stored_value = compressed_data
                    compressed = True
                    actual_size = len(compressed_data)
                except Exception as e:
                    logger.warning(f"Compression failed for {cache_key}: {e}")
                    stored_value = value
                    compressed = False
                    algorithm = 'none'
                    compression_ratio = 1.0
                    actual_size = size_bytes
            else:
                stored_value = value
                compressed = False
                algorithm = 'none'
                compression_ratio = 1.0
                actual_size = size_bytes
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=stored_value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                size_bytes=actual_size,
                compressed=compressed,
                ttl=ttl or self.default_ttl,
                tags=tags or [],
                priority=priority
            )
            
            if compressed:
                entry.compression_algorithm = algorithm
            
            # Store in cache
            if cache_key in self.cache:
                self.cache.pop(cache_key)  # Remove old entry
            
            self.cache[cache_key] = entry
            
            # Update stats
            self.stats.size = len(self.cache)
            self.stats.memory_usage_mb = self._get_memory_usage() / (1024 * 1024)
            if compressed:
                self.stats.compression_ratio = (
                    (self.stats.compression_ratio + compression_ratio) / 2
                )
            
            return True
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags"""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logger.debug(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self.lock:
            self.stats.size = len(self.cache)
            self.stats.memory_usage_mb = self._get_memory_usage() / (1024 * 1024)
            return self.stats


class RedisDistributedCache:
    """Redis-based distributed cache for cluster coordination"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None,
                 key_prefix: str = 'neuromorphic_cache:'):
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install redis-py package.")
        
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=False, socket_keepalive=True,
            socket_keepalive_options={}, health_check_interval=30
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix"""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        redis_key = self._make_key(key)
        
        try:
            start_time = time.time()
            data = self.redis_client.get(redis_key)
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize
            value = SecureSerializer.deserialize(data)
            
            self.stats.hits += 1
            access_time = (time.time() - start_time) * 1000
            self.stats.avg_access_time_ms = (
                (self.stats.avg_access_time_ms * (self.stats.hits - 1) + access_time) / self.stats.hits
            )
            
            return value
            
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = 3600) -> bool:
        """Put value in Redis cache"""
        redis_key = self._make_key(key)
        
        try:
            # Serialize
            data = SecureSerializer.serialize(value)
            
            # Store with TTL
            result = self.redis_client.setex(redis_key, ttl or 3600, data)
            
            if result:
                self.stats.size += 1
            
            return result
            
        except Exception as e:
            logger.warning(f"Redis put failed for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        redis_key = self._make_key(key)
        
        try:
            result = self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {e}")
            return False
    
    def clear_by_pattern(self, pattern: str = "*"):
        """Clear keys matching pattern"""
        try:
            full_pattern = f"{self.key_prefix}{pattern}"
            keys = self.redis_client.keys(full_pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} Redis keys matching {pattern}")
        except Exception as e:
            logger.warning(f"Redis pattern clear failed: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get Redis cache statistics"""
        try:
            info = self.redis_client.info('memory')
            self.stats.memory_usage_mb = info.get('used_memory', 0) / (1024 * 1024)
        except:
            pass
        
        return self.stats


class MultiLayerCache:
    """Multi-layer intelligent cache system combining memory and distributed caching"""
    
    def __init__(self, 
                 l1_size: int = 500,  # Memory cache size
                 l2_size: int = 5000,  # Disk cache size (if enabled)
                 enable_redis: bool = False,
                 redis_config: Optional[Dict] = None,
                 enable_compression: bool = True,
                 cache_dir: Optional[Path] = None):
        
        # Layer 1: In-memory cache (fastest)
        self.l1_cache = IntelligentLRUCache(
            max_size=l1_size,
            max_memory_mb=200.0,
            enable_compression=enable_compression
        )
        
        # Layer 2: Disk cache (persistent)
        self.cache_dir = cache_dir or Path.home() / '.neuromorphic_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_disk_cache = True
        
        # Layer 3: Distributed cache (shared)
        self.enable_redis = enable_redis and REDIS_AVAILABLE
        if self.enable_redis:
            redis_config = redis_config or {}
            try:
                self.redis_cache = RedisDistributedCache(**redis_config)
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
                self.enable_redis = False
        
        self.stats = {
            'l1': CacheStats(),
            'l2': CacheStats(),
            'l3': CacheStats()
        }
    
    def _get_disk_cache_path(self, key: str) -> Path:
        """Get disk cache file path"""
        # Use first 2 chars for directory sharding
        shard = key[:2]
        shard_dir = self.cache_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{key}.cache"
    
    def _disk_get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        cache_path = self._get_disk_cache_path(key)
        
        try:
            if cache_path.exists():
                # Check if file is not too old (24 hours)
                if time.time() - cache_path.stat().st_mtime < 86400:
                    with open(cache_path, 'rb') as f:
                        data = f.read()
                    
                    # Try to decompress
                    try:
                        return SecureSerializer.deserialize(zlib.decompress(data))
                    except:
                        return SecureSerializer.deserialize(data)
                        
        except Exception as e:
            logger.debug(f"Disk cache read failed for {key}: {e}")
        
        return None
    
    def _disk_put(self, key: str, value: Any) -> bool:
        """Put value in disk cache"""
        cache_path = self._get_disk_cache_path(key)
        
        try:
            # Serialize and compress
            data = SecureSerializer.serialize(value)
            compressed_data = zlib.compress(data, level=6)
            
            with open(cache_path, 'wb') as f:
                f.write(compressed_data)
            
            return True
            
        except Exception as e:
            logger.debug(f"Disk cache write failed for {key}: {e}")
            return False
    
    def get(self, key: Union[str, Any], **kwargs) -> Optional[Any]:
        """Get value from multi-layer cache"""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        # Try L1 cache first (memory)
        value = self.l1_cache.get(cache_key, **kwargs)
        if value is not None:
            self.stats['l1'].hits += 1
            return value
        self.stats['l1'].misses += 1
        
        # Try L2 cache (disk)
        if self.enable_disk_cache:
            value = self._disk_get(cache_key)
            if value is not None:
                # Promote to L1
                self.l1_cache.put(cache_key, value, **kwargs)
                self.stats['l2'].hits += 1
                return value
            self.stats['l2'].misses += 1
        
        # Try L3 cache (Redis)
        if self.enable_redis:
            value = self.redis_cache.get(cache_key)
            if value is not None:
                # Promote to L1 and L2
                self.l1_cache.put(cache_key, value, **kwargs)
                if self.enable_disk_cache:
                    self._disk_put(cache_key, value)
                self.stats['l3'].hits += 1
                return value
            self.stats['l3'].misses += 1
        
        return None
    
    def put(self, key: Union[str, Any], value: Any, **kwargs) -> bool:
        """Put value in multi-layer cache"""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        success = True
        
        # Store in L1 (memory)
        success &= self.l1_cache.put(cache_key, value, **kwargs)
        
        # Store in L2 (disk)
        if self.enable_disk_cache:
            success &= self._disk_put(cache_key, value)
        
        # Store in L3 (Redis)
        if self.enable_redis:
            ttl = kwargs.get('ttl', 3600)
            success &= self.redis_cache.put(cache_key, value, ttl=ttl)
        
        return success
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags"""
        self.l1_cache.invalidate_by_tags(tags)
        
        if self.enable_redis:
            # Redis doesn't support tag-based invalidation directly
            # This would require additional metadata storage
            pass
    
    def clear(self, layer: Optional[str] = None):
        """Clear cache layers"""
        if layer is None or layer == 'l1':
            self.l1_cache.clear()
        
        if layer is None or layer == 'l2':
            # Clear disk cache
            if self.enable_disk_cache:
                for cache_file in self.cache_dir.glob('*/*.cache'):
                    try:
                        cache_file.unlink()
                    except:
                        pass
        
        if layer is None or layer == 'l3':
            if self.enable_redis:
                self.redis_cache.clear_by_pattern()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get comprehensive cache statistics"""
        stats = {
            'l1': self.l1_cache.get_stats(),
            'l2': self.stats['l2'],
            'l3': self.redis_cache.get_stats() if self.enable_redis else self.stats['l3']
        }
        
        # Calculate disk cache stats
        if self.enable_disk_cache:
            cache_files = list(self.cache_dir.glob('*/*.cache'))
            total_size = sum(f.stat().st_size for f in cache_files)
            stats['l2'].size = len(cache_files)
            stats['l2'].memory_usage_mb = total_size / (1024 * 1024)
        
        return stats
    
    def _generate_key(self, key_data: Any) -> str:
        """Generate cache key from arbitrary data"""
        if isinstance(key_data, dict):
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        else:
            key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()


# Global cache instance
neuromorphic_cache = MultiLayerCache()


def cached_neuromorphic_operation(cache_key_func: Optional[Callable] = None, 
                                ttl: int = 3600,
                                tags: Optional[List[str]] = None,
                                priority: int = 2):
    """Decorator for caching neuromorphic operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = neuromorphic_cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}, executing...")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache the result
            neuromorphic_cache.put(
                cache_key, result, 
                ttl=ttl, tags=tags, priority=priority
            )
            
            logger.debug(f"Cached result for {func.__name__} (execution time: {execution_time:.3f}s)")
            return result
        
        return wrapper
    return decorator