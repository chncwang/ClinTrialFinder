import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path


logger = logging.getLogger(__name__)


class PromptCache:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_data = OrderedDict()
        self.last_save_time = 0
        self.modified_since_save = False
        self._load_cache()
        logger.info(f"Initialized prompt cache with {len(self.cache_data)} entries")

    def _load_cache(self):
        """Load the cache from disk."""
        cache_path = self.cache_dir / "prompt_cache.json"
        if cache_path.exists():
            try:
                start_time = time.time()
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    self.cache_data = OrderedDict(data)
                load_time = time.time() - start_time
                logger.info(f"Loaded {len(self.cache_data)} cache entries in {load_time:.2f}s")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache: {str(e)}")
                self.cache_data = OrderedDict()
        else:
            logger.info(f"No cache file found at {cache_path}")

    def _save_cache(self, force=False):
        """Save the cache to disk if modified or forced."""
        current_time = time.time()
        # Only save if it's been modified and either forced or it's been > 5 minutes
        if (self.modified_since_save or force) and (force or current_time - self.last_save_time > 300):
            start_time = time.time()
            cache_path = self.cache_dir / "prompt_cache.json"
            try:
                with open(cache_path, "w") as f:
                    json.dump(list(self.cache_data.items()), f)
                self.last_save_time = current_time
                self.modified_since_save = False
                save_time = time.time() - start_time
                logger.info(f"Saved {len(self.cache_data)} cache entries in {save_time:.2f}s")
            except IOError as e:
                logger.error(f"Failed to save cache: {str(e)}")
        elif self.modified_since_save:
            logger.debug(f"Skipping cache save (last save was {current_time - self.last_save_time:.1f}s ago, will save after 300s)")

    def get(self, cache_key: str) -> str | None:
        """Get cached result for a given cache key."""
        if cache_key in self.cache_data:
            try:
                # Ensure we're returning a string, not a dictionary
                result = self.cache_data[cache_key]
                if isinstance(result, dict):
                    logger.warning(f"Converting dict to string for cache key: {cache_key[:8]}...")
                    result = json.dumps(result)
                logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
                return result
            except Exception as e:
                logger.error(f"Error retrieving cache entry: {e}")
                return None
        logger.debug(f"Cache MISS for key: {cache_key[:8]}...")
        return None

    def set(self, cache_key: str, result: str):
        """Cache result for a given cache key."""
        logger.debug(f"Caching result for key: {cache_key[:8]}...")

        # Enforce cache size limit
        while len(self.cache_data) >= self.max_size:
            # Remove oldest entry
            oldest_key, _ = next(iter(self.cache_data.items()))
            self.cache_data.popitem(last=False)
            logger.debug(f"Removed oldest cache entry: {oldest_key[:8]}...")

        # Ensure we're storing a string
        if not isinstance(result, str):
            try:
                if isinstance(result, dict):
                    logger.warning(f"Converting dict to string for cache key: {cache_key[:8]}...")
                    result = json.dumps(result)
                else:
                    logger.warning(f"Converting {type(result)} to string for cache key: {cache_key[:8]}...")
                    result = str(result)
            except Exception as e:
                logger.error(f"Error converting result to string: {e}")
                return

        # Save new entry
        self.cache_data[cache_key] = result
        self.modified_since_save = True
        self._save_cache()
        
    def __del__(self):
        """Ensure cache is saved when object is destroyed."""
        self._save_cache(force=True)
