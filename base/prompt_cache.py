import json
import time
import atexit
from collections import OrderedDict
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)


class PromptCache:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_data: OrderedDict[str, str] = OrderedDict()
        self.last_save_time = 0
        self.modified_since_save = False
        self._load_cache()
        logger.info(f"Initialized prompt cache with {len(self.cache_data)} entries")
        
        # Register cleanup function to run at program exit
        atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        """Cleanup function registered with atexit to save cache on program exit."""
        try:
            if hasattr(self, 'modified_since_save') and self.modified_since_save:
                logger.info("Saving cache on program exit...")
                self._write_cache_to_disk()
        except Exception as e:
            logger.error(f"Error saving cache on exit: {e}")

    def __del__(self):
        """Destructor to ensure cache is saved when object is destroyed."""
        try:
            if hasattr(self, 'modified_since_save') and self.modified_since_save:
                logger.info("Saving cache before destruction...")
                self._write_cache_to_disk()
        except Exception as e:
            # Don't raise exceptions in destructor
            logger.error(f"Error saving cache during destruction: {e}")

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
                self.cache_data: OrderedDict[str, str] = OrderedDict()
        else:
            logger.info(f"No cache file found at {cache_path}")

    def _write_cache_to_disk(self):
        """Write the cache data to disk."""
        start_time = time.time()
        cache_path = self.cache_dir / "prompt_cache.json"
        try:
            with open(cache_path, "w") as f:
                json.dump(list(self.cache_data.items()), f)
            save_time = time.time() - start_time
            logger.info(f"Saved {len(self.cache_data)} cache entries in {save_time:.2f}s")
            return True
        except IOError as e:
            logger.error(f"Failed to save cache: {str(e)}")
            return False

    def _save_cache(self, force: bool = False):
        """Save the cache to disk if modified or forced."""
        current_time = time.time()
        # Only save if it's been modified and either forced or it's been > 5 minutes
        if (self.modified_since_save or force) and (force or current_time - self.last_save_time > 300):
            if self._write_cache_to_disk():
                self.last_save_time = current_time
                self.modified_since_save = False
        elif self.modified_since_save:
            logger.debug(f"Skipping cache save (last save was {current_time - self.last_save_time:.1f}s ago, will save after 300s)")

    def get(self, cache_key: str) -> str | None:
        """Get cached result for a given cache key."""
        if cache_key in self.cache_data:
            logger.debug(f"PromptCache.get: Cache HIT for key: {cache_key[:8]}...")
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
        logger.debug(f"PromptCache.get: Cache MISS for key: {cache_key[:8]}...")
        return None

    def set(self, cache_key: str, result: Any):
        """Cache result for a given cache key."""
        logger.debug(f"PromptCache.set: Caching result for key: {cache_key[:8]}...")

        # Enforce cache size limit
        while len(self.cache_data) >= self.max_size:
            # Remove oldest entry
            oldest_key, _ = next(iter(self.cache_data.items()))
            self.cache_data.popitem(last=False)
            logger.debug(f"PromptCache.set: Removed oldest cache entry: {oldest_key[:8]}...")

        # Ensure we're storing a string
        if not isinstance(result, str):
            try:
                if isinstance(result, dict):
                    logger.warning(f"PromptCache.set: Converting dict to string for cache key: {cache_key[:8]}...")
                    result = json.dumps(result)
                else:
                    logger.warning(f"PromptCache.set: Converting {type(result)} to string for cache key: {cache_key[:8]}...")
                    result = str(result)
            except Exception as e:
                logger.error(f"PromptCache.set: Error converting result to string: {e}")
                return

        # Save new entry
        self.cache_data[cache_key] = result
        self.modified_since_save = True
        self._save_cache()

    def save(self, force: bool = True):
        """Manually save the cache to disk."""
        logger.info("Manual cache save requested")
        if self._write_cache_to_disk():
            self.last_save_time = time.time()
            self.modified_since_save = False
            logger.info("Manual cache save completed successfully")
        else:
            logger.error("Manual cache save failed")
        
