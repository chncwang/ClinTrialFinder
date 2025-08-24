import hashlib
import json
import time
import atexit
from collections import OrderedDict
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)


class PromptCache:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000, enable_validation: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.enable_validation = enable_validation
        self.cache_data: OrderedDict[str, Any] = OrderedDict()
        self._load_cache()
        logger.info(f"Initialized prompt cache with {len(self.cache_data)} entries")

        # Register cleanup function to run at program exit
        atexit.register(self._cleanup_on_exit)

    def set(self, original_key: str, result: Any) -> None:
        """Cache result for a given original key."""
        # Generate hash key internally
        cache_key = hashlib.sha256(original_key.encode("utf-8")).hexdigest()

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

        # Store both the original key and the result for debugging
        cache_entry = {
            "key": original_key,
            "value": result
        }

        # Save new entry
        self.cache_data[cache_key] = cache_entry
        logger.debug(f"After adding, cache_data[{cache_key}] = {cache_entry}")
        self._write_cache_to_disk()

    def get(self, original_key: str) -> str | None:
        """Get cached result for a given original key."""
        # Generate hash key internally
        cache_key = hashlib.sha256(original_key.encode("utf-8")).hexdigest()

        if cache_key in self.cache_data:
            logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
            try:
                # Get the value from the cache entry
                cache_entry = self.cache_data[cache_key]
                result = cache_entry["value"]

                # Ensure we're returning a string
                if isinstance(result, dict):
                    logger.warning(f"Converting dict to string for cache key: {cache_key[:8]}...")
                    result = json.dumps(result)
                elif not isinstance(result, str):
                    result = str(result)

                logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
                return result
            except Exception as e:
                logger.error(f"Error retrieving cache entry: {e}")
                return None
        logger.debug(f"Cache MISS for key: {cache_key[:8]}...")
        return None

    def _cleanup_on_exit(self):
        """Cleanup function registered with atexit to save cache on program exit."""
        try:
            logger.info("Saving cache on program exit...")
            self._write_cache_to_disk()
        except Exception as e:
            logger.error(f"Error saving cache on exit: {e}")

    def __del__(self):
        """Destructor to ensure cache is saved when object is destroyed."""
        try:
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
                self.cache_data: OrderedDict[str, Any] = OrderedDict()
        else:
            logger.info(f"No cache file found at {cache_path}")

    def _write_cache_to_disk(self):
        """Write the cache data to disk."""
        start_time = time.time()
        cache_path = self.cache_dir / "prompt_cache.json"
        try:
            with open(cache_path, "w") as f:
                to_dump_list: list[tuple[str, dict[str, Any]]] = list(self.cache_data.items())
                logger.debug(f"to_dump_list:")
                for item in to_dump_list:
                    logger.debug(f"  {item}")
                json.dump(to_dump_list, f, indent=2)

            # Validate that each first item in the tuple can be found in the text file (if enabled)
            if self.enable_validation:
                self._validate_cache_file(cache_path, to_dump_list)

            save_time = time.time() - start_time
            logger.info(f"Saved {len(self.cache_data)} cache entries in {save_time:.2f}s")
            return True
        except IOError as e:
            logger.error(f"Failed to save cache: {str(e)}")
            return False

    def _validate_cache_file(self, cache_path: Path, to_dump_list: list[tuple[str, dict[str, Any]]]) -> None:
        """Validate that each first item in the tuple can be found in the text file."""
        # Read the file as text to validate the dump
        with open(cache_path, "r") as f:
            file_content = f.read()

        # Check each first item (cache key) can be found in the text file
        missing_keys: list[str] = []
        for cache_key, _ in to_dump_list:
            if cache_key not in file_content:
                missing_keys.append(cache_key)

        if missing_keys:
            logger.error(f"Validation failed: {len(missing_keys)} cache keys not found in file: {missing_keys[:5]}...")
            logger.error(f"file_content: {file_content}")
            logger.error(f"missing_keys: {missing_keys}")
            raise ValueError(f"Validation failed: {len(missing_keys)} cache keys not found in file: {missing_keys[:5]}...")

    def get_original_key(self, cache_key: str) -> str | None:
        """Get the original key (before hashing) for a given cache key."""
        if cache_key in self.cache_data:
            cache_entry = self.cache_data[cache_key]
            return cache_entry["key"]
        return None

    def list_cache_entries(self) -> list[tuple[str, str, str]]:
        """List all cache entries with their original keys and values (first 100 chars)."""
        entries: list[tuple[str, str, str]] = []
        for cache_key, cache_entry in self.cache_data.items():
            original_key = cache_entry["key"]
            value = cache_entry["value"]

            # Truncate value for display
            value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            entries.append((cache_key, original_key, value_preview))

        return entries

