import hashlib
import json
from collections import OrderedDict
from pathlib import Path


class PromptCache:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_data = OrderedDict()
        self._load_cache()

    def _load_cache(self):
        """Load the cache from disk."""
        cache_path = self.cache_dir / "prompt_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    self.cache_data = OrderedDict(data)
            except (json.JSONDecodeError, IOError):
                self.cache_data = OrderedDict()

    def _save_cache(self):
        """Save the cache to disk."""
        with open(self.cache_dir / "prompt_cache.json", "w") as f:
            json.dump(list(self.cache_data.items()), f)

    def get(self, cache_key: str) -> dict | None:
        """Get cached result for a given cache key."""
        if cache_key in self.cache_data:
            return self.cache_data[cache_key]
        return None

    def set(self, cache_key: str, result: dict):
        """Cache result for a given cache key."""

        # Enforce cache size limit
        while len(self.cache_data) >= self.max_size:
            # Remove oldest entry
            self.cache_data.popitem(last=False)

        # Save new entry
        self.cache_data[cache_key] = result
        self._save_cache()
