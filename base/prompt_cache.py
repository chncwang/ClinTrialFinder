import hashlib
import json
import pickle
from collections import OrderedDict
from pathlib import Path


class PromptCache:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_index = OrderedDict()
        self._load_cache_index()

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Generate a unique cache key for a prompt and temperature."""
        cache_input = f"{prompt}_{temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _load_cache_index(self):
        """Load the cache index from disk."""
        index_path = self.cache_dir / "cache_index.pkl"
        if index_path.exists():
            with open(index_path, "rb") as f:
                self.cache_index = pickle.load(f)

    def _save_cache_index(self):
        """Save the cache index to disk."""
        with open(self.cache_dir / "cache_index.pkl", "wb") as f:
            pickle.dump(self.cache_index, f)

    def get(self, prompt: str, temperature: float) -> dict | None:
        """Get cached result for a prompt and temperature."""
        cache_key = self._get_cache_key(prompt, temperature)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return json.load(f)
        return None

    def set(self, prompt: str, temperature: float, result: dict):
        """Cache result for a prompt and temperature."""
        cache_key = self._get_cache_key(prompt, temperature)

        # Enforce cache size limit
        while len(self.cache_index) >= self.max_size:
            # Remove oldest entry
            oldest_key, _ = self.cache_index.popitem(last=False)
            (self.cache_dir / f"{oldest_key}.json").unlink(missing_ok=True)

        # Save new entry
        with open(self.cache_dir / f"{cache_key}.json", "w") as f:
            json.dump(result, f)

        self.cache_index[cache_key] = True
        self._save_cache_index()
