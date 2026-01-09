"""
Generic Fine-Tuning Framework - Paths Configuration
Manages all file paths for fine-tuning embedding models on any corpus.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


class Paths:
    """
    Path configuration for fine-tuning project.
    Works with any domain corpus.
    """

    def __init__(self, env_file=".env"):
        """Initialize paths by loading .env file"""
        load_dotenv(override=True)

        # Project root
        self.PROJECT_ROOT_PATH = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
        self.RAW_PATH = self.PROJECT_ROOT_PATH / "raw"
        self.DATA_PATH = self.PROJECT_ROOT_PATH / "data"
        self.DATA_PROCESSED_PATH = self.DATA_PATH / "processed"

        self.FILE_BOOK_00_RAW = self.RAW_PATH / "The Hobbit.epub"
        self.FILE_BOOK_00_PROCESSED = self.DATA_PROCESSED_PATH / "book_00_processed.json"
        self.FILE_ENTITIES_RAW = self.DATA_PROCESSED_PATH / "entities_raw.json"
        self.FILE_ENTITIES_DEDUPLICATED = self.DATA_PROCESSED_PATH / "entities_deduplicated.json"
        self.FILE_BOOKS_CHUNKED = self.DATA_PROCESSED_PATH / "books_chunked.json"

        self.LOG_PATH = self.PROJECT_ROOT_PATH / "logs"
        self.LOG_STATISTICS_PATH = self.LOG_PATH / "statistics"
        self.FILE_MAIN_LOG = self.LOG_PATH / "westmarch_codex.log"

        # Create all directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for name, value in self.__dict__.items():
            if name.endswith("_PATH") and isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        """String representation"""
        return f"Paths(PROJECT_ROOT={self.PROJECT_ROOT_PATH})"


# Global paths instance
_paths = None


def get_paths():
    """Get the global paths instance"""
    global _paths
    if _paths is None:
        _paths = Paths()
    return _paths


# Convenience function for testing
def print_paths():
    """Print current paths (useful for debugging)"""
    paths = get_paths()

    print("=" * 70)
    print("Westmarch Codex - Paths & Files")
    print("=" * 70)

    print("\n📁 DIRECTORY PATHS:")
    for name, value in paths.__dict__.items():
        if name.endswith("_PATH") and isinstance(value, Path):
            exists = "✅" if value.exists() else "❌"
            print(f"  {exists} {name:30s}: {value}")

    print("\n📄 FILE PATHS:")
    for name, value in paths.__dict__.items():
        if name.startswith("FILE_") and isinstance(value, Path):
            exists = "✅" if value.exists() else "❌"
            print(f"  {exists} {name:30s}: {value}")


if __name__ == "__main__":
    print_paths()
