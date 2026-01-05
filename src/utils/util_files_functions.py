import fnmatch
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

from src.utils.util_logging import get_logger

logger = get_logger(__name__)


def get_object_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)


def deserialize_object(input_file, log=False):
    if log:
        logger.debug(f"\n📂 Loading object from: {input_file}")

    with open(input_file, "rb") as f:
        return pickle.load(f)


def serialize_object(data, output_file, log=False):
    if log:
        logger.debug(f"\n💾 Saving object to: {output_file}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def remove_file(filepath: str):
    """Remove a file safely with proper checks."""
    file_path = Path(filepath)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    if not file_path.is_file():
        print(f"❌ Path exists but is not a file: {file_path}")
        return False

    try:
        file_path.unlink()
        print(f"✅ Removed file: {file_path}")
        return True
    except PermissionError:
        print(f"❌ Permission denied: unable to delete {file_path}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error removing file: {e}")
        return False


def find_files_in_folder(path_folder, extension=".txt", exception=True, recursive=False, sort=True):
    """
    Find files in a folder with a given extension.

    Args:
        path_folder (str | Path): Folder to search
        extension (str): Extension filter, e.g. ".txt" or "txt"
        recursive (bool): If True, use rglob() to include subfolders

    Returns:
        List[Path]: List of file paths found
    """
    folder = Path(path_folder)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    # Normalize extension
    ext = extension.lstrip(".")
    pattern = f"*.{ext}"

    # Choose search strategy
    files = folder.rglob(pattern) if recursive else folder.glob(pattern)

    if sort:
        files = sorted(files)

    found_files = list(files)

    if not found_files:
        logger.warning(f"No files found in {folder} with extension: {extension}")
        if exception:
            raise FileNotFoundError(f"No files found in {folder} with extension: {extension}")
    else:
        logger.info(f"Found {len(found_files)} file(s) in {folder} with extension: {extension}")

    return found_files


def copy_files(
    src_folder,
    dst_folder,
    extension=".txt",
    recursive=False,
    overwrite=True,
    log=True,
    exclude_pattern=None,  # 👈 NEW
):
    """
    Copy files from src_folder to dst_folder.

    Args:
        src_folder (str | Path): Source folder
        dst_folder (str | Path): Destination folder
        extension (str): Extension filter, e.g. ".txt" or "txt"
        recursive (bool): If True, use rglob() to include subfolders
        overwrite (bool): If False, skip files that already exist
        exclude_pattern (str | None): Glob pattern to exclude
            (e.g. "*_SKIPPED.txt"). If None, nothing is excluded.
    """

    src = Path(src_folder)
    dst = Path(dst_folder)

    # --- Validate source folder ---
    if not src.exists() or not src.is_dir():
        logger.error(f"Source folder does not exist or is not a directory: {src}")
        return False

    # --- Ensure destination folder exists ---
    dst.mkdir(parents=True, exist_ok=True)

    # Normalize extension
    ext = extension.lstrip(".")
    pattern = f"*.{ext}"

    # Choose search strategy
    files = src.rglob(pattern) if recursive else src.glob(pattern)

    copied_count = 0

    for file_path in files:
        # 🔹 Exclude pattern (only if provided)
        if exclude_pattern and fnmatch.fnmatch(file_path.name, exclude_pattern):
            if log:
                logger.info(f"Excluded by pattern: {file_path.name}")
            continue

        dest_file = dst / file_path.name

        if dest_file.exists() and not overwrite:
            if log:
                logger.info(f"Skipping existing file: {dest_file}")
            continue

        try:
            shutil.copy2(file_path, dest_file)
            if log:
                logger.info(f"Copied: {file_path} → {dest_file}")
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {file_path}: {e}")

    if copied_count == 0:
        if log:
            logger.warning(f"No files copied from {src} (pattern: {pattern})")
    else:
        if log:
            logger.info(f"Copied {copied_count} file(s) from {src} → {dst}")

    return True


def load_json_line_by_line(file, log=True):
    """
    Load a file line by line as JSON objects.

    Args:
        file (str or Path): Path to the JSONL file.
    """
    lines = []

    input_file = Path(file)

    if not input_file.exists():
        raise FileNotFoundError(f"❌ Error: Text file not found: {file}")

    if log:
        logger.debug(f"📂 Loading file: {file}")

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))

    if log:
        logger.info(f"   Loaded: {len(lines):,} lines")

    return lines


def load_txt_line_by_line(file, log=True):
    """
    Load a file line by line as JSON objects.

    Args:
        file (str or Path): Path to the JSONL file.
    """
    lines = []

    input_file = Path(file)

    if not input_file.exists():
        raise FileNotFoundError(f"❌ Error: Text file not found: {file}")

    if log:
        logger.debug(f"📂 Loading file: {file}")

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if log:
        logger.info(f"   Loaded: {len(lines):,} lines")

    return lines


def load_text_from_file(file, log=True):
    """
    Load text data from a file. Raises FileNotFoundError if the file does not exist.

    Args:
        file (str or Path): Path to the text file.
    """

    input_file = Path(file)

    if not input_file.exists():
        raise FileNotFoundError(f"❌ Error: Text file not found: {file}")

    if log:
        logger.debug(f"📂 Loading file: {file}")

    with open(input_file, "r", encoding="utf-8") as f:
        text_data = f.read()

    if log:
        logger.info(f"📂 Loaded file: {file}")

    return text_data


def load_json_from_file(file, log=True):
    """
    Load JSON data from a file. Raises FileNotFoundError if the file does not exist.

    Args:
        file (str or Path): Path to the JSON file.

    Returns:
        dict or list: Loaded JSON data.
    """
    input_file = Path(file)

    if not input_file.exists():
        raise FileNotFoundError(f"❌ Error: file not found: {file}")

    if log:
        logger.debug(f"📂 Loading file: {file}")

    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if log:
        logger.info(f"📂 Loaded file: {file}")
    return json_data


def save_jsonl_to_file(data: List[Dict], output_file, indent: int = None):
    # Save to JSONL
    logger.debug(f"\n💾 Saving {len(data)} chunks to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for json_element in data:
            f.write(json.dumps(json_element, indent=indent, ensure_ascii=False) + "\n")

    logger.info(f"\n💾 Saved {len(data)} json elements to: {output_file}")


def save_json_to_file(data: List[Dict], output_file, indent: int = None, log=True):
    # Save to JSON
    if log:
        logger.debug(f"\n💾 Saving {len(data)} chunks to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    if log:
        logger.info(f"\n💾 Saved {len(data)} json elements to: {output_file}")
