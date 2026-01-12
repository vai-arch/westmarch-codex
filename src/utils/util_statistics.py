import logging
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from tabulate import tabulate
from tqdm import tqdm

from src.config import configuration_to_string, get_config
from src.paths import get_paths

config = get_config()
paths = get_paths()

# Load .env file
load_dotenv()  # looks for .env in the current working directory by default

# Get the path from env


def format_metric(key, value):
    """
    Smart formatting based on metric name.
    Extend or override easily.
    """

    if value is None:
        return "—"

    # Convert key to string for safe string operations
    key_str = str(key)

    # --- SPECIAL CASE: (value, percentage) ---
    if isinstance(value, (tuple, list, set)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        val, pct = value
        return f"{int(val)} ({pct:.1f}%)"

    # Convert timedelta automatically
    if isinstance(value, timedelta):
        value = value.total_seconds()

    # --- TIME METRICS ---
    if key_str.endswith("_time"):
        seconds = float(value)

        # Break apart
        days, seconds = divmod(seconds, 86400)  # 24*60*60
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        seconds_str = f"{seconds:.1f}".rstrip("0").rstrip(".")  # pretty seconds

        parts = []
        if days >= 1:
            parts.append(f"{int(days)}d")
        if hours >= 1 or days > 0:
            parts.append(f"{int(hours)}h")
        if minutes >= 1 or hours > 0 or days > 0:
            parts.append(f"{int(minutes)}m")

        # Always show seconds if everything else is zero
        if seconds > 0 or not parts:
            parts.append(f"{seconds_str}s")

        return " ".join(parts)

    # --- AVERAGE METRICS ---
    if key_str.startswith("avg_"):
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)  # fallback safely

    # --- TOKEN METRICS ---
    if key_str.endswith("_tokens") or key_str.startswith("max_"):
        return f"{int(value)}"

    # --- AUTO HANDLE NUMBERS ---
    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def tabulate_results(results):
    if isinstance(results, dict):
        results_array = []
        results_array.append(results)
        results = results_array

    # Collect all possible metric keys (columns)
    all_keys = list(results[0]["metrics"].keys())

    table = []
    for r in results:
        row = [r["name"]]
        metrics = r["metrics"]

        for key in all_keys:
            value = metrics.get(key)
            row.append(format_metric(key, value))

        table.append(row)

    return all_keys, table


def print_processed_time(total_time):
    formatted_time = format_metric("total_time", total_time)
    print(f"Total Duration: {formatted_time}")


def print_results_table(results, main_message=""):
    """
    Results = list of:
    return {
        "name": "Batch-Procesing-Ollama",
        "metrics":{
            "total_time": duration,
            "avg_time": duration.total_seconds() / BATCH_SIZE,
            "avg_tokens": avg_tokens,
            "max_tokens": -1
        }
    }
    """

    headers, rows = tabulate_results(results)

    if main_message != "":
        print(f"\n=== {main_message} ===\n")
    print(tabulate(rows, headers, tablefmt="grid"))


def print_results(results, main_message=""):
    """
    Print results line by line.

    Supports:
    - scalars
    - dicts (recursive, indented)
    - list of dicts
    """

    def _print_value(key, value, indent=2):
        pad = " " * indent

        # List of dicts
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            print(f"{pad}{key}:")
            for item in value:
                for sub_key, sub_value in item.items():
                    print(f"{pad}  - {sub_key}: {format_metric(sub_key, sub_value)}")

        # Dict (recursive)
        elif isinstance(value, dict):
            print(f"{pad}{key}:")
            for sub_key, sub_value in value.items():
                _print_value(sub_key, sub_value, indent + 2)

        # Scalar
        else:
            print(f"{pad}{key}: {format_metric(key, value)}")

    # Normalize input
    if isinstance(results, dict):
        results = [results]

    if main_message:
        print(f"\n=== {main_message} ===\n")

    for r in results:
        name = r.get("name", "Unknown")
        print(f"[{name}]")

        metrics = r.get("metrics", {})

        for key, value in metrics.items():
            _print_value(key, value, indent=2)

        print("-" * 40)


def reset_log(log_file):
    """
    Removes the current log file and resets the logger so a fresh
    file is created the next time get_stats_logger() is called.
    """
    log_path = paths.LOG_STATISTICS_PATH / f"{log_file}.log"

    # Remove the file if it exists
    if log_path.exists():
        log_path.unlink()

    # Reset the logger's handlers so Python does not keep old file handles open
    logger = logging.getLogger("stats_logger")
    logger.handlers.clear()


def get_stats_logger(logfile="stats.log"):
    logger = logging.getLogger(Path(logfile).stem)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # avoid duplicate handlers
        handler = RotatingFileHandler(
            paths.LOG_STATISTICS_PATH / logfile,
            maxBytes=20_000_000,  # 2 MB per file
            backupCount=5,
        )
        formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_results_table(results, log_file="stats", main_message="RESULTS"):
    logger = get_stats_logger(f"{log_file}.log")

    headers, rows = tabulate_results(results)

    table_str = "\n" + tabulate(rows, headers=headers) + "\n"
    logger.info(table_str)


def log_results(results, log_file="stats", main_message="RESULTS"):
    """
    Log results line by line.

    Supports:
    - scalars
    - dicts (recursive, indented)
    - list of dicts
    """

    def _log_value(key, value, indent=0):
        pad = " " * indent

        # List of dicts
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            logger.info(f"{pad}{key}:")
            for item in value:
                for sub_key, sub_value in item.items():
                    logger.info(f"{pad}  - {sub_key}: {format_metric(sub_key, sub_value)}")

        # Dict (recursive)
        elif isinstance(value, dict):
            logger.info(f"{pad}{key}:")
            for sub_key, sub_value in value.items():
                _log_value(sub_key, sub_value, indent + 2)

        # Scalar
        else:
            logger.info(f"{pad}{key}: {format_metric(key, value)}")

    # Normalize input
    if isinstance(results, dict):
        results = [results]

    logger = get_stats_logger(f"{log_file}.log")

    logger.info(f"=== {main_message} ===")

    for r in results:
        test_name = r.get("name", "unknown_test")
        logger.info(f"[{test_name}]")

        metrics = r.get("metrics", {})

        for key, value in metrics.items():
            _log_value(key, value, indent=2)

        logger.info("-" * 40)


# def log_results(results, log_file="stats", main_message="RESULTS"):
#     if isinstance(results, dict):
#         results = [results]

#     logger = get_stats_logger(f"{log_file}.log")

#     logger.info(f"=== {main_message} ===")

#     for r in results:
#         test_name = r.get("name", "unknown_test")
#         logger.info(f"[{test_name}]")

#         metrics = r.get("metrics", {})

#         for metric, value in metrics.items():
#             # 🔹 Case 1: list of dictionaries
#             if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
#                 logger.info(f"{metric}:")
#                 for item in value:
#                     for sub_key, sub_value in item.items():
#                         logger.info(f"  - {sub_key}: {format_metric(sub_key, sub_value)}")

#             # 🔹 Case 2: dictionary
#             elif isinstance(value, dict):
#                 logger.info(f"{metric}:")
#                 for sub_key, sub_value in value.items():
#                     logger.info(f"  {sub_key}: {format_metric(sub_key, sub_value)}")

#             # 🔹 Case 3: scalar
#             else:
#                 logger.info(f"{metric}: {format_metric(metric, value)}")

#         logger.info("-" * 40)


def log_processed_time(log_file, total_time):
    formatted_time = format_metric("total_time", total_time)
    logger = get_stats_logger(f"{log_file}.log")
    logger.info(f"Total Duration: {formatted_time}")


def log_configuration(log_file, config_section):
    logger = get_stats_logger(f"{log_file}.log")
    config.log_configuration(logger, config_section)


def total_statistics_logging(statistics, total_time, title, log_name, tables=True, configuration_section=None):
    if tables:
        print_results_table(statistics, title.upper())
    else:
        print_results(statistics, title)

    if total_time:
        print_processed_time(total_time)

    reset_log(log_name)

    log_results(statistics, log_name, title.upper())
    if tables:
        log_results_table(statistics, log_name, title.upper())

    if configuration_section is not None:
        logger = get_stats_logger(f"{log_name}.log")
        logger.info(configuration_to_string(configuration_section))
        print(configuration_to_string(configuration_section))

    log_processed_time(log_name, total_time)


def progress_bar(iterable, enable=True, **tqdm_kwargs):
    """
    Wrapper around tqdm that can be disabled with a flag.

    Args:
        iterable: any iterable
        enable (bool): whether to show the progress bar
        **tqdm_kwargs: forwarded to tqdm() when enabled

    Returns:
        iterable or tqdm-wrapped iterable
    """
    if enable:
        return tqdm(iterable, **tqdm_kwargs)
    else:
        # Return a plain iterator (no progress bar)
        return iterable
