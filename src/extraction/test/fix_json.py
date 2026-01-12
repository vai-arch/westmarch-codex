import json
import re


def fix_json_almost(json_text: str) -> list:
    """
    Fixes JSON that:
      - Has single-line comments // ...
      - Is missing commas between key-value pairs
      - Can be a list of objects
    Returns a Python list of dicts.
    """

    # 1️⃣ Remove single-line comments
    no_comments = re.sub(r"//.*", "", json_text)

    # 2️⃣ Add missing commas between quoted key-value pairs
    # Pattern: "value""key":  ->  "value", "key":
    fixed_commas = re.sub(r'(".*?")\s*(".*?")\s*:', r"\1, \2:", no_comments)

    # 3️⃣ Wrap in [] if it’s a list but missing surrounding brackets
    trimmed = fixed_commas.strip()
    if not trimmed.startswith("["):
        trimmed = f"[{trimmed}]"

    # 4️⃣ Parse safely
    return json.loads(trimmed)


# Load broken JSON
with open("C:\\Users\\Usuario\\Documents\\_AI\\westmarch-codex\\data\\temp\\entities_raw.stage2_20260112_105605_223878.txt", "r", encoding="utf-8") as f:
    broken_text = f.read()

fixed_text = fix_json_almost(broken_text)

print(fixed_text)
