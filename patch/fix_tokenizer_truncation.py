"""Removes the stale truncation.max_length=2048 config baked into this
checkpoint's tokenizer.json, which silently truncates any request whose
token count exceeds 2048 -- trivially hit by multi-image vision requests,
essentially never hit by single-image or text-only use. See README.md for
the full root-cause writeup.

Usage:
    python fix_tokenizer_truncation.py /path/to/checkpoint/tokenizer.json

Edits the file in place. Makes a .bak copy first.
"""

import json
import shutil
import sys


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    backup_path = path + ".bak"

    shutil.copy2(path, backup_path)
    print(f"Backed up original to {backup_path}")

    with open(path) as f:
        data = json.load(f)

    before = data.get("truncation")
    print(f"Current truncation config: {before}")

    if before is None:
        print("Already null -- nothing to do.")
        return

    data["truncation"] = None

    with open(path, "w") as f:
        json.dump(data, f)

    print(f"Wrote truncation: null to {path}")


if __name__ == "__main__":
    main()
