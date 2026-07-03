"""Audit the public release directory for accidental sensitive artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = {
    "internal_project": re.compile(
        "|".join(["".join(["NOE", "NOP"]), "".join(["Noe", "Nop"]), "".join(["M", "DM"]), "".join(["VLA", "-Adapter-", "Mo", "e"])]),
        re.IGNORECASE,
    ),
    "private_path": re.compile(
        "|".join(["".join(["/", "root/"]), "".join(["C:", r"\\Users"]), "".join(["D:", r"\\VLAproject"]), "".join(["auto", "dl"])]),
        re.IGNORECASE,
    ),
    "credential": re.compile(
        "|".join([r"api[_-]?key", "".join(["sec", "ret"]), "".join(["pass", "word"]), r"access[_-]?" + "".join(["to", "ken"])]),
        re.IGNORECASE,
    ),
    "large_artifact": re.compile(r"\.(pt|pth|ckpt|bin|safetensors|tfrecord|mp4|mov|avi|mkv)$", re.IGNORECASE),
}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.name in {"audit_release.py", ".gitignore"}:
            continue
        if path.is_file() and ".git" not in path.parts:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Release directory to audit.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    issues = []

    for path in iter_files(root):
        rel = path.relative_to(root)
        for name, pattern in PATTERNS.items():
            if name == "large_artifact" and pattern.search(path.name):
                issues.append((name, rel, "binary artifact extension"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if name == "large_artifact":
                continue
            for match in pattern.finditer(text):
                issues.append((name, rel, match.group(0)))

    if issues:
        print("Release audit found issues:")
        for category, rel, value in issues:
            print(f"- {category}: {rel}: {value}")
        raise SystemExit(1)

    print(f"Release audit passed: {root}")


if __name__ == "__main__":
    main()
