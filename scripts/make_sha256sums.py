"""Regenerate SHA256SUMS.txt for the release tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_NAMES = {OUTPUT.name, ".DS_Store", "Thumbs.db"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        rows.append(f"{digest(path)}  {relative.as_posix()}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} entries to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
