"""Collection of advances file operations."""

import os
from collections.abc import Iterable
from pathlib import Path


def get_files(
    root: Path | str, extensions: Iterable[str], relative_to: Path | str, depth=2
) -> list[Path]:
    root = str(root)
    relative_to = str(relative_to)

    files = []

    if depth == 0:
        return files

    with os.scandir(root) as it:
        for file in it:
            if file.is_dir():
                files += get_files(file.path, extensions, relative_to, depth - 1)
            else:
                f = Path(file.path)
                extension = f.suffix.lower()
                if extension in extensions:
                    files.append(f.relative_to(relative_to))

    return files
