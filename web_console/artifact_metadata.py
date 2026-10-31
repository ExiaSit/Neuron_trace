"""Expose the actual path of already registered, job-scoped result artifacts."""
from pathlib import Path


def with_storage_paths(root, items):
    base = Path(root).resolve()
    result = []
    for item in items:
        try:
            relative = Path(item['path'])
            if relative.is_absolute():
                continue
            path = (base / relative).resolve()
            if path == base or base not in path.parents:
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        result.append(dict(item, storage_path=str(path)))
    return result
