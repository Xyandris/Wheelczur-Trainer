"""Locate completed gait-model versions without importing ML dependencies."""

from pathlib import Path
import re


MODEL_FILES = ("front_model.joblib", "hind_model.joblib", "metadata.json")


def available_versions(root):
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    versions = []
    for path in root.iterdir():
        match = re.fullmatch(r"model_V(\d+)", path.name)
        if match and path.is_dir() and all((path / name).is_file() for name in MODEL_FILES):
            versions.append((int(match[1]), path))
    return [path for _, path in sorted(versions, key=lambda item: item[0], reverse=True)]


def resolve_version(value, root):
    root = Path(root).expanduser()
    if value is None:
        versions = available_versions(root)
        if not versions:
            raise ValueError(f"Brak wytrenowanej wersji w {root}. Skopiuj folder model/ albo uruchom trening.")
        path = versions[0]
    else:
        path = Path(value).expanduser()
        if not path.is_absolute() and len(path.parts) == 1 and re.fullmatch(r"model_V\d+", path.name):
            path = root / path
    missing = [name for name in MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Niepełna wersja modelu: {path}. Brak: {', '.join(missing)}")
    return path.resolve()
