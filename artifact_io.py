import json
import os
from pathlib import Path
import shutil
from uuid import uuid4


def temporary_path(destination: Path) -> Path:
    destination = Path(destination)
    return destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())


def atomic_write_json(destination: Path, data) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_path(destination)

    try:
        write_json(temp_path, data)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def publish_staged_files(staged_files) -> None:
    staged_files = [
        (Path(destination), Path(staged_path))
        for destination, staged_path in staged_files
    ]
    backups = []
    published = []

    try:
        for destination, staged_path in staged_files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup_path = None
            if destination.exists():
                backup_path = destination.with_name(
                    f".{destination.name}.{uuid4().hex}.bak"
                )
                shutil.copy2(destination, backup_path)
            backups.append((destination, backup_path))
            os.replace(staged_path, destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup_path in reversed(backups):
            if backup_path is not None and backup_path.exists():
                os.replace(backup_path, destination)
        raise
    else:
        for _, backup_path in backups:
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)
    finally:
        for _, staged_path in staged_files:
            staged_path.unlink(missing_ok=True)
