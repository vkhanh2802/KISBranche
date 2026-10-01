import json
import os
from pathlib import Path
from uuid import uuid4


def temporary_path(destination: Path) -> Path:
    destination = Path(destination)
    return destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )


def atomic_write_json(destination: Path, data) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_path(destination)

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
