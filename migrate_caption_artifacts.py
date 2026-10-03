import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from artifact_io import atomic_write_json, temporary_path
from paths import (
    CAPTION_DIR,
    CAPTION_MAPPING_DIR,
    INDEX_DIR,
    canonical_video_artifact_path,
    video_group,
)

ARTIFACT_SPECS = (
    ("caption", CAPTION_DIR, ".json"),
    ("mapping", CAPTION_MAPPING_DIR, ".json"),
    ("index", INDEX_DIR, ".index"),
)


class MigrationTriplet(list):
    def __init__(self, entries, cleanup=()):
        super().__init__(entries)
        self.cleanup = list(cleanup)


def file_digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(root: Path, suffix: str):
    artifacts = {}
    pattern = f"L*_V*{suffix}"
    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue
        video_group(path.stem)
        artifacts.setdefault(path.stem, []).append(path)
    return artifacts


def plan_migration(specs=ARTIFACT_SPECS):
    by_kind = {
        kind: (root, suffix, collect_artifacts(root, suffix))
        for kind, root, suffix in specs
    }
    video_ids = sorted(
        set().union(*(set(data) for _, _, data in by_kind.values()))
    )
    plan = []

    for video_id in video_ids:
        triplet = []
        cleanup = []
        for kind, (root, suffix, artifacts) in by_kind.items():
            candidates = artifacts.get(video_id, [])
            destination = canonical_video_artifact_path(
                root,
                video_id,
                suffix,
            )
            if len(candidates) > 1:
                canonical_candidates = [
                    path
                    for path in candidates
                    if path.resolve() == destination.resolve()
                ]
                if not canonical_candidates or any(
                    file_digest(path) != file_digest(canonical_candidates[0])
                    for path in candidates
                ):
                    locations = ", ".join(str(path) for path in candidates)
                    raise RuntimeError(
                        f"Conflicting {kind} artifacts for {video_id}: {locations}"
                    )
                source = canonical_candidates[0]
                cleanup.extend(
                    path
                    for path in candidates
                    if path.resolve() != source.resolve()
                )
                candidates = [source]
            if candidates:
                source = candidates[0]
                triplet.append((kind, source, destination))

        if not triplet:
            continue
        if len(triplet) != len(specs):
            present = ", ".join(kind for kind, _, _ in triplet)
            raise RuntimeError(
                f"Incomplete artifact triplet for {video_id}: {present}"
            )
        if cleanup or any(
            source.resolve() != destination.resolve()
            for _, source, destination in triplet
        ):
            plan.append((video_id, MigrationTriplet(triplet, cleanup)))

    return plan


def migrate_triplet(video_id, triplet):
    staged = []
    published = []
    try:
        for kind, source, destination in triplet:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and source.resolve() != destination.resolve():
                raise FileExistsError(f"Destination already exists: {destination}")
            if source.resolve() == destination.resolve():
                continue

            temp_path = temporary_path(destination)
            shutil.copy2(source, temp_path)
            if file_digest(source) != file_digest(temp_path):
                raise RuntimeError(f"Copy verification failed for {source}")
            staged.append((kind, source, destination, temp_path))

        for _, _, destination, temp_path in staged:
            os.replace(temp_path, destination)
            published.append(destination)

        for _, source, destination, _ in staged:
            if file_digest(source) != file_digest(destination):
                raise RuntimeError(
                    f"Published artifact verification failed: {destination}"
                )

    except Exception:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for _, _, _, temp_path in staged:
            temp_path.unlink(missing_ok=True)

    # Cleanup is retry-safe and must not trigger publication rollback.
    for _, source, _, _ in staged:
        source.unlink()
    for redundant_path in getattr(triplet, "cleanup", ()):
        redundant_path.unlink(missing_ok=True)

    migrated_count = len(staged) + len(getattr(triplet, "cleanup", ()))
    print(f"[{video_id}] migrated {migrated_count} artifacts")


def import_aggregate_captions(
    aggregate_path=CAPTION_DIR / "captions.json",
    caption_root=CAPTION_DIR,
    apply=False,
):
    if not aggregate_path.is_file():
        return []

    records = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Aggregate captions must be an array: {aggregate_path}")

    grouped = {}
    for record in records:
        video_id = record.get("video_id")
        video_group(video_id)
        grouped.setdefault(video_id, []).append(record)

    pending = []
    for video_id, video_records in sorted(grouped.items()):
        destination = canonical_video_artifact_path(
            caption_root,
            video_id,
            ".json",
        )
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != video_records:
                raise FileExistsError(
                    f"Cannot import aggregate captions; conflicting destination "
                    f"exists: {destination}"
                )
            continue
        pending.append((destination, video_records))

    destinations = [destination for destination, _ in pending]
    if apply:
        written = []
        try:
            for destination, video_records in pending:
                atomic_write_json(destination, video_records)
                written.append(destination)
        except Exception:
            for destination in written:
                destination.unlink(missing_ok=True)
            raise

        aggregate_path.rename(aggregate_path.with_suffix(".json.migrated"))
    return destinations


def migrate_caption_artifacts(apply=False, specs=ARTIFACT_SPECS):
    plan = plan_migration(specs)
    for video_id, triplet in plan:
        sources = ", ".join(str(source) for _, source, _ in triplet)
        print(f"[{video_id}] {sources}")
        if apply:
            migrate_triplet(video_id, triplet)

    imported = import_aggregate_captions(apply=apply)
    print(
        f"Migration {'applied' if apply else 'dry run'}: "
        f"{len(plan)} triplets, {len(imported)} aggregate outputs"
    )
    return {"triplets": len(plan), "aggregate_outputs": len(imported)}


def main():
    parser = argparse.ArgumentParser(
        description="Migrate caption artifacts to the canonical per-video layout."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the migration; without this flag only print the plan",
    )
    args = parser.parse_args()
    migrate_caption_artifacts(apply=args.apply)


if __name__ == "__main__":
    main()
