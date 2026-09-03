"""Recoverable quarantine and atomic publication of audited triplets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Sequence

from .contracts import ROLES, atomic_json, iter_jsonl, sha256_file


def quarantine_triplets(
    dataset_root: Path,
    candidate_ids: Path,
    quarantine_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    values = [
        line.strip()
        for line in candidate_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("candidate list is empty or duplicated")
    current = {
        role: {path.stem for path in (dataset_root / role).glob("*.png")}
        for role in ROLES
    }
    if any(current[role] != current["gt"] for role in ROLES):
        raise ValueError("dataset triplet membership differs")
    missing = set(values) - current["gt"]
    if missing:
        raise ValueError(f"candidate IDs missing from dataset: {sorted(missing)[:5]}")
    for role in ROLES:
        (quarantine_root / role).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    moves: list[tuple[Path, Path]] = []
    try:
        for sample_id in values:
            hashes: dict[str, str] = {}
            for role in ROLES:
                source = dataset_root / role / f"{sample_id}.png"
                destination = quarantine_root / role / source.name
                if destination.exists():
                    raise ValueError(f"quarantine destination exists: {destination}")
                hashes[role] = sha256_file(source)
                shutil.move(source, destination)
                moves.append((source, destination))
                if sha256_file(destination) != hashes[role]:
                    raise RuntimeError("content changed during quarantine")
            records.append({"sample_id": sample_id, "content_sha256": hashes})
    except BaseException:
        for source, destination in reversed(moves):
            if destination.exists() and not source.exists():
                shutil.move(destination, source)
        raise
    after = {
        role: {path.stem for path in (dataset_root / role).glob("*.png")}
        for role in ROLES
    }
    if any(after[role] != after["gt"] for role in ROLES):
        raise RuntimeError("triplet membership differs after quarantine")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "before_count": len(current["gt"]),
        "candidate_count": len(values),
        "retained_count": len(after["gt"]),
        "dataset": str(dataset_root),
        "quarantine": str(quarantine_root.resolve()),
        "records": records,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def publish_release(
    source_root: Path | Sequence[Path],
    accepted_records: Path | Sequence[Path],
    destination: Path,
    receipt_root: Path,
    materialize: str = "hardlink",
    replace: bool = False,
    workers: int = 32,
) -> dict[str, Any]:
    if materialize not in {"hardlink", "copy"}:
        raise ValueError("materialize must be hardlink or copy")
    roots = [source_root] if isinstance(source_root, Path) else list(source_root)
    manifests = (
        [accepted_records]
        if isinstance(accepted_records, Path)
        else list(accepted_records)
    )
    if not roots or len(roots) != len(manifests):
        raise ValueError("source roots and accepted manifests must be paired")
    entries: list[tuple[dict[str, Any], Path]] = []
    for root, manifest in zip(roots, manifests):
        resolved_root = root.resolve(strict=True)
        entries.extend(
            (row, resolved_root) for row in iter_jsonl(manifest.resolve(strict=True))
        )
    if not entries:
        raise ValueError("accepted record manifest is empty")
    selected: list[tuple[dict[str, Any], Path]] = []
    by_id: dict[str, tuple[str, str, str]] = {}
    by_content: dict[tuple[str, str, str], str] = {}
    duplicate_exclusions: list[dict[str, str]] = []
    for row, root in entries:
        sample_id = str(row.get("sample_id") or "")
        hashes = row.get("content_sha256") or {}
        signature = tuple(str(hashes.get(role) or "") for role in ROLES)
        if not sample_id or any(len(value) != 64 for value in signature):
            raise ValueError("accepted record identity or content hash is invalid")
        if sample_id in by_id:
            if by_id[sample_id] != signature:
                raise ValueError(f"same sample ID has conflicting content: {sample_id}")
            duplicate_exclusions.append(
                {"sample_id": sample_id, "canonical": sample_id}
            )
            continue
        if signature in by_content:
            duplicate_exclusions.append(
                {"sample_id": sample_id, "canonical": by_content[signature]}
            )
            continue
        by_id[sample_id] = signature
        by_content[signature] = sample_id
        selected.append((row, root))
    destination = destination.resolve()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.staging.{os.getpid()}"
    if staging.exists():
        raise ValueError(f"staging path exists: {staging}")
    for role in ROLES:
        (staging / role).mkdir(parents=True)
    try:
        tasks: list[tuple[Path, Path, str, str, str]] = []
        for row, source_root in selected:
            sample_id = str(row["sample_id"])
            hashes = row.get("content_sha256") or {}
            for role in ROLES:
                source = source_root / role / f"{sample_id}.png"
                target = staging / role / source.name
                tasks.append((source, target, str(hashes[role]), sample_id, role))

        def materialize_one(task: tuple[Path, Path, str, str, str]) -> None:
            source, target, expected, sample_id, role = task
            mode = source.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(f"invalid release source: {sample_id}:{role}")
            if materialize == "hardlink":
                os.link(source, target)
            else:
                shutil.copy2(source, target)
            if sha256_file(target) != expected:
                raise RuntimeError(f"published hash mismatch: {sample_id}:{role}")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            list(executor.map(materialize_one, tasks))
        readme = (
            "# DriveHarm audited pair release\n\n"
            f"Triplets: {len(selected):,}\n\n"
            "Each sample has matching names under `gt`, `input`, and `target`. "
            "Every image passed identity, geometry, grounding, orientation, foreground-occlusion, "
            "decode, dimensions, hash and duplicate checks. All six nuScenes cameras are eligible.\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8")
        backup: Path | None = None
        if destination.exists():
            if not replace:
                raise ValueError(f"destination exists: {destination}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = parent / f".{destination.name}.previous.{stamp}"
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "input_record_count": len(entries),
        "triplet_count": len(selected),
        "image_count": len(selected) * len(ROLES),
        "duplicate_exclusion_count": len(duplicate_exclusions),
        "duplicate_exclusions": duplicate_exclusions,
        "destination": str(destination),
        "materialize": materialize,
        "materialize_workers": max(1, workers),
        "previous_release": str(backup) if backup is not None else None,
    }
    receipt_root.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_root / "publish.json", receipt)
    return receipt
