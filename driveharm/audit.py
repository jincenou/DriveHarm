"""Independent full-release audit for identity, geometry, occlusion and content."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import stat
from typing import Any

from PIL import Image

from .compose import geometry_pass
from .contracts import (
    CAMERAS,
    IMAGE_SIZE,
    ROLES,
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    indexed_rows,
    iter_jsonl,
    sha256_file,
)
from .review import hard_failure


def _inspect(task: tuple[str, str, Path, str]) -> dict[str, Any]:
    sample_id, role, path, expected = task
    errors: list[str] = []
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            return {
                "sample_id": sample_id,
                "role": role,
                "errors": ["not_regular_file"],
            }
        if sha256_file(path) != expected:
            errors.append("hash_mismatch")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                errors.append("not_png")
            if image.mode != "RGB":
                errors.append("not_rgb")
            if image.size != IMAGE_SIZE:
                errors.append("wrong_dimensions")
    except Exception as exception:
        errors.append(f"{type(exception).__name__}: {exception}")
    return {"sample_id": sample_id, "role": role, "errors": errors}


def _record_errors(row: dict[str, Any], visual: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    sample_id = str(row.get("sample_id") or "")
    if not sample_id or str(row.get("camera_id")) not in CAMERAS:
        errors.append("invalid_identity_or_camera")
    layers = row.get("asset_layers") or []
    if not layers:
        errors.append("no_asset_layers")
    for layer in layers:
        if not all(
            str(layer.get(key) or "")
            for key in ("obj_id", "instance_token", "asset_sha256")
        ):
            errors.append("incomplete_asset_identity")
        dimensions = layer.get("official_dimensions_m") or []
        if (
            len(dimensions) != 3
            or any(float(value) <= 0 for value in dimensions)
            or layer.get("forward_axis") != "+X"
        ):
            errors.append("invalid_dimensions_or_axis")
        if not geometry_pass(layer.get("quality") or {}):
            errors.append("geometry_candidate")
        occlusion = layer.get("occlusion") or {}
        if occlusion.get("renderer_occlusion_regression") is True:
            errors.append("occlusion_regression_candidate")
        if int(occlusion.get("pixels") or 0) and occlusion.get("applied") is not True:
            errors.append("unverified_partial_occlusion")
    review = visual.get(sample_id)
    if review is not None:
        unsigned_review = dict(review)
        claimed_review = str(unsigned_review.pop("decision_sha256", ""))
        decision = review.get("decision") or {}
        if claimed_review != canonical_sha256(unsigned_review) or bool(
            review.get("hard_failure")
        ) is not hard_failure(decision):
            errors.append("visual_review_binding_mismatch")
        elif review.get("hard_failure") is True:
            errors.append("visual_review_candidate")
    unsigned = dict(row)
    claimed = str(unsigned.pop("record_sha256", ""))
    if claimed and claimed != canonical_sha256(unsigned):
        errors.append("record_binding_mismatch")
    return sorted(set(errors))


def audit_release(
    dataset_root: Path,
    records_path: Path,
    output_root: Path,
    visual_decisions: Path | None = None,
    workers: int = 48,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    records = list(iter_jsonl(records_path.resolve(strict=True)))
    ids = [str(row.get("sample_id") or "") for row in records]
    if not records or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("records contain empty or duplicate sample IDs")
    names = {f"{sample_id}.png" for sample_id in ids}
    for role in ROLES:
        root = dataset_root / role
        observed = {path.name for path in root.iterdir() if path.is_file()}
        if observed != names:
            raise ValueError(f"{role} membership differs from records")
    visual = indexed_rows(visual_decisions, "sample_id") if visual_decisions else {}
    if visual_decisions and set(visual) != set(ids):
        raise ValueError("visual review membership differs from records")
    tasks: list[tuple[str, str, Path, str]] = []
    for row in records:
        hashes = row.get("content_sha256") or {}
        for role in ROLES:
            expected = str(hashes.get(role) or "")
            if len(expected) != 64:
                raise ValueError(
                    f"incomplete content hash: {row.get('sample_id')}:{role}"
                )
            tasks.append(
                (
                    str(row["sample_id"]),
                    role,
                    dataset_root / role / f"{row['sample_id']}.png",
                    expected,
                )
            )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        image_results = list(executor.map(_inspect, tasks))
    image_errors: dict[str, list[str]] = {}
    for result in image_results:
        if result["errors"]:
            image_errors.setdefault(result["sample_id"], []).extend(result["errors"])
    signatures: dict[tuple[str, str, str], list[str]] = {}
    for row in records:
        hashes = row["content_sha256"]
        signature = tuple(str(hashes[role]) for role in ROLES)
        signatures.setdefault(signature, []).append(str(row["sample_id"]))
    duplicate_ids = {
        sample_id
        for group in signatures.values()
        if len(group) > 1
        for sample_id in group[1:]
    }
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row["sample_id"])
        reasons = _record_errors(row, visual) + image_errors.get(sample_id, [])
        if sample_id in duplicate_ids:
            reasons.append("duplicate_triplet_content")
        if reasons:
            candidates.append({"sample_id": sample_id, "reasons": sorted(set(reasons))})
        else:
            accepted.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    accepted_path = output_root / "accepted_records.jsonl"
    candidate_path = output_root / "candidates.jsonl"
    atomic_jsonl(accepted_path, accepted)
    atomic_jsonl(candidate_path, candidates)
    summary = {
        "schema_version": 1,
        "status": "pass" if not candidates else "candidates_excluded_from_acceptance",
        "record_count": len(records),
        "image_count": len(image_results),
        "all_images_checked": len(image_results) == len(records) * len(ROLES),
        "accepted_count": len(accepted),
        "candidate_count": len(candidates),
        "candidate_policy": "exclude",
        "duplicate_triplet_count": len(duplicate_ids),
        "accepted_records": str(accepted_path.resolve()),
        "accepted_records_sha256": canonical_sha256(accepted),
        "candidates": str(candidate_path.resolve()),
        "candidates_sha256": canonical_sha256(candidates),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary
