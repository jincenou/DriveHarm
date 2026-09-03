"""Build deterministic six-camera render opportunities from visible observations."""

from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from .contracts import (
    CAMERAS,
    AssetBinding,
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    indexed_rows,
    iter_jsonl,
    load_asset_post,
    sha256_file,
)


def _window_selections(
    members: dict[str, dict[str, Any]], maximum: int, seed: int, window_key: str
) -> list[tuple[str, ...]]:
    """Reproduce train's instance-safe deterministic subset expansion."""
    by_instance: dict[str, list[str]] = {}
    for obj_id, job in sorted(members.items()):
        instance = str(job["assets"][0]["instance_token"])
        by_instance.setdefault(instance, []).append(obj_id)
    choices = [[None, *by_instance[key]] for key in sorted(by_instance)]
    selections = {
        tuple(sorted(str(value) for value in values if value is not None))
        for values in product(*choices)
        if any(value is not None for value in values)
    }
    mandatory = {(obj_id,) for obj_id in members}
    if maximum and len(selections) > maximum:
        remaining = sorted(
            selections - mandatory,
            key=lambda values: canonical_sha256(
                {
                    "seed": seed,
                    "window": window_key,
                    "selected_obj_ids": values,
                }
            ),
        )
        selections = mandatory | set(remaining[: max(0, maximum - len(mandatory))])
    return sorted(selections, key=lambda values: (len(values), values))


def _cameras(row: dict[str, Any]) -> list[str]:
    raw = row.get("visible_cameras")
    if raw is None and isinstance(row.get("camera_observations"), dict):
        raw = [key for key, value in row["camera_observations"].items() if value]
    values = sorted({str(value).removeprefix("c") for value in (raw or [])})
    if any(value not in CAMERAS for value in values):
        raise ValueError(f"invalid camera list: {values}")
    return values


def _observations(path: Path):
    """Read normalized rows or the train pipeline's exact-asset capacity rows."""
    for row in iter_jsonl(path.resolve(strict=True)):
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            yield row
            continue
        scene = str(row.get("scene_name") or "")
        context_frames = [int(value) for value in row.get("context_frames") or []]
        target_frames = [int(value) for value in row.get("target_frames") or []]
        if (
            not scene
            or not context_frames
            or not target_frames
            or context_frames != sorted(set(context_frames))
            or target_frames != sorted(set(target_frames))
            or int(row.get("context_start", -1)) != context_frames[0]
        ):
            raise ValueError("invalid train temporal-window contract")
        window = f"w{int(row.get('context_start', 0)):06d}"
        for candidate in candidates:
            raw_size = candidate.get("official_size_wlh_m") or []
            official_dimensions = (
                [float(raw_size[1]), float(raw_size[0]), float(raw_size[2])]
                if len(raw_size) == 3
                else []
            )
            keys = candidate.get("visible_target_frame_camera_keys") or [
                f"{int(candidate.get('target_frame', -1)):06d}:c{candidate.get('camera_id')}"
            ]
            for key in keys:
                raw_frame, raw_camera = str(key).split(":c", 1)
                yield {
                    "scene_name": scene,
                    "window_id": window,
                    "frame_index": int(raw_frame),
                    "obj_id": candidate.get("obj_id"),
                    "instance_token": candidate.get("instance_token"),
                    "area_pixels": candidate.get("projected_area_native_pixels", 0),
                    "visibility_level": candidate.get("visibility_token", 0),
                    "visible_cameras": [raw_camera],
                    "official_dimensions_m": official_dimensions,
                    "review_images": candidate.get("review_images") or {},
                    "train_capacity_candidate": candidate,
                    "train_capacity_window": {
                        "context_start": row.get("context_start"),
                        "context_frames": context_frames,
                        "target_frames": target_frames,
                        "quality_tier": row.get("quality_tier"),
                        "report": row.get("report"),
                    },
                }


def build_plan(
    asset_post: Path,
    observations: Path,
    output_root: Path,
    identity_manifest: Path | None = None,
    minimum_area_pixels: int = 30,
    minimum_visibility_level: int = 2,
    maximum_combinations_per_window: int = 256,
    combination_seed: int = 20260818,
) -> dict[str, Any]:
    if maximum_combinations_per_window < 0:
        raise ValueError("maximum combinations must be nonnegative")
    assets = load_asset_post(asset_post)
    identity_rows = (
        indexed_rows(identity_manifest.resolve(strict=True), "obj_id")
        if identity_manifest
        else {}
    )
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    camera_counts: Counter[str] = Counter()
    for observation in _observations(observations):
        obj_id = str(observation.get("obj_id") or "")
        binding: AssetBinding | None = assets.get(obj_id)
        if binding is None:
            continue
        if str(observation.get("instance_token") or "") != binding.instance_token:
            raise ValueError(f"instance token differs for {obj_id}")
        if int(observation.get("area_pixels") or 0) < minimum_area_pixels:
            continue
        if int(observation.get("visibility_level") or 0) < minimum_visibility_level:
            continue
        scene = str(observation.get("scene_name") or "")
        frame = int(observation.get("frame_index", -1))
        window = str(observation.get("window_id") or f"w{frame:06d}")
        if not scene or frame < 0:
            raise ValueError("observation has no scene/frame identity")
        identity = identity_rows.get(obj_id) or {}
        if identity_manifest and not identity:
            raise ValueError(f"train identity manifest has no row for {obj_id}")
        if identity and (
            str(identity.get("instance_token") or "") != binding.instance_token
            or str(identity.get("category") or "") != binding.category
            or str(identity.get("canonical_asset_sha256") or "") != binding.ply_sha256
        ):
            raise ValueError(
                f"train identity evidence differs from exact asset binding: {obj_id}"
            )
        review_images = observation.get("review_images") or {
            label: identity[field]
            for label, field in (
                ("identity_sheet", "review_sheet"),
                ("source", "source_raw_image"),
                ("canonical_views", "ply_evidence"),
                ("canonical_heading", "heading_evidence"),
            )
            if identity.get(field)
        }
        for label, raw_path in review_images.items():
            path = Path(str(raw_path)).resolve(strict=True)
            expected = identity.get(
                f"{dict(identity_sheet='review_sheet', canonical_views='ply_evidence', canonical_heading='heading_evidence').get(label, label)}_sha256"
            )
            if expected and sha256_file(path) != str(expected):
                raise ValueError(f"identity evidence hash mismatch: {obj_id}:{label}")
        # Every visible camera is eligible. There is deliberately no c0 condition.
        for camera in _cameras(observation):
            key = (scene, window, frame, camera, obj_id)
            if key in seen:
                raise ValueError(f"duplicate observation: {key}")
            seen.add(key)
            sample_id = (
                f"{scene}__{window}__{binding.asset_id[:12]}__f{frame:03d}_c{camera}"
            )
            official_dimensions = (
                observation.get("official_dimensions_m") or binding.dimensions_m
            )
            job = {
                "schema_version": 1,
                "sample_id": sample_id,
                "scene_name": scene,
                "window_id": window,
                "frame_index": frame,
                "camera_id": camera,
                "obj_id": obj_id,
                "instance_token": binding.instance_token,
                "category": binding.category,
                "asset_id": binding.asset_id,
                "asset_path": str(binding.ply_path),
                "asset_sha256": binding.ply_sha256,
                "official_dimensions_m": list(official_dimensions),
                "forward_axis": binding.forward_axis,
                "observation": observation,
                "review_images": review_images,
                "review_context": {
                    "stage": "asset_identity",
                    "obj_id": obj_id,
                    "instance_token": binding.instance_token,
                    "category": binding.category,
                    "official_dimensions_m": list(official_dimensions),
                    "camera_id": camera,
                    "train_identity_evidence": identity,
                },
                "selected_obj_ids": [obj_id],
                "assets": [
                    {
                        "asset_id": binding.asset_id,
                        "obj_id": obj_id,
                        "instance_token": binding.instance_token,
                        "category": binding.category,
                        "asset_path": str(binding.ply_path),
                        "asset_sha256": binding.ply_sha256,
                        "official_dimensions_m": list(official_dimensions),
                        "forward_axis": binding.forward_axis,
                    }
                ],
            }
            job["job_sha256"] = canonical_sha256(job)
            jobs.append(job)
            camera_counts[camera] += 1
    groups: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = {}
    windows: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for job in jobs:
        key = (
            job["scene_name"],
            job["window_id"],
            job["frame_index"],
            job["camera_id"],
        )
        groups.setdefault(key, {})[job["obj_id"]] = job
        window_key = (job["scene_name"], job["window_id"])
        previous = windows.setdefault(window_key, {}).setdefault(job["obj_id"], job)
        if previous["assets"] != job["assets"]:
            raise ValueError(
                f"asset binding changes inside temporal window: {window_key}"
            )
    combinations: list[dict[str, Any]] = []
    planned_by_window = {
        key: _window_selections(
            members, maximum_combinations_per_window, combination_seed, ":".join(key)
        )
        for key, members in windows.items()
    }
    for (scene, window, frame, camera), visible in groups.items():
        for selected in planned_by_window[(scene, window)]:
            if len(selected) < 2 or not set(selected).issubset(visible):
                continue
            subset = [visible[obj_id] for obj_id in selected]
            instances = [member["instance_token"] for member in subset]
            if len(instances) != len(set(instances)):
                raise ValueError("combination repeats an official instance")
            assets_for_job = sorted(
                (member["assets"][0] for member in subset),
                key=lambda row: row["obj_id"],
            )
            combination_id = canonical_sha256(
                [row["asset_id"] for row in assets_for_job]
            )[:12]
            job = {
                "schema_version": 1,
                "sample_id": f"{scene}__{window}__cmb-{combination_id}__f{frame:03d}_c{camera}",
                "scene_name": scene,
                "window_id": window,
                "frame_index": frame,
                "camera_id": camera,
                "selected_obj_ids": [row["obj_id"] for row in assets_for_job],
                "assets": assets_for_job,
                "review_images": {
                    f"asset_{index}_{label}": path
                    for index, member in enumerate(subset)
                    for label, path in sorted(member["review_images"].items())
                },
                "review_context": {
                    "stage": "multi_asset_identity",
                    "camera_id": camera,
                    "assets": assets_for_job,
                },
            }
            job["job_sha256"] = canonical_sha256(job)
            combinations.append(job)
            camera_counts[camera] += 1
    jobs.extend(combinations)
    jobs.sort(key=lambda row: row["sample_id"])
    if not jobs:
        raise ValueError("no six-camera render opportunities were accepted")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "jobs.jsonl"
    atomic_jsonl(manifest, jobs)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "job_count": len(jobs),
        "asset_count": len(
            {asset["asset_id"] for row in jobs for asset in row["assets"]}
        ),
        "combination_job_count": len(combinations),
        "maximum_combinations_per_window": maximum_combinations_per_window,
        "combination_seed": combination_seed,
        "combination_policy": "train_instance_safe_deterministic_subsets",
        "scene_count": len({row["scene_name"] for row in jobs}),
        "camera_counts": dict(sorted(camera_counts.items())),
        "camera_policy": "any_visible_camera",
        "train_identity_manifest": str(identity_manifest.resolve())
        if identity_manifest
        else None,
        "jobs": str(manifest.resolve()),
        "jobs_sha256": canonical_sha256(jobs),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary
