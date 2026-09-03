"""Bounded asynchronous identity and rendered-pair visual review."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .contracts import (
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    indexed_rows,
    iter_jsonl,
)


DECISION_SCHEMA: dict[str, Any] = {
    "name": "driveharm_quality_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["accept", "reject", "unverifiable"],
            },
            "same_identity": {"type": "boolean"},
            "broken_or_doubled_asset": {"type": "boolean"},
            "orientation_valid": {"type": "boolean"},
            "scale_valid": {"type": "boolean"},
            "grounding_valid": {"type": "boolean"},
            "severe_occlusion_error": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": [
            "decision",
            "same_identity",
            "broken_or_doubled_asset",
            "orientation_valid",
            "scale_valid",
            "grounding_valid",
            "severe_occlusion_error",
            "confidence",
            "reason",
        ],
    },
}


def _image_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _content(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = (
        "Audit this exact-identity driving asset or triplet. Reject only clear hard failures: "
        "wrong identity/category, broken or doubled geometry, reversed orientation, severe scale "
        "or grounding error, or physically impossible foreground occlusion. Ordinary color, brand, "
        "trim, lighting and minor boundary differences are acceptable. Return only the schema.\n"
        f"Metadata: {json.dumps(row.get('review_context', {}), ensure_ascii=False)}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    images = row.get("review_images") or {}
    if isinstance(images, dict):
        for label, raw_path in sorted(images.items()):
            path = Path(str(raw_path)).resolve(strict=True)
            content.append({"type": "text", "text": str(label)})
            content.append(
                {"type": "image_url", "image_url": {"url": _image_url(path)}}
            )
    if len(content) == 1:
        raise ValueError(f"review item has no images: {row.get('sample_id')}")
    return content


def hard_failure(decision: dict[str, Any]) -> bool:
    return bool(
        decision.get("decision") != "accept"
        or decision.get("same_identity") is not True
        or decision.get("broken_or_doubled_asset") is True
        or decision.get("orientation_valid") is not True
        or decision.get("scale_valid") is not True
        or decision.get("grounding_valid") is not True
        or decision.get("severe_occlusion_error") is True
    )


async def review_manifest(
    manifest: Path,
    output_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = 16,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    rows = list(iter_jsonl(manifest.resolve(strict=True)))
    ids = [str(row.get("sample_id") or row.get("asset_id") or "") for row in rows]
    if not rows or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("review manifest has empty or duplicate identities")
    decisions_path = output_root / "decisions.jsonl"
    previous = (
        indexed_rows(decisions_path, "sample_id") if decisions_path.is_file() else {}
    )
    reused = 0
    queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
    results: list[dict[str, Any] | None] = [None] * len(rows)
    for index, row in enumerate(rows):
        prior = previous.get(ids[index])
        if prior is not None:
            unsigned = dict(prior)
            claimed = str(unsigned.pop("decision_sha256", ""))
            if (
                prior.get("request_sha256") == canonical_sha256(row)
                and prior.get("request_error") is None
                and claimed == canonical_sha256(unsigned)
            ):
                results[index] = prior
                reused += 1
                continue
        queue.put_nowait((index, row))
    pending_count = len(rows) - reused
    worker_count = min(concurrency, pending_count)
    for _ in range(worker_count):
        queue.put_nowait(None)

    client = (
        AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        if pending_count
        else None
    )

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                index, row = item
                identity = ids[index]
                try:
                    if client is None:
                        raise RuntimeError("review client is unavailable")
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": _content(row)}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": DECISION_SCHEMA,
                        },
                        temperature=0,
                        max_tokens=500,
                    )
                    raw = response.choices[0].message.content or ""
                    decision = json.loads(raw)
                    if set(decision) != set(DECISION_SCHEMA["schema"]["required"]):
                        raise ValueError("model response keys differ from schema")
                    error = None
                except Exception as exception:
                    decision = {
                        "decision": "unverifiable",
                        "same_identity": False,
                        "broken_or_doubled_asset": False,
                        "orientation_valid": False,
                        "scale_valid": False,
                        "grounding_valid": False,
                        "severe_occlusion_error": False,
                        "confidence": 0.0,
                        "reason": "review request failed",
                    }
                    error = f"{type(exception).__name__}: {exception}"
                result = {
                    "schema_version": 1,
                    "sample_id": identity,
                    "request_sha256": canonical_sha256(row),
                    "decision": decision,
                    "hard_failure": hard_failure(decision),
                    "request_error": error,
                }
                result["decision_sha256"] = canonical_sha256(result)
                results[index] = result
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
    await queue.join()
    await asyncio.gather(*tasks)
    if client is not None:
        await client.close()
    completed = [row for row in results if row is not None]
    if len(completed) != len(rows):
        raise RuntimeError("review workers did not cover the manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(decisions_path, completed)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "reviewed_count": len(completed),
        "reused_count": reused,
        "requested_count": pending_count,
        "accepted_count": sum(not row["hard_failure"] for row in completed),
        "candidate_count": sum(bool(row["hard_failure"]) for row in completed),
        "request_error_count": sum(
            row["request_error"] is not None for row in completed
        ),
        "decisions": str(decisions_path.resolve()),
        "decisions_sha256": canonical_sha256(completed),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary
