# DriveHarm

DriveHarm is the compact production implementation of the six-camera nuScenes
STORM asset re-insertion workflow used to build `gt/input/target` harmonization
triplets. The repository contains code and documentation only. Assets,
checkpoints, nuScenes images, generated pairs and audit logs stay outside Git.

The code is organized as a reusable pair-production core with a frozen train
production profile. Dataset adapters provide either the native train capacity
manifest or normalized observation JSONL; renderer implementations satisfy one
small executable/result contract; GPU topology, split destination and release
sources are runtime inputs. New splits or compatible datasets therefore reuse
planning, review, composition, full audit and atomic publication instead of
copying the pipeline.

The immutable upstream asset generator is intentionally outside this project.
DriveHarm starts from its completed, SHA-256-bound hand-off and never changes
how PLY assets are produced.

The authoritative behavior is the production flow that created
`nusc_pair/train`. The validation split is not used as a generation template.
One bug found while auditing validation data is retained only as a regression
check at the final quality gate.

## Pair definition

- `gt`: aligned real sensor RGB, retained for real-domain evaluation.
- `input`: actor-removed STORM background plus the exact canonical Gaussian
  asset, composed back-to-front with physically verified foreground occlusion.
- `target`: unedited STORM render from the identical checkpoint, camera and
  exposure timestamp.

An asset visible in any of the six cameras is eligible. Camera 0 has no special
status and is not required.

## Production stages

1. Verify the upstream asset manifest hash, every PLY hash, `obj_id`, official
   `instance_token`, category, official dimensions and canonical `+X` axis.
2. Read the train `exact_asset_windows.jsonl`, validate its temporal windows,
   and build deterministic single-asset plus compatible multi-asset render
   opportunities for every visible camera.
3. Review each unique asset UID once, then propagate its decision to every
   frame/camera/combination. Bounded asynchronous workers use the official
   OpenAI Python client and a strict JSON schema. Hash-bound completed decisions
   are reused; request errors remain pending and are requested again.
4. Verify the exact STORM, CVAC and DCN artifact hashes, split accepted jobs
   across a shared async shard queue. Eight GPUs and two workers per GPU are the
   train production profile; each worker pulls several smaller shards so a fast
   card does not sit idle behind another card's long tail. Jobs from one scene
   window stay together to reuse STORM context, and hash-valid completed shards
   resume without another GPU pass.
5. Compose premultiplied asset layers far-to-near. A foreground mask is applied
   only when an independently bound official instance is distinct from and
   strictly nearer than the inserted target. Target-instance support is never
   allowed to erase that physical ordering.
   The real camera timestamp and normalized STORM time are bound in every row;
   actor removal must be effective and unchanged outside its exact edit mask.
6. Apply the same reasonable geometry policy as the validated train release:
   area-aware silhouette/box/orientation limits, official 3-D ground lock,
   15% minimum complete-asset visibility and hard rejection of broken, doubled
   or reversed assets. Color, brand, trim and ordinary lighting differences are
   audit signals, not rejection criteria.
   Multi-asset edit overlap is capped at the train value of 2% of the smaller
   edit mask, and both removal and insertion must change at least 20 pixels.
7. Review every rendered triplet, then independently check every PNG, content
   hash, triplet membership, duplicate signature, identity receipt, geometry
   receipt and occlusion receipt. Every candidate is excluded from acceptance.
8. Materialize only the accepted manifest into a temporary sibling directory,
   verify hashes again, and atomically activate the release.

## Train parity matrix

| train production capability | implementation |
|---|---|
| immutable original-asset boundary | `contracts.py` verifies the hand-off, exact manifest and every PLY hash; asset generation is untouched |
| temporal accepted windows and capacity | `planning.py` reads train `exact_asset_windows.jsonl` and validates context/target frames |
| all six cameras, no front-only condition | every `visible_target_frame_camera_key` becomes an opportunity |
| single and multi-asset variants | deterministic subsets are planned once per window, never repeat an official instance, and are projected only where every selected actor is visible; multi-asset masks must pass 2% overlap |
| exact identity and canonical heading | `obj_id`, `instance_token`, category, PLY hash, official dimensions and `+X` are cross-bound |
| identity visual review | `review.py` uses bounded async workers and strict JSON-schema decisions |
| STORM/CVAC/DCN consistency | `render.py` verifies the artifact contract and job/frame/camera binding, then batches across eight GPUs |
| actor removal and same-domain pair | `compose.py` binds camera/STORM time and requires a hash-bound usable cleanup receipt, exact edit mask, STORM target-quality pass and effective removal |
| scale, pose, grounding and integrity | area-aware train limits plus official dimensions, bottom lock, orientation and broken/doubled checks |
| physical foreground occlusion | distinct official instance, independent mask, strictly nearer depth, 4%/16-pixel materiality and direct alpha clipping |
| complete asset visibility | full alpha is retained unless physical foreground evidence exists; visible fraction must be at least 15% |
| complete release audit | `audit.py` checks every image, record, role membership, hash, duplicate, identity, geometry, occlusion and visual decision |
| old/new audited union | `release.py` merges repeated source/manifest pairs, excludes duplicate IDs/content, verifies again and atomically activates |
| direct candidate removal | `quarantine` moves all three matching files together with a recoverable hash receipt |

The frozen geometry limits are:

| projected pixels | IoU | center px | width/height ratio | yaw |
|---:|---:|---:|---:|---:|
| `<100` | 0.30 | 12 | 0.70–1.40 | 25° |
| `<400` | 0.33 | 11 | 0.72–1.37 | 25° |
| `<1600` | 0.36 | 10 | 0.75–1.33 | 24° |
| `>=1600` | 0.40 | 10 | 0.78–1.30 | 22° |

Foreground occlusion uses the train release's 4% materiality floor with a
minimum of 16 pixels. This catches the val-only regression in which a distinct
nearer official instance was incorrectly suppressed by target-instance mask
protection. It also catches material restoration of occlusion already verified
by the renderer.

## External contracts

The asset hand-off JSON points to the immutable exact-asset JSON used by train,
or to an equivalent normalized JSONL manifest:

```json
{
  "status": "complete",
  "manifest": "/data/drivelab_asset.json",
  "manifest_sha256": "<sha256>",
  "quarantine_manifest": "/data/asset_quarantine.json",
  "quarantine_manifest_sha256": "<sha256>"
}
```

The native train registry `bridge_summary.json` is also accepted directly; its
`outputs.exact_asset_manifest.path/sha256` fields are interpreted as the same
immutable hand-off. A capacity summary carrying
`exact_asset_manifest`/`exact_asset_manifest_sha256` is accepted as well.

The existing train shape (`global_uid`, `obj_id`, `instance_token`, `category`,
`asset_path`, `size_xyz`, and the canonical hash/axis object) is accepted
directly. Official `wlh` dimensions come from each train capacity candidate and
are converted to renderer `length,width,height` order. The original identity
`review_manifest.jsonl` supplies the source view, canonical views, heading view
and their hashes. The quarantine fields are optional; when supplied, they are
hash-verified and excluded before any identity review or rendering.

An adapter for another compatible source emits one row per visible actor with
`scene_name`, `window_id`, `frame_index`, `obj_id`, `instance_token`,
`area_pixels`, `visibility_level`, `visible_cameras`, official dimensions and
hash-bound review images. Visibility may contain any subset of camera `0`–`5`.
The planner groups variants by official instance, applies the train deterministic
seed and 256-combination window cap, and never combines two variants of the same
instance.

The render contract binds code/checkpoint artifacts before any GPU starts:

```json
{
  "artifacts": {
    "storm": {"path": "/models/storm.ckpt", "sha256": "<sha256>"},
    "cvac": {"path": "/models/cvac.ckpt", "sha256": "<sha256>"},
    "dcn": {"path": "/models/dcn.ckpt", "sha256": "<sha256>"}
  }
}
```

The company renderer is an executable accepting:

```text
--jobs MANIFEST --output-root DIRECTORY --results RESULTS_JSONL --gpu 0
```

DriveHarm exposes exactly one physical GPU to each process through
`CUDA_VISIBLE_DEVICES`. Each result row must have `status=complete`, matching
`sample_id`, job hash, checkpoint hashes, hash-bound `real_gt`,
`storm_baseline`, `actor_removed`, exact removal-mask paths and one or more
exact-identity asset layers. Layer receipts carry projection
metrics plus independently verified foreground masks and official-instance
depth decisions.

## Installation and review server

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

For a local multimodal review service, start vLLM with the company-approved
model and reserve 90% of each selected GPU's memory:

```bash
vllm serve /models/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen3-VL-8B-Instruct \
  --gpu-memory-utilization 0.90
```

The review implementation uses `AsyncOpenAI`, awaits
`client.chat.completions.create`, and requires JSON-schema output. Set the key
in an environment variable; do not place credentials in manifests or command
history.

## End-to-end run

```bash
export OPENAI_API_KEY=local
driveharm run \
  --asset-post /data/asset_post.json \
  --observations /data/exact_asset_windows.jsonl \
  --identity-manifest /data/review_manifest.jsonl \
  --renderer /company/bin/storm_pair_renderer \
  --render-contract /data/render_contract.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 2 \
  --shards-per-worker 4 \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-8B-Instruct \
  --review-concurrency 16 \
  --work-root /data/driveharm_run \
  --destination /data/nusc_pair/train
```

Every stage is also available separately through `plan`, `review`, `render`,
`compose`, `audit`, `quarantine`, and `release`. Run `driveharm COMMAND --help`
for its exact arguments.

For a multi-batch train union, repeat `--source-root` and
`--accepted-records` in matching order on the `release` command. Equal triplet
content is retained once; conflicting content under the same sample ID stops
publication.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the native train exact-asset/capacity/identity shapes,
all-camera planning, single and combined jobs, checkpoint and job binding,
area-aware orientation rejection, 2% edit overlap, exact-mask/nearer-occluder
ordering, actor-removal locality, composition, independent audit, publication,
and recoverable triplet quarantine. They also enforce the compact file count
and asynchronous OpenAI JSON-schema contract.

## Verified reference release

The implementation preserves the policies used for
`/mnt/ojc/workplace2/dataset/nusc_pair/train`: 51,132 triplets from 616 official
train scenes, with all 153,396 images decoded and hash-checked after release.
The frozen receipt is
`/mnt/ojc/workplace2/sixcam_run/nusc_pair_release_v9_official_train_val_v1/summary.json`.
The four area bins above were also compared directly against the original train
producer function and matched exactly. The distinct-nearer occluder case is an
additional regression test and does not change the train generation stages or
their reasonable thresholds.
