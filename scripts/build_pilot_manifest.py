"""Build a manifest of abc-pilot annotation episodes that we located in one of
the YAM dataset buckets.

Cross-references an annotation delivery on s3://xdof-bair-abc/data/deliveries/
against the collected.json files of candidate dataset buckets, and writes a
JSON manifest with one entry per episode for which both the annotation and
the source dataset files exist.

Usage:
    AWS_PROFILE=<xdof-profile> uv run python scripts/build_pilot_manifest.py \
        --preset finetuned --output pilot_manifest.json
    AWS_PROFILE=<xdof-profile> uv run python scripts/build_pilot_manifest.py \
        --preset pretrained --output pilot_manifest_pretrained.json
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

# (manifest_name, dataset_path_under_robot_path)
FINETUNED_CANDIDATE_DATASETS = [
    ("perf_dexcin_20260407", "performance_tasks_dexcin_20260407_wandb"),
    ("perf_realsense_20260407", "performance_tasks_realsense_20260407_wandb"),
    ("pretrain_20260407_v2", "pretraining_tasks_20260407_wandb_v2"),
    ("perf_dexcin_20260409_224", "performance_tasks_dexcin_20260409_224"),
    ("perf_realsense_20260409_224", "performance_tasks_realsense_20260409_224"),
    ("pretrain_20260409_224", "pretraining_tasks_20260409_224"),
]

PRETRAINED_CANDIDATE_DATASETS = [
    ("pretrain_20260407_v2", "pretraining_tasks_20260407_wandb_v2"),
    ("pretrain_20260409_224", "pretraining_tasks_20260409_224"),
    ("pretrain_20260407_wandb", "pretraining_tasks_20260407_wandb"),
    ("pretrain_20260407", "pretraining_tasks_20260407"),
    ("perf_dexcin_20260407", "performance_tasks_dexcin_20260407_wandb"),
    ("perf_realsense_20260407", "performance_tasks_realsense_20260407_wandb"),
    ("perf_dexcin_20260409_224", "performance_tasks_dexcin_20260409_224"),
    ("perf_realsense_20260409_224", "performance_tasks_realsense_20260409_224"),
]

DATASET_ROOT = "s3://far-research-internal/datasets/robot_path"

PRESETS = {
    "finetuned": {
        "annotation_s3": "s3://xdof-bair-abc/data/deliveries/abc_pilot_sample_annotation_20260416/",
        "candidates": FINETUNED_CANDIDATE_DATASETS,
    },
    "pretrained": {
        "annotation_s3": "s3://xdof-bair-abc/data/deliveries/abc_pilot_pretraining_20260424/",
        "candidates": PRETRAINED_CANDIDATE_DATASETS,
    },
}


def list_annotations(profile: str, annotation_s3: str) -> dict[str, dict]:
    """Returns {episode_id: {task_folder, annotation_s3, mcap_size}}."""
    cmd = [
        "aws", "s3", "ls", "--recursive", "--profile", profile, annotation_s3,
    ]
    out = subprocess.check_output(cmd, text=True)
    # Match: any prefix path that ends with .../<task>/<episode>/output.mcap
    delivery_prefix = annotation_s3.replace("s3://", "").split("/", 1)[1]
    pat = re.compile(
        r"^\S+\s+\S+\s+(\d+)\s+"
        + re.escape(delivery_prefix)
        + r"([^/]+)/([^/]+)/output\.mcap$"
    )
    eps = {}
    for line in out.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        size, task, ep = m.group(1), m.group(2), m.group(3)
        eps[ep] = {
            "task_folder": task,
            "annotation_s3": (
                f"{annotation_s3.rstrip('/')}/{task}/{ep}/output.mcap"
            ),
            "mcap_size_bytes": int(size),
        }
    return eps


def fetch_collected(dataset_path: str, cache_dir: Path,
                    profile: str | None = None) -> dict:
    local = cache_dir / f"{dataset_path}__collected.json"
    if not local.exists():
        s3 = f"{DATASET_ROOT}/{dataset_path}/metadata/collected.json"
        cmd = ["aws", "s3", "cp", "--quiet", s3, str(local)]
        if profile:
            cmd.extend(["--profile", profile])
        subprocess.check_call(cmd)
    with open(local) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="finetuned",
                   help="Annotation delivery + candidate dataset list to use.")
    p.add_argument("--annotation-profile", default="xdof",
                   help="AWS profile with read access to xdof-bair-abc.")
    p.add_argument("--source-profile", default=None,
                   help="AWS profile for far-research-internal "
                        "(default: inherit AWS_PROFILE).")
    p.add_argument("--output", default="pilot_manifest.json")
    p.add_argument("--cache-dir", default="/tmp/pilot_manifest_cache")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    annotation_s3 = preset["annotation_s3"]
    candidates = preset["candidates"]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preset={args.preset}] listing annotations under {annotation_s3} ...")
    ann = list_annotations(args.annotation_profile, annotation_s3)
    print(f"  found {len(ann)} annotated episodes "
          f"across {len({v['task_folder'] for v in ann.values()})} task folders")

    print("Loading collected.json from candidate datasets ...")
    membership: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for manifest_name, dataset_path in candidates:
        try:
            coll = fetch_collected(dataset_path, cache_dir,
                                   profile=args.source_profile)
        except subprocess.CalledProcessError as e:
            print(f"  skip {manifest_name}: {e}")
            continue
        n_hits = 0
        for ep_id in ann:
            if ep_id in coll:
                membership[ep_id].append(
                    (manifest_name, dataset_path, coll[ep_id])
                )
                n_hits += 1
        print(f"  {manifest_name}: {n_hits} of {len(ann)} annotated episodes present")

    entries = []
    missing = []
    for ep_id, info in sorted(ann.items()):
        cands = membership.get(ep_id, [])
        if not cands:
            missing.append((info["task_folder"], ep_id))
            continue
        manifest_name, dataset_path, coll_meta = cands[0]
        entries.append({
            "episode_id": ep_id,
            "task_folder": info["task_folder"],
            "task_name": coll_meta.get("task_name"),
            "annotation_s3": info["annotation_s3"],
            "mcap_size_bytes": info["mcap_size_bytes"],
            "source_dataset": manifest_name,
            "source_data_s3": (
                f"{DATASET_ROOT}/{dataset_path}/data/{ep_id}"
            ),
            "source_dataset_path": dataset_path,
            "alternate_datasets": [d for d, _, _ in cands[1:]],
        })

    out = {
        "annotation_root": annotation_s3,
        "candidate_datasets": [
            {"name": n, "dataset_path": d} for n, d in candidates
        ],
        "n_total_annotated": len(ann),
        "n_resolved": len(entries),
        "n_missing": len(missing),
        "missing": [
            {"task_folder": t, "episode_id": e} for t, e in missing
        ],
        "episodes": entries,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}: {len(entries)} resolved, {len(missing)} missing")


if __name__ == "__main__":
    main()
