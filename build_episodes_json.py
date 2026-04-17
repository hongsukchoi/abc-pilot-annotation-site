"""Generate episodes.json for the pilot static site from the materialized
bundles + rendered mp4s.

Usage:
    uv run python pilot_site/build_episodes_json.py \
        --bundle-root pilot_data \
        --render-root pilot_renders \
        --output pilot_site/episodes.json
"""

import argparse
import json
import subprocess
from pathlib import Path


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ], text=True).strip()
        return float(out) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-root", default="pilot_data")
    p.add_argument("--render-root", default="pilot_renders")
    p.add_argument("--output", default="pilot_site/episodes.json")
    args = p.parse_args()

    bundle_root = Path(args.bundle_root)
    render_root = Path(args.render_root)

    entries = []
    for bundle_dir in sorted(bundle_root.iterdir()):
        if not bundle_dir.is_dir():
            continue
        info_path = bundle_dir / "bundle_info.json"
        labels_path = bundle_dir / "labels.json"
        if not info_path.exists() or not labels_path.exists():
            continue

        with open(info_path) as f:
            info = json.load(f)
        with open(labels_path) as f:
            labels = json.load(f)

        ep_id = info["episode_id"]
        mp4_path = render_root / f"{ep_id}.mp4"
        if not mp4_path.exists():
            print(f"  skip {ep_id}: no rendered mp4")
            continue

        subtasks = labels.get("subtasks") or {}
        segs = subtasks.get("segments", []) or []
        duration_s = ffprobe_duration(mp4_path)
        mp4_size = mp4_path.stat().st_size

        subtask_list = []
        for s in segs:
            label = str(s.get("label", ""))
            subtask_list.append({
                "label": label,
                "start_time": s.get("start_time_ns", 0) / 1e9,
                "end_time": s.get("end_time_ns", 0) / 1e9,
                "is_special": label.lower().startswith("special="),
            })

        entries.append({
            "episode_id": ep_id,
            "task_folder": info["task_folder"],
            "task_name": info["task_name"],
            "instruction": info["instruction"],
            "source_dataset": info["source_dataset"],
            "mp4_filename": mp4_path.name,
            "mp4_size_bytes": mp4_size,
            "duration_seconds": duration_s,
            "n_subtasks": len(segs),
            "n_special_subtasks": sum(1 for s in subtask_list if s["is_special"]),
            "subtasks": subtask_list,
        })

    out = {
        "generated_from": {
            "bundle_root": str(bundle_root),
            "render_root": str(render_root),
        },
        "n_episodes": len(entries),
        "episodes": entries,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}: {len(entries)} episodes")


if __name__ == "__main__":
    main()
