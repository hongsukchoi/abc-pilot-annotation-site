"""Download per-episode bundles for the abc pilot annotation set.

Reads pilot_manifest.json and, for each episode entry:
  1. Downloads the dataset files (per-camera mp4s, frame_mappings, episode_metadata,
     states_actions.bin) from far-research-internal.
  2. Downloads the annotation mcap from xdof-bair-abc and extracts the single
     /instruction message into labels.json.
  3. Optionally deletes the bulky mcap to save disk after extraction.

Output layout:
    <bundle_root>/<episode_id>/
        combined_camera-images-rgb.mp4
        top_camera-images-rgb.mp4
        left_camera-images-rgb.mp4
        right_camera-images-rgb.mp4
        *_frame_mappings.json
        episode_metadata.json
        states_actions.bin
        labels.json   <-- new (from mcap /instruction)
        bundle_info.json
        output.mcap   (kept only if --keep-mcap)

Usage:
    AWS_PROFILE=<...> uv run python deploy/scripts/materialize_pilot_bundles.py \
        --manifest pilot_manifest.json \
        --bundle-root pilot_data \
        --annotation-profile xdof
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf import descriptor_pool, message_factory

SUBTASK_TOPIC = "/subtask-raw-annotation"
DEFAULT_FPS = 30


# Files to copy from the source dataset prefix
_DATASET_FILES_REQUIRED = [
    "states_actions.bin",
    "episode_metadata.json",
]
_DATASET_FILES_OPTIONAL_VIDEOS = [
    "combined_camera-images-rgb.mp4",
    "top_camera-images-rgb.mp4",
    "left_camera-images-rgb.mp4",
    "right_camera-images-rgb.mp4",
]
_DATASET_FILES_OPTIONAL_FRAMEMAPS = [
    "combined_camera-images-rgb_frame_mappings.json",
    "top_camera-images-rgb_frame_mappings.json",
    "left_camera-images-rgb_frame_mappings.json",
    "right_camera-images-rgb_frame_mappings.json",
]


def _aws_cp(src: str, dst: str | Path, profile: str | None = None,
            quiet: bool = True) -> int:
    cmd = ["aws", "s3", "cp", src, str(dst)]
    if profile:
        cmd.extend(["--profile", profile])
    if quiet:
        cmd.append("--quiet")
    return subprocess.call(cmd)


def _aws_exists(s3_uri: str, profile: str | None = None) -> bool:
    cmd = ["aws", "s3", "ls", s3_uri]
    if profile:
        cmd.extend(["--profile", profile])
    return subprocess.call(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) == 0


def _build_instructions_decoder(schema_data: bytes):
    """Compile the protobuf FileDescriptorSet from the mcap schema and return
    a constructor for the Instructions message."""
    fds = FileDescriptorSet()
    fds.ParseFromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    for fproto in fds.file:
        try:
            pool.Add(fproto)
        except TypeError:
            pass
    msg_desc = pool.FindMessageTypeByName("Instructions")
    return message_factory.GetMessageClass(msg_desc)


def extract_instruction(mcap_path: Path) -> dict:
    """Read the single /instruction message from an output.mcap and return
    its decoded payload as {timestamp, data}."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        instr_schema = next(
            s for s in summary.schemas.values() if s.name == "Instructions"
        )
        Instructions = _build_instructions_decoder(instr_schema.data)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(
            topics=["/instruction"]
        ):
            msg = Instructions()
            msg.ParseFromString(message.data)
            ts = None
            if msg.HasField("timestamp"):
                ts = {
                    "seconds": int(msg.timestamp.seconds),
                    "nanos": int(msg.timestamp.nanos),
                }
            return {
                "log_time_ns": int(message.log_time),
                "publish_time_ns": int(message.publish_time),
                "timestamp": ts,
                "data": msg.data,
            }
    raise RuntimeError(f"no /instruction message in {mcap_path}")


def _build_annotation_decoder(schema_data: bytes):
    fds = FileDescriptorSet()
    fds.ParseFromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    for fproto in fds.file:
        try:
            pool.Add(fproto)
        except TypeError:
            pass
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("Annotation")
    )


def _proto_ts_to_ns(ts) -> int:
    return int(ts.seconds) * 1_000_000_000 + int(ts.nanos)


def extract_subtask_events(mcap_path: Path) -> tuple[bool, list[dict]]:
    """Return (has_topic, events). Each event = {ts_ns, log_time_ns, label}."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        topic_ch = None
        for ch in summary.channels.values():
            if ch.topic == SUBTASK_TOPIC:
                topic_ch = ch
                break
        if topic_ch is None:
            return False, []
        Annotation = _build_annotation_decoder(
            summary.schemas[topic_ch.schema_id].data
        )

    events = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _s, _c, msg in reader.iter_messages(topics=[SUBTASK_TOPIC]):
            decoded = Annotation()
            decoded.ParseFromString(msg.data)
            ts_ns = (_proto_ts_to_ns(decoded.timestamp)
                     if decoded.HasField("timestamp") else int(msg.log_time))
            events.append({
                "log_time_ns": int(msg.log_time),
                "ts_ns": ts_ns,
                "label": decoded.data,
            })
    events.sort(key=lambda e: e["ts_ns"])
    return True, events


def segments_from_events(events: list[dict], episode_duration_ns: int,
                         fps: int = DEFAULT_FPS) -> list[dict]:
    """Convert point-in-time start events into closed segments with
    [start_ns, end_ns, start_frame, end_frame]."""
    segments = []
    for i, e in enumerate(events):
        start_ns = e["ts_ns"]
        end_ns = (events[i + 1]["ts_ns"]
                  if i + 1 < len(events) else episode_duration_ns)
        segments.append({
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "start_frame": int(round(start_ns / 1e9 * fps)),
            "end_frame": int(round(end_ns / 1e9 * fps)),
            "label": e["label"],
        })
    return segments


def materialize_episode(entry: dict, bundle_root: Path, *,
                        annotation_profile: str, keep_mcap: bool,
                        skip_existing: bool) -> dict:
    ep_id = entry["episode_id"]
    out_dir = bundle_root / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_info_path = out_dir / "bundle_info.json"
    if skip_existing and bundle_info_path.exists():
        with open(bundle_info_path) as f:
            existing = json.load(f)
        if existing.get("status") == "complete":
            return {"episode_id": ep_id, "status": "skipped"}

    src = entry["source_data_s3"].rstrip("/")

    # 1. Required dataset files
    for fname in _DATASET_FILES_REQUIRED:
        rc = _aws_cp(f"{src}/{fname}", out_dir / fname)
        if rc != 0:
            raise RuntimeError(f"missing required {fname} for {ep_id}")

    # 2. Best-effort videos / frame_mappings (combined is required, per-cam optional)
    for fname in _DATASET_FILES_OPTIONAL_VIDEOS + _DATASET_FILES_OPTIONAL_FRAMEMAPS:
        s3_uri = f"{src}/{fname}"
        rc = _aws_cp(s3_uri, out_dir / fname)
        if rc != 0:
            # Quietly skip — not all episodes have all per-cam mp4s.
            (out_dir / fname).unlink(missing_ok=True)

    # 3. Annotation mcap
    mcap_path = out_dir / "output.mcap"
    rc = _aws_cp(entry["annotation_s3"], mcap_path, profile=annotation_profile)
    if rc != 0:
        raise RuntimeError(f"failed to download mcap for {ep_id}")

    # 4. Extract instruction → labels.json
    instruction = extract_instruction(mcap_path)
    # Derive "source" from the annotation S3 URI, e.g.
    # s3://xdof-bair-abc/data/deliveries/abc_pilot_pretraining_20260424/...
    # -> "xdof-bair-abc/abc_pilot_pretraining_20260424"
    ann_parts = entry["annotation_s3"].replace("s3://", "").split("/")
    src_tag = (f"{ann_parts[0]}/{ann_parts[3]}"
               if len(ann_parts) >= 4 else "unknown")
    labels = {
        "version": "1.0",
        "source": src_tag,
        "instruction": instruction["data"],
        "raw": instruction,
    }

    # 5. Extract /subtask-raw-annotation segments while we still have the mcap.
    has_topic, events = extract_subtask_events(mcap_path)
    n_rows = np.fromfile(
        out_dir / "states_actions.bin", dtype=np.float64
    ).size // (14 + 14)
    ep_duration_ns = int(n_rows / DEFAULT_FPS * 1e9)
    segments = segments_from_events(events, ep_duration_ns, fps=DEFAULT_FPS)
    labels["subtasks"] = {
        "topic": SUBTASK_TOPIC,
        "schema": "Annotation (proto: timestamp + data:string)",
        "n_events": len(events),
        "fps_for_frame_conversion": DEFAULT_FPS,
        "episode_frames": int(n_rows),
        "segments": segments,
    }

    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    if not keep_mcap:
        mcap_path.unlink()

    info = {
        "episode_id": ep_id,
        "task_folder": entry["task_folder"],
        "task_name": entry.get("task_name"),
        "source_dataset": entry["source_dataset"],
        "source_data_s3": entry["source_data_s3"],
        "annotation_s3": entry["annotation_s3"],
        "instruction": instruction["data"],
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "status": "complete",
    }
    with open(bundle_info_path, "w") as f:
        json.dump(info, f, indent=2)
    return {"episode_id": ep_id, "status": "complete",
            "instruction": instruction["data"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="pilot_manifest.json")
    p.add_argument("--bundle-root", default="pilot_data")
    p.add_argument("--annotation-profile", default="xdof")
    p.add_argument("--keep-mcap", action="store_true",
                   help="Retain the (large) output.mcap after extracting labels.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episodes whose bundle_info.json reports complete.")
    p.add_argument("--episode", default=None,
                   help="Materialize only this single episode_id.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of episodes processed (for smoke testing).")
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    bundle_root = Path(args.bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    entries = manifest["episodes"]
    if args.episode:
        entries = [e for e in entries if e["episode_id"] == args.episode]
        if not entries:
            raise SystemExit(f"episode {args.episode} not in manifest")
    if args.limit:
        entries = entries[: args.limit]

    print(f"Materializing {len(entries)} episode(s) -> {bundle_root}/")
    summaries = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['episode_id']} "
              f"({entry['source_dataset']})", flush=True)
        try:
            res = materialize_episode(
                entry, bundle_root,
                annotation_profile=args.annotation_profile,
                keep_mcap=args.keep_mcap, skip_existing=args.skip_existing,
            )
            summaries.append(res)
            print(f"   -> {res['status']}: {res.get('instruction', '')}")
        except Exception as e:
            print(f"   FAILED: {e}")
            summaries.append({"episode_id": entry["episode_id"],
                              "status": "failed", "error": str(e)})

    summary_path = bundle_root / "_materialize_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    n_ok = sum(1 for s in summaries if s["status"] == "complete")
    n_skip = sum(1 for s in summaries if s["status"] == "skipped")
    n_fail = sum(1 for s in summaries if s["status"] == "failed")
    print(f"\nDone: {n_ok} complete, {n_skip} skipped, {n_fail} failed. "
          f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
