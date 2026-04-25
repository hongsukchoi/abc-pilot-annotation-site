"""Second-pass extraction for the updated abc_pilot_sample_annotation_20260416
delivery, which now includes a `/subtask-raw-annotation` channel on each
output.mcap.

For each episode in pilot_manifest.json:
  1. Re-download the mcap from S3.
  2. Confirm presence of /subtask-raw-annotation and count its messages.
  3. Decode each Annotation message to (timestamp_ns, text).
  4. Update labels.json in the materialized bundle with:
       labels["subtasks"] = [{start_time_ns, end_time_ns, start_frame,
                              end_frame, label}, ...]
     (end_time taken as the next subtask's start_time, or episode end.)
  5. Delete the mcap unless --keep-mcap.
  6. Print a per-episode summary and a final count.

Usage:
    uv run python deploy/scripts/extract_subtask_annotations.py \
        --manifest pilot_manifest.json \
        --bundle-root pilot_data \
        --annotation-profile xdof
"""

import argparse
import json
import subprocess
from pathlib import Path

from mcap.reader import make_reader
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf import descriptor_pool, message_factory

TOPIC = "/subtask-raw-annotation"


def _build_annotation_decoder(schema_data: bytes):
    fds = FileDescriptorSet()
    fds.ParseFromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    for fproto in fds.file:
        try:
            pool.Add(fproto)
        except TypeError:
            pass
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("Annotation"))


def _proto_ts_to_ns(ts) -> int:
    return int(ts.seconds) * 1_000_000_000 + int(ts.nanos)


def extract_subtasks(mcap_path: Path) -> dict:
    """Return dict with summary info + list of decoded subtask events."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        # Does the topic exist?
        topic_ch = None
        for ch in summary.channels.values():
            if ch.topic == TOPIC:
                topic_ch = ch
                break
        topic_count = (
            summary.statistics.channel_message_counts.get(topic_ch.id, 0)
            if topic_ch is not None
            else 0
        )
        # Build annotation decoder from its schema
        Annotation = None
        if topic_ch is not None:
            Annotation = _build_annotation_decoder(
                summary.schemas[topic_ch.schema_id].data
            )

    events = []
    if topic_ch is not None:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for _s, _c, msg in reader.iter_messages(topics=[TOPIC]):
                decoded = Annotation()
                decoded.ParseFromString(msg.data)
                ts_ns = (
                    _proto_ts_to_ns(decoded.timestamp)
                    if decoded.HasField("timestamp")
                    else int(msg.log_time)
                )
                events.append({
                    "log_time_ns": int(msg.log_time),
                    "ts_ns": ts_ns,
                    "label": decoded.data,
                })
    events.sort(key=lambda e: e["ts_ns"])

    return {
        "has_topic": topic_ch is not None,
        "n_messages": topic_count,
        "events": events,
    }


def segments_from_events(events: list[dict], episode_duration_ns: int,
                         fps: int = 30) -> list[dict]:
    """Convert point-in-time start events into closed segments with
    [start_ns, end_ns, start_frame, end_frame]. end is the next event's
    start (or episode_duration_ns for the last)."""
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="pilot_manifest.json")
    p.add_argument("--bundle-root", default="pilot_data")
    p.add_argument("--annotation-profile", default="xdof")
    p.add_argument("--keep-mcap", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    bundle_root = Path(args.bundle_root)
    results = []
    for i, entry in enumerate(manifest["episodes"], 1):
        ep_id = entry["episode_id"]
        bundle = bundle_root / ep_id
        if not bundle.exists():
            print(f"[{i}/{len(manifest['episodes'])}] {ep_id}: no bundle, skipping")
            continue

        mcap_local = bundle / "output.mcap"
        if not mcap_local.exists():
            print(f"[{i}/{len(manifest['episodes'])}] {ep_id}: downloading mcap")
            rc = subprocess.call([
                "aws", "s3", "cp", "--quiet",
                "--profile", args.annotation_profile,
                entry["annotation_s3"], str(mcap_local),
            ])
            if rc != 0:
                print(f"   FAILED to download mcap")
                results.append({"episode_id": ep_id, "status": "download_failed"})
                continue
        else:
            print(f"[{i}/{len(manifest['episodes'])}] {ep_id}: using cached mcap")

        info = extract_subtasks(mcap_local)

        # Determine episode frame count from states_actions.bin
        import numpy as np
        n_rows = np.fromfile(
            bundle / "states_actions.bin", dtype=np.float64
        ).size // (14 + 14)
        ep_duration_ns = int(n_rows / args.fps * 1e9)
        segments = segments_from_events(info["events"], ep_duration_ns,
                                        fps=args.fps)

        # Merge into labels.json
        labels_path = bundle / "labels.json"
        with open(labels_path) as f:
            labels = json.load(f)
        labels["subtasks"] = {
            "topic": TOPIC,
            "schema": "Annotation (proto: timestamp + data:string)",
            "n_events": len(info["events"]),
            "fps_for_frame_conversion": args.fps,
            "episode_frames": n_rows,
            "segments": segments,
        }
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        if not args.keep_mcap:
            mcap_local.unlink()

        status = ("ok" if info["has_topic"] and info["n_messages"] > 0
                  else "NO_SUBTASK_TOPIC")
        print(f"   {status}: n_messages={info['n_messages']}, "
              f"segments={len(segments)}")
        if segments:
            print(f"   first: [{segments[0]['start_frame']}..{segments[0]['end_frame']}] "
                  f"{segments[0]['label'][:80]}")
            print(f"   last:  [{segments[-1]['start_frame']}..{segments[-1]['end_frame']}] "
                  f"{segments[-1]['label'][:80]}")

        results.append({
            "episode_id": ep_id,
            "task_folder": entry["task_folder"],
            "has_topic": info["has_topic"],
            "n_messages": info["n_messages"],
            "n_segments": len(segments),
        })

    out_summary = bundle_root / "_subtask_summary.json"
    with open(out_summary, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if r.get("has_topic") and r.get("n_messages", 0) > 0)
    n_empty = len(results) - n_ok
    print(f"\n=== Summary ===")
    print(f"episodes with /subtask-raw-annotation: {n_ok}/{len(results)}")
    print(f"episodes missing subtask topic / empty: {n_empty}")
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
