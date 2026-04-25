"""Render a diagnostic mp4 for a materialized abc-pilot episode bundle.

Mirrors the layout of deploy/scripts/teleop_recording_to_video.py
(cameras stacked left, joint trajectories on the right with a rolling time
window) but reads from the dataset bundle format produced by
materialize_pilot_bundles.py:

    <bundle>/<episode_id>/
        top_camera-images-rgb.mp4
        left_camera-images-rgb.mp4
        right_camera-images-rgb.mp4
        states_actions.bin     [N, 28] float64 = [state(14) | action(14)]
        labels.json            { "instruction": "..." }
        episode_metadata.json

The label/instruction is rendered as a persistent banner directly under the
camera column, near the images.

Usage:
    uv run python deploy/scripts/dataset_episode_to_video.py \
        --bundle pilot_data/episode_019d6863-... \
        --output pilot_renders/episode_019d6863-....mp4
    # Render every bundle under a root:
    uv run python deploy/scripts/dataset_episode_to_video.py \
        --bundle-root pilot_data --output-root pilot_renders
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


CAM_ORDER = ("top", "left", "right")  # YAM source camera display order
DOF_PER_ARM = 7   # 6 joints + 1 gripper for YAM
N_STATES = 14
N_ACTIONS = 14


def load_states_actions(bin_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (state, action) shaped (N, 14) and (N, 14) float64."""
    raw = np.fromfile(bin_path, dtype=np.float64)
    expected = N_STATES + N_ACTIONS
    if raw.size % expected != 0:
        raise ValueError(
            f"{bin_path} has {raw.size} float64 elements, "
            f"not divisible by {expected}"
        )
    arr = raw.reshape(-1, expected)
    return arr[:, :N_STATES].copy(), arr[:, N_STATES:].copy()


@dataclass
class CameraStream:
    name: str
    cap: cv2.VideoCapture
    n_frames: int
    height: int
    width: int

    def read_frame(self, idx: int) -> np.ndarray | None:
        # cv2 sequential reads are fastest; we always advance one at a time.
        # Caller is expected to call in order; if not, seek.
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame


def open_camera_streams(bundle: Path) -> list[CameraStream]:
    streams = []
    for cam in CAM_ORDER:
        path = bundle / f"{cam}_camera-images-rgb.mp4"
        if not path.exists():
            print(f"  WARNING: missing camera mp4 {path.name}")
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  WARNING: could not open {path.name}")
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        streams.append(CameraStream(cam, cap, n, h, w))
    if not streams:
        raise RuntimeError(f"no camera streams found in {bundle}")
    return streams


@dataclass
class SubplotGeom:
    px_left: int
    px_right: int
    px_top: int
    px_bottom: int
    y_min: float
    y_max: float


def render_episode(
    bundle: Path,
    output_path: Path,
    *,
    fps: int = 30,
    fig_w: int = 1920,
    fig_h: int = 1080,
    layout: str = "simple",
) -> None:
    """Render an episode to mp4.

    layout:
        "plots"  — cameras on left (~35% width), state/action plots on right.
        "simple" — cameras side-by-side filling width, large INSTRUCTION/SUBTASK
                   banner below. No state/action plots.
    """
    if layout not in ("plots", "simple"):
        raise ValueError(f"unknown layout {layout!r}")
    state, action = load_states_actions(bundle / "states_actions.bin")
    n_frames = state.shape[0]

    with open(bundle / "labels.json") as f:
        labels = json.load(f)
    instruction = labels.get("instruction", "").strip() or "(no instruction)"

    subtask_segments = []
    if isinstance(labels.get("subtasks"), dict):
        for s in labels["subtasks"].get("segments", []):
            subtask_segments.append({
                "start_frame": int(s["start_frame"]),
                "end_frame": int(s["end_frame"]),
                "label": str(s["label"]),
            })
    subtask_segments.sort(key=lambda s: s["start_frame"])
    subtask_starts = np.array(
        [s["start_frame"] for s in subtask_segments], dtype=np.int64
    )

    metadata_path = bundle / "episode_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    streams = open_camera_streams(bundle)
    print(f"  cameras: {[s.name for s in streams]}, frames per cam: "
          f"{[s.n_frames for s in streams]}, states_actions rows: {n_frames}")

    # Use the minimum of (states_actions rows, every camera's frame count) as
    # the playback length. They should match for canonical dataset episodes.
    n_play = min(n_frames, *[s.n_frames for s in streams])

    cam0 = streams[0]
    cam_aspect = cam0.width / cam0.height

    # ---- Layout ----
    cam_margin = 12
    cam_gap = 8
    title_h = 32
    n_cams = len(streams)

    cam_regions: dict[str, tuple[int, int, int, int]] = {}
    cam_label_pos: dict[str, tuple[int, int]] = {}

    if layout == "plots":
        cam_col_w = int(fig_w * 0.35)
        label_banner_h = 168
        avail_h = (fig_h - title_h - 2 * cam_margin
                   - (n_cams - 1) * cam_gap - label_banner_h)
        each_h = avail_h // n_cams
        each_w = min(int(each_h * cam_aspect), cam_col_w - 2 * cam_margin)
        each_h = int(each_w / cam_aspect)

        y = title_h + cam_margin
        for s in streams:
            cam_regions[s.name] = (y, y + each_h, cam_margin,
                                   cam_margin + each_w)
            cam_label_pos[s.name] = (cam_margin + 4, y + 18)
            y += each_h + cam_gap
        label_banner_y0 = y
        label_banner_x0 = cam_margin
        label_banner_x1 = cam_margin + (cam_col_w - 2 * cam_margin)
    else:  # "simple"
        # Cameras side-by-side filling width; big banner beneath.
        label_banner_h = int(fig_h * 0.42)
        avail_h = fig_h - title_h - 2 * cam_margin - label_banner_h
        avail_w = fig_w - 2 * cam_margin - (n_cams - 1) * cam_gap
        each_w = avail_w // n_cams
        each_h = int(each_w / cam_aspect)
        if each_h > avail_h:
            each_h = avail_h
            each_w = int(each_h * cam_aspect)
        row_w = each_w * n_cams + (n_cams - 1) * cam_gap
        x0 = (fig_w - row_w) // 2
        y_top = title_h + cam_margin + (avail_h - each_h) // 2
        x = x0
        for s in streams:
            cam_regions[s.name] = (y_top, y_top + each_h, x, x + each_w)
            cam_label_pos[s.name] = (x + 8, y_top + 26)
            x += each_w + cam_gap
        label_banner_y0 = title_h + cam_margin + avail_h + cam_margin
        label_banner_x0 = cam_margin
        label_banner_x1 = fig_w - cam_margin

    BG_HEX = "#FDFAF5"
    BG_RGB = (253, 250, 245)
    BG_BGR = (BG_RGB[2], BG_RGB[1], BG_RGB[0])

    plot_rows: list[tuple] = []
    dof_geoms: list[SubplotGeom] = []

    if layout == "plots":
        # Per arm: 6 joints + gripper. Left first, then right.
        for arm_idx, arm_name in enumerate(("left", "right")):
            base = arm_idx * DOF_PER_ARM
            for j in range(6):
                plot_rows.append(
                    (arm_name, base + j, f"{arm_name[0].upper()}-j{j+1}", False)
                )
            plot_rows.append(
                (arm_name, base + 6, f"{arm_name[0].upper()}-grip", True)
            )
        n_plots = len(plot_rows)
        y_lims = []
        for _arm, didx, _lbl, _is_grip in plot_rows:
            vals = np.concatenate([state[:n_play, didx], action[:n_play, didx]])
            margin = (vals.max() - vals.min()) * 0.15 + 1e-6
            y_lims.append((vals.min() - margin, vals.max() + margin))

        plot_left_frac = (int(fig_w * 0.35) + 4) / fig_w
        fig = plt.figure(figsize=(fig_w / 100, fig_h / 100), dpi=100)
        fig.patch.set_facecolor(BG_HEX)
        gs = fig.add_gridspec(
            n_plots, 1, hspace=0.12, top=0.96, bottom=0.04,
            left=plot_left_frac, right=0.94,
        )
        dof_axes = []
        for i, (_arm, _didx, label, _is_grip) in enumerate(plot_rows):
            ax = fig.add_subplot(gs[i, 0])
            ax.set_facecolor(BG_HEX)
            for spine in ("top", "left", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.8)
            ax.spines["bottom"].set_color("#333333")
            ax.tick_params(axis="x", length=0)
            ax.set_xticks([])
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=22,
                          va="center", fontfamily="sans-serif",
                          fontweight="bold")
            ax.tick_params(axis="y", labelsize=6, length=3, width=0.5,
                           colors="#555555")
            ax.set_ylim(*y_lims[i])
            dof_axes.append(ax)

        legend_elements = [
            Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="State"),
            Line2D([0], [0], color="#ff7f0e", linewidth=1.5, linestyle="--",
                   label="Action"),
        ]
        leg = dof_axes[0].legend(handles=legend_elements, loc="upper right",
                                  fontsize=7, ncol=2)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor("#cccccc")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        def extract_geom(ax) -> SubplotGeom:
            renderer = fig.canvas.get_renderer()
            bbox = ax.get_window_extent(renderer)
            fig_h_px = fig.get_size_inches()[1] * fig.dpi
            return SubplotGeom(
                px_left=int(round(bbox.x0)),
                px_right=int(round(bbox.x1)),
                px_top=int(round(fig_h_px - bbox.y1)),
                px_bottom=int(round(fig_h_px - bbox.y0)),
                y_min=ax.get_ylim()[0],
                y_max=ax.get_ylim()[1],
            )

        dof_geoms = [extract_geom(ax) for ax in dof_axes]
        static_bg = cv2.cvtColor(
            np.frombuffer(fig.canvas.buffer_rgba(),
                          dtype=np.uint8).reshape(h, w, 4),
            cv2.COLOR_RGBA2BGR,
        ).copy()
        plt.close(fig)
    else:  # "simple"
        w, h = fig_w, fig_h
        static_bg = np.full((h, w, 3), BG_BGR, dtype=np.uint8)

    # ---- Helpers ----
    def data_to_pixel(t, v, geom: SubplotGeom, t_left, t_right):
        t_span = max(t_right - t_left, 1e-9)
        y_span = max(geom.y_max - geom.y_min, 1e-9)
        px_x = geom.px_left + (t - t_left) / t_span * (geom.px_right - geom.px_left)
        px_y = geom.px_top + (geom.y_max - v) / y_span * (geom.px_bottom - geom.px_top)
        pts = np.stack([px_x, px_y], axis=-1).astype(np.int32)
        return pts.reshape(-1, 1, 2)

    def clip_pts(pts, geom: SubplotGeom):
        p = pts.reshape(-1, 2).copy()
        p[:, 0] = np.clip(p[:, 0], geom.px_left, geom.px_right)
        p[:, 1] = np.clip(p[:, 1], geom.px_top, geom.px_bottom)
        return p.reshape(-1, 1, 2).astype(np.int32)

    def data_x_to_pixel(t, geom: SubplotGeom, t_left, t_right) -> int:
        t_span = max(t_right - t_left, 1e-9)
        return int(round(geom.px_left + (t - t_left) / t_span
                         * (geom.px_right - geom.px_left)))

    color_state = (180, 119, 31)     # #1f77b4 in BGR
    color_action = (14, 127, 255)    # #ff7f0e in BGR
    color_now = (0, 0, 200)
    color_text = (0x33, 0x33, 0x33)

    # ---- Video writer ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_ffmpeg = shutil.which("ffmpeg") is not None
    ffmpeg_proc = None
    writer = None
    if use_ffmpeg:
        ffmpeg_proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-preset", "fast",
             "-movflags", "+faststart", "-an", str(output_path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output_path}")

    print(f"  rendering {n_play} frames -> {output_path}")

    # Per-frame relative time (state and camera are aligned 1:1 in dataset format).
    cam_times = np.arange(n_play) / float(fps)
    total_dur = cam_times[-1] - cam_times[0] if n_play > 1 else 0.0
    window_past_s = 2.0
    window_future_s = 0.5

    def tick_step(window_width: float) -> float:
        target_n = 9
        raw = window_width / target_n
        mag = 10 ** np.floor(np.log10(raw))
        res = raw / mag
        if res < 1.5:
            nice = 1.0
        elif res < 3.5:
            nice = 2.0
        elif res < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * mag

    tstep = tick_step(window_past_s + window_future_s)

    # Pre-wrap label banner text once. Font sizes differ per layout.
    banner_text = instruction
    banner_font = cv2.FONT_HERSHEY_DUPLEX
    if layout == "simple":
        banner_scale = 1.3
        banner_thickness = 2
        header_scale = 0.8
        header_thickness = 2
        cam_badge_scale = 0.7
        cam_badge_thickness = 2
        title_scale = 0.7
        title_thickness = 2
        header_line_gap = 36
        body_line_gap = 44
    else:
        banner_scale = 0.7
        banner_thickness = 1
        header_scale = 0.42
        header_thickness = 1
        cam_badge_scale = 0.40
        cam_badge_thickness = 1
        title_scale = 0.50
        title_thickness = 1
        header_line_gap = 18
        body_line_gap = 22

    def wrap_text(text: str, max_px: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines, cur = [], words[0]
        for word in words[1:]:
            trial = cur + " " + word
            (tw, _), _ = cv2.getTextSize(trial, banner_font, banner_scale,
                                         banner_thickness)
            if tw <= max_px:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        return lines

    banner_max_w = (label_banner_x1 - label_banner_x0) - 24
    banner_lines = wrap_text(banner_text, banner_max_w)

    # ---- Per-frame loop ----
    last_progress = -1
    for fi in range(n_play):
        t_now = cam_times[fi]
        frame = static_bg.copy()
        t_left = t_now - window_past_s
        t_right = t_now + window_future_s

        # Camera tiles
        for s in streams:
            ry0, ry1, rx0, rx1 = cam_regions[s.name]
            tw = rx1 - rx0
            th = ry1 - ry0
            cam_frame = s.read_frame(fi)
            if cam_frame is None:
                continue
            resized = cv2.resize(cam_frame, (tw, th))
            frame[ry0:ry1, rx0:rx1] = resized

            # Camera badge
            lx, ly = cam_label_pos[s.name]
            (tw_l, th_l), _ = cv2.getTextSize(
                s.name.upper(), cv2.FONT_HERSHEY_DUPLEX,
                cam_badge_scale, cam_badge_thickness
            )
            pad = 5
            cv2.rectangle(frame, (lx - pad, ly - th_l - pad),
                          (lx + tw_l + pad, ly + pad), (255, 255, 255), -1)
            cv2.rectangle(frame, (lx - pad, ly - th_l - pad),
                          (lx + tw_l + pad, ly + pad), (200, 200, 200), 1)
            cv2.putText(frame, s.name.upper(), (lx, ly),
                        cv2.FONT_HERSHEY_DUPLEX, cam_badge_scale, (0, 0, 0),
                        cam_badge_thickness, cv2.LINE_AA)

        # Title
        title_str = (f"{bundle.name}  |  t = {t_now:.2f}s / {total_dur:.1f}s"
                     f"  |  {metadata.get('task_name', '')}")
        cv2.putText(frame, title_str, (cam_margin, 22),
                    cv2.FONT_HERSHEY_DUPLEX, title_scale, (0, 0, 0),
                    title_thickness, cv2.LINE_AA)

        # Label banner: INSTRUCTION (static top) + SUBTASK (per-frame bottom).
        banner_x0 = label_banner_x0
        banner_x1 = label_banner_x1

        # -- INSTRUCTION sub-banner --
        instr_y0 = label_banner_y0
        instr_y1 = instr_y0 + label_banner_h // 2
        cv2.rectangle(frame, (banner_x0, instr_y0),
                      (banner_x1, instr_y1), (245, 235, 215), -1)
        cv2.rectangle(frame, (banner_x0, instr_y0),
                      (banner_x1, instr_y1), (140, 110, 60), 2)
        header_str = "INSTRUCTION"
        (_hw, hh), _ = cv2.getTextSize(header_str, cv2.FONT_HERSHEY_DUPLEX,
                                        header_scale, header_thickness)
        cv2.putText(frame, header_str, (banner_x0 + 12, instr_y0 + hh + 8),
                    cv2.FONT_HERSHEY_DUPLEX, header_scale, (90, 60, 20),
                    header_thickness, cv2.LINE_AA)
        line_y = instr_y0 + hh + 8 + header_line_gap
        for line in banner_lines:
            (_lw, lh), _ = cv2.getTextSize(line, banner_font, banner_scale,
                                            banner_thickness)
            if line_y + lh > instr_y1 - 6:
                break
            cv2.putText(frame, line, (banner_x0 + 12, line_y), banner_font,
                        banner_scale, (15, 15, 15), banner_thickness,
                        cv2.LINE_AA)
            line_y += lh + 8

        # -- SUBTASK sub-banner (per-frame) --
        sub_y0 = instr_y1 + 4
        sub_y1 = min(sub_y0 + label_banner_h // 2 - 4, fig_h - cam_margin)

        if subtask_segments:
            idx = int(np.searchsorted(subtask_starts, fi, side="right")) - 1
            idx = max(0, idx)
            seg = subtask_segments[idx]
            is_special = seg["label"].lower().startswith("special=")
            in_range = seg["start_frame"] <= fi <= seg["end_frame"]
            if is_special:
                bg_color = (180, 220, 255)
                border_color = (20, 100, 200)
                header_color = (20, 60, 150)
            else:
                bg_color = (230, 245, 230)
                border_color = (60, 140, 60)
                header_color = (30, 90, 30)
        else:
            idx = -1
            seg = None
            in_range = False
            bg_color = (235, 235, 235)
            border_color = (160, 160, 160)
            header_color = (80, 80, 80)

        cv2.rectangle(frame, (banner_x0, sub_y0),
                      (banner_x1, sub_y1), bg_color, -1)
        cv2.rectangle(frame, (banner_x0, sub_y0),
                      (banner_x1, sub_y1), border_color, 2)

        if seg is None:
            header_str = "SUBTASK (none)"
            (_hw, shh), _ = cv2.getTextSize(
                header_str, cv2.FONT_HERSHEY_DUPLEX, header_scale,
                header_thickness
            )
            cv2.putText(frame, header_str, (banner_x0 + 12, sub_y0 + shh + 8),
                        cv2.FONT_HERSHEY_DUPLEX, header_scale, header_color,
                        header_thickness, cv2.LINE_AA)
        else:
            n_total = len(subtask_segments)
            progress = (f"SUBTASK {idx + 1}/{n_total}"
                        f"  [{seg['start_frame']}..{seg['end_frame']}]")
            if not in_range:
                progress += "  (past end)"
            (_hw, shh), _ = cv2.getTextSize(
                progress, cv2.FONT_HERSHEY_DUPLEX, header_scale,
                header_thickness
            )
            cv2.putText(frame, progress, (banner_x0 + 12, sub_y0 + shh + 8),
                        cv2.FONT_HERSHEY_DUPLEX, header_scale, header_color,
                        header_thickness, cv2.LINE_AA)
            sub_lines = wrap_text(seg["label"], banner_max_w)
            line_y = sub_y0 + shh + 8 + header_line_gap
            for line in sub_lines:
                (_lw, lh), _ = cv2.getTextSize(
                    line, banner_font, banner_scale, banner_thickness
                )
                if line_y + lh > sub_y1 - 6:
                    break
                cv2.putText(frame, line, (banner_x0 + 12, line_y),
                            banner_font, banner_scale, (15, 15, 15),
                            banner_thickness, cv2.LINE_AA)
                line_y += lh + 8

        if layout == "plots":
            # Joint trajectories
            ts = cam_times
            for i, (_arm, didx, _lbl, is_grip) in enumerate(plot_rows):
                geom = dof_geoms[i]
                mask = (ts >= t_left) & (ts <= t_right)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue
                lo = max(indices[0] - 1, 0)
                hi = min(indices[-1] + 2, len(ts))
                ts_vis = ts[lo:hi]

                v_state = state[lo:hi, didx]
                pts_s = clip_pts(
                    data_to_pixel(ts_vis, v_state, geom, t_left, t_right), geom
                )
                cv2.polylines(frame, [pts_s], False, color_state, 2,
                              cv2.LINE_AA)

                v_action = action[lo:hi, didx]
                pts_a = clip_pts(
                    data_to_pixel(ts_vis, v_action, geom, t_left, t_right),
                    geom,
                )
                cv2.polylines(frame, [pts_a], False, color_action, 1,
                              cv2.LINE_AA)

                px_now = data_x_to_pixel(t_now, geom, t_left, t_right)
                if geom.px_left <= px_now <= geom.px_right:
                    cv2.line(frame, (px_now, geom.px_top),
                             (px_now, geom.px_bottom), color_now, 1,
                             cv2.LINE_AA)

            # X-axis ticks on bottom subplot
            bottom_geom = dof_geoms[-1]
            t_tick = np.ceil(t_left / tstep) * tstep
            spine_color = (0x33, 0x33, 0x33)
            while t_tick <= t_right:
                px_x = data_x_to_pixel(t_tick, bottom_geom, t_left, t_right)
                if bottom_geom.px_left <= px_x <= bottom_geom.px_right:
                    lstr = f"{t_tick:.1f}"
                    cv2.line(frame, (px_x, bottom_geom.px_bottom - 2),
                             (px_x, bottom_geom.px_bottom + 2), spine_color, 1)
                    (tw_t, th_t), _ = cv2.getTextSize(
                        lstr, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1
                    )
                    cv2.putText(frame, lstr,
                                (px_x - tw_t // 2,
                                 bottom_geom.px_bottom + 4 + th_t),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, color_text, 1,
                                cv2.LINE_AA)
                t_tick += tstep

        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(frame.tobytes())
        else:
            writer.write(frame)

        pct = int(100 * (fi + 1) / n_play)
        if pct != last_progress and (pct % 10 == 0):
            print(f"    {pct}% ({fi+1}/{n_play})", flush=True)
            last_progress = pct

    for s in streams:
        s.cap.release()
    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    if writer is not None:
        writer.release()
    print(f"  done -> {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=None,
                   help="Path to a single materialized episode bundle.")
    p.add_argument("--bundle-root", default=None,
                   help="Render every bundle subdir under this root.")
    p.add_argument("--output", default=None,
                   help="Output mp4 path (single bundle mode).")
    p.add_argument("--output-root", default="pilot_renders",
                   help="Output directory (bundle-root mode).")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--layout", choices=("simple", "plots"), default="simple",
                   help="simple = cameras + label banner only (default); "
                        "plots = include 14-DoF state/action subplots.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip bundles whose output mp4 already exists.")
    args = p.parse_args()

    if args.bundle is None and args.bundle_root is None:
        sys.exit("must supply --bundle or --bundle-root")

    if args.bundle:
        bundle = Path(args.bundle)
        out = Path(args.output) if args.output else (
            Path(args.output_root) / f"{bundle.name}.mp4"
        )
        render_episode(bundle, out, fps=args.fps, layout=args.layout)
        return

    root = Path(args.bundle_root)
    out_root = Path(args.output_root)
    bundles = sorted(p for p in root.iterdir()
                     if p.is_dir() and (p / "states_actions.bin").exists())
    print(f"Found {len(bundles)} bundles under {root}")
    for i, b in enumerate(bundles, 1):
        out = out_root / f"{b.name}.mp4"
        if args.skip_existing and out.exists() and out.stat().st_size > 0:
            print(f"[{i}/{len(bundles)}] {b.name} -> SKIP (exists)")
            continue
        print(f"[{i}/{len(bundles)}] {b.name}")
        try:
            render_episode(b, out, fps=args.fps, layout=args.layout)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
