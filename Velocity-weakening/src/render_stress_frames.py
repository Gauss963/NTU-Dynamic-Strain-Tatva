from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_CTX: dict[str, object] = {}


def von_mises_2d(stress: np.ndarray) -> np.ndarray:
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    sxy = stress[..., 0, 1]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def von_mises_3d(stress: np.ndarray) -> np.ndarray:
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    szz = stress[..., 2, 2]
    sxy = stress[..., 0, 1]
    syz = stress[..., 1, 2]
    szx = stress[..., 2, 0]
    return np.sqrt(
        np.maximum(
            0.5
            * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (sxy * sxy + syz * syz + szx * szx)
            ),
            0.0,
        )
    )


def stress_scalar(stress: np.ndarray, mode: str) -> np.ndarray:
    dim = int(stress.shape[-1])
    if mode == "von_mises":
        return von_mises_2d(stress) if dim == 2 else von_mises_3d(stress)
    if mode == "sigma_xy":
        return stress[..., 0, 1]
    raise ValueError(f"Unsupported stress mode: {mode}")


def compute_global_ranges(
    h5_path: Path,
    batch_size: int = 64,
    *,
    stress_mode: str,
    stress_percentile: float = 99.5,
    sample_stride: int = 32,
) -> tuple[float, float, float]:
    stress_abs_max = 0.0
    disp_max = 0.0
    stress_samples: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as h5:
        n_frames = h5["history"].shape[0]
        moving_parent = (
            np.asarray(h5["moving/plot_parent_elements"], dtype=np.int32)
            if "plot_parent_elements" in h5["moving"]
            else None
        )
        stationary_parent = (
            np.asarray(h5["stationary/plot_parent_elements"], dtype=np.int32)
            if "plot_parent_elements" in h5["stationary"]
            else None
        )
        for start in range(0, n_frames, batch_size):
            stop = min(start + batch_size, n_frames)
            moving_stress = np.asarray(h5["moving/stress"][start:stop], dtype=np.float32)
            stationary_stress = np.asarray(
                h5["stationary/stress"][start:stop], dtype=np.float32
            )
            if moving_parent is not None:
                moving_stress = moving_stress[:, moving_parent]
            if stationary_parent is not None:
                stationary_stress = stationary_stress[:, stationary_parent]
            moving_disp = np.asarray(
                h5["moving/displacement"][start:stop], dtype=np.float32
            )
            stationary_disp = np.asarray(
                h5["stationary/displacement"][start:stop], dtype=np.float32
            )
            moving_scalar = stress_scalar(moving_stress, stress_mode)
            stationary_scalar = stress_scalar(stationary_stress, stress_mode)
            stress_abs_max = max(
                stress_abs_max,
                float(np.abs(moving_scalar).max(initial=0.0)),
                float(np.abs(stationary_scalar).max(initial=0.0)),
            )
            disp_max = max(
                disp_max,
                float(np.linalg.norm(moving_disp, axis=-1).max(initial=0.0)),
                float(np.linalg.norm(stationary_disp, axis=-1).max(initial=0.0)),
            )
            stress_samples.append(moving_scalar.reshape(-1)[::sample_stride])
            stress_samples.append(stationary_scalar.reshape(-1)[::sample_stride])
    sampled = np.concatenate(stress_samples) if stress_samples else np.zeros(1, dtype=np.float32)
    if stress_mode == "von_mises":
        robust_max = float(np.percentile(sampled, stress_percentile))
    else:
        robust_max = float(np.percentile(np.abs(sampled), stress_percentile))
    robust_max = max(robust_max, 1e-6)
    return stress_abs_max, robust_max, disp_max


def _init_worker(
    h5_path: str,
    output_dir: str,
    deform_scale: float,
    stress_max: float,
    stress_clip_label: str,
    dpi: int,
    figsize: tuple[float, float],
    swap_axes: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    stress_mode: str,
) -> None:
    _CTX["h5_path"] = h5_path
    _CTX["output_dir"] = output_dir
    _CTX["deform_scale"] = deform_scale
    _CTX["stress_max"] = stress_max
    _CTX["stress_clip_label"] = stress_clip_label
    _CTX["dpi"] = dpi
    _CTX["figsize"] = figsize
    _CTX["swap_axes"] = swap_axes
    _CTX["xlim"] = xlim
    _CTX["ylim"] = ylim
    _CTX["stress_mode"] = stress_mode
    _CTX["file"] = h5py.File(h5_path, "r")


def _render_frame(frame_idx: int) -> str:
    h5: h5py.File = _CTX["file"]  # type: ignore[assignment]
    deform_scale = float(_CTX["deform_scale"])
    stress_max = float(_CTX["stress_max"])
    stress_clip_label = str(_CTX["stress_clip_label"])
    dpi = int(_CTX["dpi"])
    figsize = tuple(_CTX["figsize"])  # type: ignore[arg-type]
    swap_axes = bool(_CTX["swap_axes"])
    xlim = tuple(_CTX["xlim"])  # type: ignore[arg-type]
    ylim = tuple(_CTX["ylim"])  # type: ignore[arg-type]
    stress_mode = str(_CTX["stress_mode"])
    output_dir = Path(str(_CTX["output_dir"]))

    moving_coords = np.asarray(h5["moving/coords"], dtype=np.float32)
    moving_elements = np.asarray(
        h5["moving/plot_elements"] if "plot_elements" in h5["moving"] else h5["moving/elements"],
        dtype=np.int32,
    )
    stationary_coords = np.asarray(h5["stationary/coords"], dtype=np.float32)
    stationary_elements = np.asarray(
        h5["stationary/plot_elements"]
        if "plot_elements" in h5["stationary"]
        else h5["stationary/elements"],
        dtype=np.int32,
    )
    moving_parent = (
        np.asarray(h5["moving/plot_parent_elements"], dtype=np.int32)
        if "plot_parent_elements" in h5["moving"]
        else None
    )
    stationary_parent = (
        np.asarray(h5["stationary/plot_parent_elements"], dtype=np.int32)
        if "plot_parent_elements" in h5["stationary"]
        else None
    )

    moving_disp = np.asarray(h5["moving/displacement"][frame_idx], dtype=np.float32)
    stationary_disp = np.asarray(
        h5["stationary/displacement"][frame_idx], dtype=np.float32
    )
    moving_stress = np.asarray(h5["moving/stress"][frame_idx], dtype=np.float32)
    stationary_stress = np.asarray(h5["stationary/stress"][frame_idx], dtype=np.float32)
    hist = np.asarray(h5["history"][frame_idx], dtype=np.float32)

    if moving_parent is not None:
        moving_stress = moving_stress[moving_parent]
    if stationary_parent is not None:
        stationary_stress = stationary_stress[stationary_parent]

    moving_scalar = stress_scalar(moving_stress, stress_mode)
    stationary_scalar = stress_scalar(stationary_stress, stress_mode)
    moving_xy = moving_coords + deform_scale * moving_disp
    stationary_xy = stationary_coords + deform_scale * stationary_disp
    if moving_xy.shape[1] == 3:
        moving_xy = moving_xy[:, :2]
        stationary_xy = stationary_xy[:, :2]
    if swap_axes:
        moving_xy = moving_xy[:, [1, 0]]
        stationary_xy = stationary_xy[:, [1, 0]]

    if stress_mode == "sigma_xy":
        cmap = "RdBu_r"
        vmin = -stress_max
        vmax = stress_max
        cbar_label = "sigma_xy shear stress"
        title_label = "sigma_xy shear stress"
    else:
        cmap = "magma"
        vmin = 0.0
        vmax = stress_max
        cbar_label = "von Mises stress"
        title_label = "von Mises stress"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    trip1 = ax.tripcolor(
        moving_xy[:, 0],
        moving_xy[:, 1],
        moving_elements,
        facecolors=moving_scalar,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.tripcolor(
        stationary_xy[:, 0],
        stationary_xy[:, 1],
        stationary_elements,
        facecolors=stationary_scalar,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(trip1, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    time_ms = float(hist[0]) * 1e3
    applied_shear = float(hist[1])
    avg_tau = float(hist[2])
    max_slip = float(hist[5])
    ax.set_title(
        f"Velocity-weakening tatva {title_label}\n"
        f"frame={frame_idx:04d}  time={time_ms:.3f} ms  "
        f"applied_shear={applied_shear:.3f}  avg_tau={avg_tau:.4e}  "
        f"max_slip={max_slip:.4e}  deform_scale={deform_scale:.3g}  {stress_clip_label}"
        ,
        fontsize=11,
        pad=10,
    )
    ax.set_aspect("equal")
    if swap_axes:
        ax.set_xlabel("y")
        ax.set_ylabel("x")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(False)

    output_path = output_dir / f"stress_{frame_idx:05d}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def render_all_frames(
    h5_path: Path,
    output_dir: Path,
    *,
    workers: int,
    dpi: int,
    width: float,
    height: float,
    deform_scale: float | None,
    frame_limit: int | None,
    stress_percentile: float,
    swap_axes: bool,
    margin: float,
    stress_mode: str,
) -> dict[str, float | str | bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stress_abs_max, stress_plot_max, disp_max = compute_global_ranges(
        h5_path,
        stress_mode=stress_mode,
        stress_percentile=stress_percentile,
    )
    if deform_scale is None:
        deform_scale = 1.0 if disp_max <= 0.0 else min(5000.0, 20.0 / disp_max)

    with h5py.File(h5_path, "r") as h5:
        n_frames = int(h5["history"].shape[0])
        moving_coords = np.asarray(h5["moving/coords"], dtype=np.float32)
        stationary_coords = np.asarray(h5["stationary/coords"], dtype=np.float32)
    if frame_limit is not None:
        n_frames = min(n_frames, max(1, int(frame_limit)))

    coords = np.vstack([moving_coords, stationary_coords])
    if coords.shape[1] == 3:
        coords = coords[:, :2]
    if swap_axes:
        coords = coords[:, [1, 0]]
    x_min = float(coords[:, 0].min() - margin)
    x_max = float(coords[:, 0].max() + margin)
    y_min = float(coords[:, 1].min() - margin)
    y_max = float(coords[:, 1].max() + margin)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            str(h5_path),
            str(output_dir),
            float(deform_scale),
            float(stress_plot_max),
            (
                f"abs(vmax)=p{stress_percentile:.1f}={stress_plot_max:.2f}"
                if stress_mode == "sigma_xy"
                else f"vmax=p{stress_percentile:.1f}={stress_plot_max:.2f}"
            ),
            int(dpi),
            (float(width), float(height)),
            bool(swap_axes),
            (x_min, x_max),
            (y_min, y_max),
            str(stress_mode),
        ),
    ) as pool:
        for idx, _ in enumerate(pool.imap_unordered(_render_frame, range(n_frames)), start=1):
            if idx % 200 == 0 or idx == n_frames:
                print(f"[render] {idx}/{n_frames} frames done", flush=True)

    return {
        "frames": float(n_frames),
        "stress_max": float(stress_abs_max),
        "stress_plot_max": float(stress_plot_max),
        "disp_max": float(disp_max),
        "deform_scale": float(deform_scale),
        "swap_axes": bool(swap_axes),
        "stress_mode": stress_mode,
    }


def make_video(
    frames_dir: Path,
    video_path: Path,
    *,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "stress_%05d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Render one stress image per frame and combine them into an mp4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "Velocity-weakening" / "data" / "velocity_weakening_tatva_cpu_20ms.h5",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=root
        / "Velocity-weakening"
        / "Plot"
        / "velocity_weakening_tatva_cpu_20ms_stress_frames",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=root / "Velocity-weakening" / "Plot" / "velocity_weakening_tatva_cpu_20ms_stress_60fps.mp4",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=6.75)
    parser.add_argument("--deform-scale", type=float, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stress-percentile", type=float, default=99.5)
    parser.add_argument("--stress-mode", choices=["von_mises", "sigma_xy"], default="von_mises")
    parser.add_argument("--swap-axes", dest="swap_axes", action="store_true")
    parser.add_argument("--no-swap-axes", dest="swap_axes", action="store_false")
    parser.set_defaults(swap_axes=True)
    parser.add_argument("--margin", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = render_all_frames(
        args.input,
        args.frames_dir,
        workers=max(1, args.workers),
        dpi=args.dpi,
        width=args.width,
        height=args.height,
        deform_scale=args.deform_scale,
        frame_limit=args.max_frames,
        stress_percentile=args.stress_percentile,
        swap_axes=args.swap_axes,
        margin=args.margin,
        stress_mode=args.stress_mode,
    )
    make_video(
        args.frames_dir,
        args.video,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
    )
    print(
        {
            "frames_dir": str(args.frames_dir),
            "video": str(args.video),
            "fps": args.fps,
            **stats,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
