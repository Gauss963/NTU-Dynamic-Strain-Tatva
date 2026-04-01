from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _configure_runtime(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--platform", choices=["cpu", "metal"], default="cpu")
    args, _ = parser.parse_known_args(argv)
    if args.platform != "cpu":
        return

    num_threads = max(1, int(args.num_threads))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))

    xla_flag = (
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={num_threads}"
    )
    current = os.environ.get("XLA_FLAGS", "")
    if xla_flag not in current:
        os.environ["XLA_FLAGS"] = f"{current} {xla_flag}".strip()


_configure_runtime(sys.argv[1:])


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tatva.legacy_velocity_weakening import (  # noqa: E402
    RunConfig,
    load_legacy_case,
    run_simulation_dumped,
    save_history_plots,
)
from plot_contact_friction_map import plot_mu_eff_maps  # noqa: E402
from plot_contact_mu_disp import plot_contact_mu_disp  # noqa: E402
from render_stress_frames import make_video, render_all_frames  # noqa: E402


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "simulation"


def _prepare_run_directory(archive_root: Path, run_label: str) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    existing_ids: list[int] = []
    for child in archive_root.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^(\d+)", child.name)
        if match:
            existing_ids.append(int(match.group(1)))
    next_id = max(existing_ids, default=0) + 1
    run_dir = archive_root / f"{next_id:04d}_{_slugify(run_label)}"
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "Plot").mkdir(parents=True, exist_ok=True)
    (run_dir / "stats").mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    base_dir = here.parents[1]
    parser = argparse.ArgumentParser(
        description="tatva rewrite of the Akantu velocity-weakening simulation."
    )
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=2)
    parser.add_argument("--thickness", type=float, default=0.0)
    parser.add_argument("--mesh-size", type=float, default=5.0)
    parser.add_argument("--simulation-time", type=float, default=None)
    parser.add_argument("--normal-phase-time", type=float, default=None)
    parser.add_argument("--shear-phase-time", type=float, default=None)
    parser.add_argument("--normal-ramp-time", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--platform", choices=["cpu", "metal"], default="cpu")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--normal-penalty", type=float, default=None)
    parser.add_argument("--tangential-penalty", type=float, default=None)
    parser.add_argument("--shear-scale", type=float, default=1.0)
    parser.add_argument("--lock-shear-edge-during-normal", action="store_true")
    parser.add_argument("--frames-per-phase", type=int, default=2400)
    parser.add_argument("--shear-frames-per-phase", type=int, default=None)
    parser.add_argument("--omit-initial-frame", action="store_true")
    parser.add_argument("--compression", choices=["lzf", "gzip"], default="lzf")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=base_dir / "runs",
    )
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base_dir / "data",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=base_dir / "Plot",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=base_dir / "stats",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="simulation",
    )
    parser.add_argument("--skip-mu-plot", action="store_true")
    parser.add_argument("--skip-mu-disp-plot", action="store_true")
    parser.add_argument(
        "--mu-disp-y-points",
        type=float,
        nargs="*",
        default=[125.0, 250.0, 375.0, 450.0],
    )
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--skip-shear-stress-animation", action="store_true")
    parser.add_argument("--animation-fps", type=int, default=60)
    parser.add_argument(
        "--animation-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
    )
    parser.add_argument("--animation-dpi", type=int, default=320)
    parser.add_argument("--animation-width", type=float, default=12.0)
    parser.add_argument("--animation-height", type=float, default=6.75)
    parser.add_argument("--deform-scale", type=float, default=None)
    parser.add_argument("--stress-percentile", type=float, default=99.5)
    parser.add_argument("--animation-margin", type=float, default=8.0)
    parser.add_argument("--no-animation-swap-axes", action="store_true")
    return parser.parse_args()


def _render_animation_bundle(
    *,
    data_path: Path,
    plot_dir: Path,
    fps: int,
    workers: int,
    dpi: int,
    width: float,
    height: float,
    deform_scale: float | None,
    stress_percentile: float,
    swap_axes: bool,
    margin: float,
    stress_mode: str,
    stem: str,
) -> dict[str, float | str | bool]:
    frames_dir = plot_dir / f"{stem}_frames"
    video_path = plot_dir / f"{stem}_{fps}fps.mp4"
    animation_stats = render_all_frames(
        data_path,
        frames_dir,
        workers=workers,
        dpi=dpi,
        width=width,
        height=height,
        deform_scale=deform_scale,
        frame_limit=None,
        stress_percentile=stress_percentile,
        swap_axes=swap_axes,
        margin=margin,
        stress_mode=stress_mode,
    )
    make_video(
        frames_dir,
        video_path,
        fps=fps,
        crf=18,
        preset="medium",
    )
    return {
        "frames_dir": str(frames_dir),
        "video": str(video_path),
        "fps": fps,
        **animation_stats,
    }


def main() -> int:
    args = parse_args()
    case = load_legacy_case(REPO_ROOT)
    if args.platform != "cpu":
        raise SystemExit("This driver currently supports CPU only.")

    sim_time = (
        case.simulation.simulation_time
        if args.simulation_time is None
        else args.simulation_time
    )
    normal_phase_time = sim_time if args.normal_phase_time is None else args.normal_phase_time
    shear_phase_time = sim_time if args.shear_phase_time is None else args.shear_phase_time
    run_label = args.run_label or args.output_prefix
    run_dir = _prepare_run_directory(args.archive_root, run_label)
    data_dir = run_dir / "data"
    plot_dir = run_dir / "Plot"
    stats_dir = run_dir / "stats"
    data_path = data_dir / f"{args.output_prefix}.h5"
    result = run_simulation_dumped(
        case,
        RunConfig(
            dimension=args.dimension,
            thickness=args.thickness,
            mesh_size=args.mesh_size,
            simulation_time=sim_time,
            cfl=args.cfl,
            dtype=args.dtype,
            normal_penalty=args.normal_penalty,
            tangential_penalty=args.tangential_penalty,
            shear_scale=args.shear_scale,
            output_prefix=None,
            normal_phase_time=normal_phase_time,
            shear_phase_time=shear_phase_time,
            normal_ramp_time=args.normal_ramp_time,
            lock_shear_edge_during_normal=args.lock_shear_edge_during_normal,
        ),
        data_path,
        frames_per_phase=args.frames_per_phase,
        shear_frames_per_phase=args.shear_frames_per_phase,
        compression=args.compression,
        include_initial_frame=not args.omit_initial_frame,
    )
    saved = save_history_plots(result, plot_dir, prefix=args.output_prefix, extension=".pdf")

    mu_outputs: dict[str, float | str] | None = None
    if not args.skip_mu_plot:
        mu_outputs = plot_mu_eff_maps(
            data_path,
            plot_dir / "mu_eff_map.pdf",
            plot_dir / "mu_eff_map_phase_split.pdf",
            mu_s=case.friction.mu_s,
            mu_k=case.friction.mu_k,
            d_c=case.friction.d_c,
        )

    mu_disp_output: dict[str, float | int | str] | None = None
    if not args.skip_mu_disp_plot:
        mu_disp_output = plot_contact_mu_disp(
            data_path,
            plot_dir / "contact_mu_disp.pdf",
            selection="max-final-slip",
            y_points=list(args.mu_disp_y_points) if args.mu_disp_y_points else None,
        )

    animation_payload: dict[str, dict[str, float | str | bool]] | None = None
    if not args.skip_animation:
        animation_payload = {
            "von_mises": _render_animation_bundle(
                data_path=data_path,
                plot_dir=plot_dir,
                fps=args.animation_fps,
                workers=max(1, args.animation_workers),
                dpi=args.animation_dpi,
                width=args.animation_width,
                height=args.animation_height,
                deform_scale=args.deform_scale,
                stress_percentile=args.stress_percentile,
                swap_axes=not args.no_animation_swap_axes,
                margin=args.animation_margin,
                stress_mode="von_mises",
                stem="stress",
            )
        }
        if not args.skip_shear_stress_animation:
            animation_payload["sigma_xy"] = _render_animation_bundle(
                data_path=data_path,
                plot_dir=plot_dir,
                fps=args.animation_fps,
                workers=max(1, args.animation_workers),
                dpi=args.animation_dpi,
                width=args.animation_width,
                height=args.animation_height,
                deform_scale=args.deform_scale,
                stress_percentile=args.stress_percentile,
                swap_axes=not args.no_animation_swap_axes,
                margin=args.animation_margin,
                stress_mode="sigma_xy",
                stem="stress_xy",
            )

    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "summary.json"
    payload = {
        "summary": result["summary"],
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "stats_path": str(stats_path),
        "plots": [str(path) for path in saved],
        "mu_plots": mu_outputs,
        "mu_disp_plot": mu_disp_output,
        "animation": animation_payload,
        "num_threads": args.num_threads,
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
