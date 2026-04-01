from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VW_ROOT = REPO_ROOT / "Velocity-weakening"
SRC_ROOT = VW_ROOT / "src"
RUNS_ROOT = VW_ROOT / "runs"
DRIVER = SRC_ROOT / "velocity_weakening_tatva.py"
DEFAULT_MU_DISP_Y_POINTS = [125.0, 250.0, 375.0, 450.0]


def _run_id(run_dir: Path) -> int:
    match = re.match(r"^(\d+)_", run_dir.name)
    return int(match.group(1)) if match else -1


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(limit: int | None = None) -> list[dict[str, Any]]:
    run_dirs = sorted(
        [path for path in RUNS_ROOT.iterdir() if path.is_dir()],
        key=_run_id,
        reverse=True,
    )
    if limit is not None:
        run_dirs = run_dirs[:limit]

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = _load_json(run_dir / "stats" / "summary.json")
        payload = summary or {}
        summary_data = payload.get("summary", {})
        rows.append(
            {
                "run_id": _run_id(run_dir),
                "run_name": run_dir.name,
                "mesh_size": summary_data.get("mesh_size"),
                "saved_frames": summary_data.get("saved_frames"),
                "dt": summary_data.get("dt"),
                "run_dir": str(run_dir),
            }
        )
    return rows


def print_runs(limit: int = 12) -> None:
    rows = list_runs(limit=limit)
    headers = ("run_id", "mesh_size", "saved_frames", "dt", "run_name")
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def resolve_run(
    *,
    run_number: int | None = None,
    run_suffix: str | None = None,
    run_name: str | None = None,
) -> Path:
    if run_name:
        candidate = RUNS_ROOT / run_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Run not found: {candidate}")

    matches = [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
    if run_number is not None:
        matches = [path for path in matches if _run_id(path) == int(run_number)]
    if run_suffix:
        matches = [path for path in matches if path.name.endswith(run_suffix)]

    matches = sorted(matches, key=_run_id, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No run matched run_number={run_number!r}, run_suffix={run_suffix!r}, run_name={run_name!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple runs matched. Narrow it with run_number or a more specific suffix:\n"
            + "\n".join(path.name for path in matches[:10])
        )
    return matches[0]


def run_paths(run_dir: Path) -> dict[str, str]:
    return {
        "run_dir": str(run_dir),
        "data": str(run_dir / "data" / "simulation.h5"),
        "summary": str(run_dir / "stats" / "summary.json"),
        "mu_map": str(run_dir / "Plot" / "mu_eff_map.pdf"),
        "mu_map_phase_split": str(run_dir / "Plot" / "mu_eff_map_phase_split.pdf"),
        "mu_disp": str(run_dir / "Plot" / "contact_mu_disp.pdf"),
        "von_mises_video": str(run_dir / "Plot" / "stress_60fps.mp4"),
        "sigma_xy_video": str(run_dir / "Plot" / "stress_xy_60fps.mp4"),
    }


def load_summary_for_run(run_dir: Path) -> dict[str, Any]:
    payload = _load_json(run_dir / "stats" / "summary.json")
    if payload is None:
        raise FileNotFoundError(f"Missing summary.json for {run_dir}")
    return payload


def default_case_config() -> dict[str, Any]:
    return {
        "dimension": 2,
        "thickness": 0.0,
        "mesh_size": 2.0,
        "cfl": 0.35, # CFL: controls the time step relative to grid size and wave speed to ensure numerical stability (smaller = safer but slower)
        "dtype": "float32",
        "num_threads": 8,
        "normal_stress": 16.0,
        "shear_tau_k": 82.75862068965516,
        "shear_tau_s": 110.34482758620689,
        "normal_phase_time": 0.04,
        "normal_ramp_time": 0.02,
        "shear_phase_time": 0.000530994,
        "shear_scale": 2.5,
        "lock_shear_edge_during_normal": True,
        "frames_per_phase": 4800,
        "shear_frames_per_phase": 28800,
        "omit_initial_frame": True,
        "animation_fps": 60,
        "animation_workers": 8,
        "animation_dpi": 320,
        "animation_width": 12.0,
        "animation_height": 6.75,
        "stress_percentile": 99.5,
        "animation_margin": 8.0,
        "output_prefix": "simulation",
        "run_label": "lockedge-normal40ms-ramp20ms-hold20ms-shear0p530994ms-shearx2p5-mesh2mm",
        "mu_disp_y_points": list(DEFAULT_MU_DISP_Y_POINTS),
        "skip_mu_plot": False,
        "skip_mu_disp_plot": False,
        "skip_animation": False,
        "skip_shear_stress_animation": False,
        "animation_swap_axes": True,
    }


def _append_bool_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def build_driver_command(config: dict[str, Any]) -> list[str]:
    cfg = default_case_config()
    cfg.update(config)

    args = [
        "conda",
        "run",
        "-n",
        "tatva",
        "python",
        str(DRIVER),
        "--platform",
        "cpu",
        "--dimension",
        str(cfg["dimension"]),
        "--thickness",
        str(cfg["thickness"]),
        "--mesh-size",
        str(cfg["mesh_size"]),
        "--cfl",
        str(cfg["cfl"]),
        "--dtype",
        str(cfg["dtype"]),
        "--num-threads",
        str(cfg["num_threads"]),
        "--normal-stress",
        str(cfg["normal_stress"]),
        "--shear-tau-k",
        str(cfg["shear_tau_k"]),
        "--shear-tau-s",
        str(cfg["shear_tau_s"]),
        "--normal-phase-time",
        str(cfg["normal_phase_time"]),
        "--normal-ramp-time",
        str(cfg["normal_ramp_time"]),
        "--shear-phase-time",
        str(cfg["shear_phase_time"]),
        "--shear-scale",
        str(cfg["shear_scale"]),
        "--frames-per-phase",
        str(cfg["frames_per_phase"]),
        "--shear-frames-per-phase",
        str(cfg["shear_frames_per_phase"]),
        "--animation-fps",
        str(cfg["animation_fps"]),
        "--animation-workers",
        str(cfg["animation_workers"]),
        "--animation-dpi",
        str(cfg["animation_dpi"]),
        "--animation-width",
        str(cfg["animation_width"]),
        "--animation-height",
        str(cfg["animation_height"]),
        "--stress-percentile",
        str(cfg["stress_percentile"]),
        "--animation-margin",
        str(cfg["animation_margin"]),
        "--output-prefix",
        str(cfg["output_prefix"]),
        "--run-label",
        str(cfg["run_label"]),
    ]

    if cfg.get("normal_penalty") is not None:
        args.extend(["--normal-penalty", str(cfg["normal_penalty"])])
    if cfg.get("tangential_penalty") is not None:
        args.extend(["--tangential-penalty", str(cfg["tangential_penalty"])])
    if cfg.get("mu_disp_y_points"):
        args.append("--mu-disp-y-points")
        args.extend(str(value) for value in cfg["mu_disp_y_points"])

    _append_bool_flag(args, "--lock-shear-edge-during-normal", bool(cfg["lock_shear_edge_during_normal"]))
    _append_bool_flag(args, "--omit-initial-frame", bool(cfg["omit_initial_frame"]))
    _append_bool_flag(args, "--skip-mu-plot", bool(cfg["skip_mu_plot"]))
    _append_bool_flag(args, "--skip-mu-disp-plot", bool(cfg["skip_mu_disp_plot"]))
    _append_bool_flag(args, "--skip-animation", bool(cfg["skip_animation"]))
    _append_bool_flag(
        args,
        "--skip-shear-stress-animation",
        bool(cfg["skip_shear_stress_animation"]),
    )
    if not bool(cfg["animation_swap_axes"]):
        args.append("--no-animation-swap-axes")

    return args


def launch_case(config: dict[str, Any], *, stream_output: bool = True) -> dict[str, Any]:
    cmd = build_driver_command(config)
    print("Launching:")
    print(" ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=not stream_output,
        check=True,
    )
    if stream_output:
        return {}
    stdout = completed.stdout.strip()
    return json.loads(stdout) if stdout else {}
