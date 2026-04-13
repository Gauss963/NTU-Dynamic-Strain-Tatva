from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if values.size == 1:
        delta = 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=np.float64)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first], mids, [last]))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot effective friction coefficient along the contact line over time."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root
        / "Velocity-weakening"
        / "data"
        / "velocity_weakening_tatva_cpu_normal20ms_shear0p088499ms_9600f.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "Velocity-weakening"
        / "Plot"
        / "velocity_weakening_tatva_cpu_normal20ms_shear0p088499ms_9600f_mu_eff_map.png",
    )
    parser.add_argument(
        "--phase-split-output",
        type=Path,
        default=root
        / "Velocity-weakening"
        / "Plot"
        / "velocity_weakening_tatva_cpu_normal20ms_shear0p088499ms_9600f_mu_eff_map_phase_split.png",
    )
    parser.add_argument("--mu-s", type=float, default=0.8)
    parser.add_argument("--mu-k", type=float, default=0.6)
    parser.add_argument("--d-c", type=float, default=8.0)
    return parser.parse_args()


def plot_mu_eff_maps(
    input_path: Path,
    output_path: Path,
    phase_split_output_path: Path,
    *,
    mu_s: float = 0.8,
    mu_k: float = 0.6,
    d_c: float = 8.0,
) -> dict[str, float | str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        master_nodes = np.asarray(h5["interface/master_nodes"], dtype=np.int32)
        y_coords = np.asarray(h5["moving/coords"], dtype=np.float32)[master_nodes, 1]
        history = np.asarray(h5["history"], dtype=np.float32)
        phase_id = np.asarray(h5["phase_id"], dtype=np.int32)
        cumulative_slip = np.asarray(h5["interface/cumulative_slip"], dtype=np.float32)

    order = np.argsort(y_coords)
    y_sorted = y_coords[order]
    cum_sorted = cumulative_slip[:, order]
    time_ms = history[:, 0] * 1e3
    y_edges = _cell_edges(y_sorted)
    time_edges = _cell_edges(time_ms)

    mu_eff = np.maximum(
        mu_k,
        mu_s - (mu_s - mu_k) * np.minimum(cum_sorted / d_c, 1.0),
    )

    normal_idx = np.where(phase_id == 1)[0]
    normal_end_idx = int(normal_idx[-1]) if normal_idx.size else None

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    im = ax.pcolormesh(
        y_edges,
        time_edges,
        mu_eff,
        cmap="viridis",
        vmin=mu_k,
        vmax=mu_s,
        shading="auto",
        rasterized=True,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Effective friction coefficient")

    if normal_end_idx is not None:
        ax.axhline(time_ms[normal_end_idx], color="white", lw=1.2, ls="--", alpha=0.9)
        ax.text(
            float(y_sorted[0]) + 8.0,
            float(time_ms[normal_end_idx]) + 0.15,
            "normal/shear boundary",
            color="white",
            fontsize=9,
            va="bottom",
        )

    normal_end_min = float(mu_eff[normal_end_idx].min()) if normal_end_idx is not None else float("nan")
    final_min = float(mu_eff[-1].min())
    ax.set_title(
        "Contact-line effective friction coefficient\n"
        f"normal-end min={normal_end_min:.3f}, final min={final_min:.3f}"
    )
    ax.set_xlabel("Contact-line y [mm]")
    ax.set_ylabel("Time [ms]")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    normal_mask = phase_id == 1
    shear_mask = phase_id == 2
    normal_time_ms = (
        time_ms[normal_mask] - float(time_ms[normal_mask][0])
        if np.any(normal_mask)
        else np.zeros(0, dtype=np.float32)
    )
    shear_time_ms = (
        time_ms[shear_mask] - float(time_ms[shear_mask][0])
        if np.any(shear_mask)
        else np.zeros(0, dtype=np.float32)
    )
    normal_time_edges = (
        _cell_edges(normal_time_ms)
        if normal_time_ms.size
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    shear_time_edges = (
        _cell_edges(shear_time_ms)
        if shear_time_ms.size
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    fig = plt.figure(figsize=(10.8, 8.6), dpi=180)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[30.0, 1.0],
        height_ratios=[4.0, 1.0],
        wspace=0.08,
        hspace=0.12,
    )
    ax_shear = fig.add_subplot(gs[0, 0])
    ax_normal = fig.add_subplot(gs[1, 0], sharex=ax_shear)
    cax = fig.add_subplot(gs[:, 1])

    shear_im = ax_shear.pcolormesh(
        y_edges,
        shear_time_edges,
        mu_eff[shear_mask],
        cmap="viridis",
        vmin=mu_k,
        vmax=mu_s,
        shading="auto",
        rasterized=True,
    )
    ax_shear.set_ylabel("Shear phase time [ms]")
    if np.any(shear_mask):
        ax_shear.set_title(
            "Contact-line effective friction coefficient by phase\n"
            f"shear abs window={float(time_ms[shear_mask][0]):.3f}–{float(time_ms[shear_mask][-1]):.3f} ms"
        )
    else:
        ax_shear.set_title("Contact-line effective friction coefficient by phase")

    normal_im = ax_normal.pcolormesh(
        y_edges,
        normal_time_edges,
        mu_eff[normal_mask],
        cmap="viridis",
        vmin=mu_k,
        vmax=mu_s,
        shading="auto",
        rasterized=True,
    )
    ax_normal.set_xlabel("Contact-line y [mm]")
    ax_normal.set_ylabel("Normal phase time [ms]")
    plt.setp(ax_shear.get_xticklabels(), visible=False)
    if np.any(normal_mask):
        ax_normal.text(
            0.01,
            1.02,
            f"normal abs window={float(time_ms[normal_mask][0]):.3f}–{float(time_ms[normal_mask][-1]):.3f} ms",
            transform=ax_normal.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

    cbar = fig.colorbar(normal_im, cax=cax)
    cbar.set_label("Effective friction coefficient")
    fig.savefig(phase_split_output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "output": str(output_path),
        "phase_split_output": str(phase_split_output_path),
        "mu_min_normal_end": normal_end_min,
        "mu_min_final": final_min,
        "mu_mean_normal_end": float(mu_eff[normal_end_idx].mean()) if normal_end_idx is not None else float("nan"),
        "mu_mean_final": float(mu_eff[-1].mean()),
        "cum_slip_max_normal_end": float(cum_sorted[normal_end_idx].max()) if normal_end_idx is not None else float("nan"),
        "cum_slip_max_final": float(cum_sorted[-1].max()),
    }


def main() -> int:
    args = parse_args()
    result = plot_mu_eff_maps(
        args.input,
        args.output,
        args.phase_split_output,
        mu_s=args.mu_s,
        mu_k=args.mu_k,
        d_c=args.d_c,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
