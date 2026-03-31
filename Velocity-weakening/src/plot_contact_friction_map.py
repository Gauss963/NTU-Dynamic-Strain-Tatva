from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

    mu_eff = np.maximum(
        mu_k,
        mu_s - (mu_s - mu_k) * np.minimum(cum_sorted / d_c, 1.0),
    )

    normal_idx = np.where(phase_id == 1)[0]
    normal_end_idx = int(normal_idx[-1]) if normal_idx.size else None

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    im = ax.imshow(
        mu_eff,
        origin="lower",
        aspect="auto",
        extent=[float(y_sorted[0]), float(y_sorted[-1]), float(time_ms[0]), float(time_ms[-1])],
        cmap="viridis",
        vmin=mu_k,
        vmax=mu_s,
        interpolation="nearest",
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

    shear_im = ax_shear.imshow(
        mu_eff[shear_mask],
        origin="lower",
        aspect="auto",
        extent=[
            float(y_sorted[0]),
            float(y_sorted[-1]),
            float(time_ms[shear_mask][0]),
            float(time_ms[shear_mask][-1]),
        ],
        cmap="viridis",
        vmin=mu_k,
        vmax=mu_s,
        interpolation="nearest",
    )
    ax_shear.set_ylabel("Shear time [ms]")
    ax_shear.set_title("Contact-line effective friction coefficient by phase")

    normal_im = ax_normal.imshow(
        mu_eff[normal_mask],
        origin="lower",
        aspect="auto",
        extent=[
            float(y_sorted[0]),
            float(y_sorted[-1]),
            float(time_ms[normal_mask][0]),
            float(time_ms[normal_mask][-1]),
        ],
        cmap="viridis",
        vmin=mu_k,
        vmax=mu_s,
        interpolation="nearest",
    )
    ax_normal.set_xlabel("Contact-line y [mm]")
    ax_normal.set_ylabel("Normal time [ms]")
    plt.setp(ax_shear.get_xticklabels(), visible=False)

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
