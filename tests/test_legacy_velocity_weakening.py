import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tatva.legacy_velocity_weakening import (
    RunConfig,
    load_legacy_case,
    run_simulation,
    run_simulation_dumped,
)


def test_legacy_velocity_weakening_smoke():
    root = Path(__file__).resolve().parents[1]
    case = load_legacy_case(root)
    result = run_simulation(
        case,
        RunConfig(
            mesh_size=5.0,
            simulation_time=1e-6,
            cfl=0.3,
            dtype="float32",
            normal_penalty=500.0,
            tangential_penalty=50.0,
            output_prefix=None,
            normal_phase_time=2e-6,
            shear_phase_time=1e-6,
            normal_ramp_time=1e-6,
            lock_shear_edge_during_normal=True,
        ),
    )
    summary = result["summary"]
    assert summary["pressure_steps"] > 0
    assert summary["shear_steps"] > 0
    assert summary["moving_nodes"] > 0
    assert result["history"].shape[1] == len(result["columns"])


def test_dumped_frame_sampling_tracks_targets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    case = load_legacy_case(root)
    output = tmp_path / "sampled.h5"
    result = run_simulation_dumped(
        case,
        RunConfig(
            mesh_size=5.0,
            simulation_time=1e-6,
            cfl=0.3,
            dtype="float32",
            normal_penalty=500.0,
            tangential_penalty=50.0,
            output_prefix=None,
            normal_phase_time=2e-6,
            shear_phase_time=1e-6,
            normal_ramp_time=1e-6,
            lock_shear_edge_during_normal=True,
        ),
        output,
        frames_per_phase=7,
        shear_frames_per_phase=9,
        include_initial_frame=False,
    )
    summary = result["summary"]
    assert summary["pressure_frames_saved"] == min(summary["pressure_steps"], 7)
    assert summary["shear_frames_saved"] == min(summary["shear_steps"], 9)
    assert summary["saved_frames"] == (
        summary["pressure_frames_saved"] + summary["shear_frames_saved"]
    )


def test_3d_dumped_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    case = load_legacy_case(root)
    output = tmp_path / "sampled_3d.h5"
    result = run_simulation_dumped(
        case,
        RunConfig(
            dimension=3,
            thickness=10.0,
            mesh_size=100.0,
            simulation_time=5e-7,
            cfl=0.2,
            dtype="float32",
            normal_penalty=500.0,
            tangential_penalty=50.0,
            output_prefix=None,
            normal_phase_time=5e-7,
            shear_phase_time=5e-7,
            normal_ramp_time=2.5e-7,
            lock_shear_edge_during_normal=True,
        ),
        output,
        frames_per_phase=4,
        shear_frames_per_phase=4,
        include_initial_frame=False,
    )
    summary = result["summary"]
    assert summary["dimension"] == 3
    assert summary["thickness"] == 10.0
    assert summary["moving_nodes"] > 0
    assert summary["saved_frames"] > 0
