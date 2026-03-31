import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tatva.legacy_velocity_weakening import RunConfig, load_legacy_case, run_simulation


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
