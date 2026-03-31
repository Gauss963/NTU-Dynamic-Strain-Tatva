# NTU Dynamic Strain Tatva Fork

This repository is a focused fork of `tatva` for Akantu-to-tatva migration work in the NTU Dynamic Strain project.
It is no longer a general showcase repository for the upstream library. Instead, it is organized around one practical goal:
building, running, and extending converted simulation cases, starting with the 2D velocity-weakening problem.

## Scope

This fork keeps:

- the `tatva` core required by the converted simulations
- the velocity-weakening parser and solver bridge
- the case driver and post-processing scripts
- one smoke test for end-to-end validation
- one example entry point for the converted case

This fork removes or de-emphasizes:

- upstream release and mirror automation
- branding assets
- unrelated examples
- unrelated test and benchmark files

## Repository Layout

```text
tatva/
  tatva/
    legacy_velocity_weakening.py
    ...
  Velocity-weakening/
    src/
      velocity_weakening_tatva.py
      plot_contact_friction_map.py
      plot_contact_mu_disp.py
      render_stress_frames.py
    Mesh/
    Materials/
    Plot/
    data/
    stats/
    runs/
  example/
    velocity_weakening_2d.py
  tests/
    test_legacy_velocity_weakening.py
  NTU-Dynamic-Strain/
    ...
```

## Local Reference Data

The current conversion reads legacy inputs from the local reference folder:

- `NTU-Dynamic-Strain/Gmsh/Block-Assembly2D.py`
- `NTU-Dynamic-Strain/Materials/NTN-LSW.dat`
- `NTU-Dynamic-Strain/Akantu/Velocity-weakening/Velocity-weakening-LSW.cc`

`NTU-Dynamic-Strain/` is intentionally ignored by git.
It is treated as local reference input only.

## Environment Setup

The recommended workflow uses the `tatva` conda environment.

If the environment already exists:

```bash
conda activate tatva
pip install -e .
pip install matplotlib h5py pytest
```

If you need a fresh environment:

```bash
conda create -n tatva python=3.12 -y
conda activate tatva
pip install -e .
pip install matplotlib h5py pytest
```

You also need `ffmpeg` on `PATH` for MP4 animation output.

## Current Execution Model

- CPU is the default runtime target.
- JAX multithreading is configured automatically by the case driver.
- Static plots are written as PDF.
- Animation frames are written as PNG and encoded to MP4.
- Animations default to standard 4K (`3840x2160`).
- Stress animations default to swapped axes: `y` is horizontal and `x` is vertical.
- Frame rendering uses multiple worker processes by default.

## Run The Existing Velocity-Weakening Case

From the repository root:

```bash
conda run -n tatva python ./Velocity-weakening/src/velocity_weakening_tatva.py \
  --mesh-size 5 \
  --dtype float32 \
  --normal-phase-time 0.04 \
  --shear-phase-time 5.30994e-04 \
  --normal-ramp-time 0.02 \
  --cfl 0.01004185 \
  --shear-scale 2.5 \
  --frames-per-phase 4800 \
  --shear-frames-per-phase 28800 \
  --omit-initial-frame \
  --lock-shear-edge-during-normal \
  --run-label lockedge-normal40ms-ramp20ms-hold20ms-shear0p530994ms-shearx2p5 \
  --num-threads 8 \
  --animation-workers 8
```

## Main Driver

The main production driver is:

- `Velocity-weakening/src/velocity_weakening_tatva.py`

Important arguments:

- `--mesh-size`
  Structured mesh spacing in millimeters.
- `--normal-phase-time`
  Duration of the normal loading phase in seconds.
- `--shear-phase-time`
  Duration of the shear loading phase in seconds.
- `--normal-ramp-time`
  Time used to ramp normal stress before the hold stage.
- `--shear-scale`
  Multiplier applied to the legacy shear traction level.
- `--frames-per-phase`
  Target dump count for the normal phase.
- `--shear-frames-per-phase`
  Target dump count for the shear phase.
- `--lock-shear-edge-during-normal`
  Fixes the future shear edge during the normal phase.
- `--num-threads`
  CPU thread count passed to JAX/XLA.
- `--animation-workers`
  Worker count for frame rendering.
- `--skip-animation`
  Skip MP4 creation.
- `--skip-mu-plot`
  Skip the friction heatmap.
- `--skip-mu-disp-plot`
  Skip the friction-versus-slip plot.
- `--no-animation-swap-axes`
  Disable the default swapped-axis animation layout.

## Where Outputs Go

Each run is archived automatically under:

```text
Velocity-weakening/runs/000X_<run-label>/
```

Inside a run directory:

- `data/simulation.h5`
  Full dumped fields, history arrays, and interface data.
- `stats/summary.json`
  Run summary and output locations.
- `Plot/*.pdf`
  Static plots in PDF format.
- `Plot/stress_frames/`
  PNG frames for the default animation.
- `Plot/stress_60fps.mp4`
  Default MP4 animation.

## Post-Processing Scripts

These scripts can be re-run after the simulation if `simulation.h5` already exists:

- `Velocity-weakening/src/plot_contact_friction_map.py`
- `Velocity-weakening/src/plot_contact_mu_disp.py`
- `Velocity-weakening/src/render_stress_frames.py`

### Rebuild friction maps

```bash
conda run -n tatva python ./Velocity-weakening/src/plot_contact_friction_map.py \
  --input ./Velocity-weakening/runs/0004_example/data/simulation.h5 \
  --output ./Velocity-weakening/runs/0004_example/Plot/mu_eff_map.pdf \
  --phase-split-output ./Velocity-weakening/runs/0004_example/Plot/mu_eff_map_phase_split.pdf
```

### Rebuild a `mu-slip` plot

```bash
conda run -n tatva python ./Velocity-weakening/src/plot_contact_mu_disp.py \
  --input ./Velocity-weakening/runs/0004_example/data/simulation.h5 \
  --output ./Velocity-weakening/runs/0004_example/Plot/contact_mu_disp.pdf
```

### Rebuild the default 4K animation

```bash
conda run -n tatva python ./Velocity-weakening/src/render_stress_frames.py \
  --input ./Velocity-weakening/runs/0004_example/data/simulation.h5 \
  --frames-dir ./Velocity-weakening/runs/0004_example/Plot/stress_frames \
  --video ./Velocity-weakening/runs/0004_example/Plot/stress_60fps.mp4 \
  --workers 8
```

## How To Add A New Case

The current velocity-weakening conversion is not a completely generic importer.
The cleanest way to add a new case is to follow the same project pattern.

### Step 1: Create a new case folder

Create a sibling folder next to `Velocity-weakening`, for example:

```text
My-New-Case/
  src/
  Mesh/
  Materials/
  Plot/
  data/
  stats/
  runs/
```

Generated outputs do not need to be tracked by git.

### Step 2: Add a dedicated legacy loader

The current loader is:

- `tatva/legacy_velocity_weakening.py`

It is hard-coded to parse the current legacy geometry, material, and simulation files.
For a new case, duplicate this module and adapt:

- geometry parsing
- material parsing
- friction law parsing
- load schedule parsing
- contact definition
- boundary conditions

Suggested naming pattern:

```text
tatva/legacy_<case_name>.py
```

### Step 3: Add a case driver

Use:

- `Velocity-weakening/src/velocity_weakening_tatva.py`

as the template for the new driver.
Replace:

- the imported loader
- the default case label
- any case-specific defaults

Suggested naming pattern:

```text
<CaseFolder>/src/<case_name>_tatva.py
```

### Step 4: Reuse or adapt the post-processing scripts

The current plotting and animation scripts assume the same HDF5 layout as the velocity-weakening workflow.
If your new case writes the same groups and field names, you can reuse them directly.
If not, duplicate the scripts and adapt the dataset names.

### Step 5: Add a smoke test

Use:

- `tests/test_legacy_velocity_weakening.py`

as the template for a new end-to-end smoke test.
The goal is to confirm that parsing, assembly, time integration, dumping, and post-processing still work together.

## Running The Smoke Test

```bash
conda run -n tatva pytest ./tests/test_legacy_velocity_weakening.py -q
```

## Git Notes

- `origin` points to this fork.
- `upstream` points to the original `smec-ethz/tatva` repository.
- `NTU-Dynamic-Strain/` is ignored on purpose.
- generated HDF5 files, plots, videos, and archived runs are ignored on purpose.

## License

This codebase remains under the GNU Lesser General Public License v3.0 or later.
See `COPYING` and `COPYING.LESSER` for the full license text.
