from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tatva import Mesh, Operator
from tatva.element import Line2, Tri3


MS_TO_S = 1e-3


@dataclass(frozen=True)
class LegacyBlockSpec:
    name: str
    origin: tuple[float, float]
    dimensions: tuple[float, float]
    tag_prefix: int


@dataclass(frozen=True)
class LegacyMaterial:
    name: str
    rho: float
    E: float
    nu: float

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lmbda(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def cp(self) -> float:
        return math.sqrt((self.lmbda + 2.0 * self.mu) / self.rho)


@dataclass(frozen=True)
class LegacyFriction:
    mu_s: float
    mu_k: float
    d_c: float


@dataclass(frozen=True)
class LegacySimulation:
    simulation_time: float
    time_factor: float
    normal_stress: float
    rise_fraction: float
    tau_k_start_fraction: float
    normal_dir: int
    slave_surface: str
    master_surface: str


@dataclass(frozen=True)
class RunConfig:
    mesh_size: float
    simulation_time: float
    cfl: float
    dtype: str
    normal_penalty: float | None
    tangential_penalty: float | None
    output_prefix: str | None = None
    normal_phase_time: float | None = None
    shear_phase_time: float | None = None
    normal_ramp_time: float | None = None
    lock_shear_edge_during_normal: bool = False
    shear_scale: float = 1.0


@dataclass(frozen=True)
class BlockModel:
    spec: LegacyBlockSpec
    mesh: Mesh
    operator: Operator
    boundary_nodes: dict[str, jax.Array]
    boundary_segments: dict[str, jax.Array]
    boundary_weights: dict[str, jax.Array]

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.coords.shape[0])


@dataclass(frozen=True)
class LegacyCase:
    moving: LegacyBlockSpec
    stationary: LegacyBlockSpec
    materials: dict[str, LegacyMaterial]
    friction: LegacyFriction
    simulation: LegacySimulation


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_legacy_geometry(script_path: Path) -> tuple[LegacyBlockSpec, LegacyBlockSpec]:
    text = _read_text(script_path)
    pattern = re.compile(
        r"Functions\.create_block_2d_quad\(\s*"
        r"origin=\(([^)]*)\),\s*"
        r"dimensions=\(([^)]*)\),\s*"
        r"mesh_size=mesh_size,\s*"
        r"block_name=\"([^\"]+)\",\s*"
        r"tag_prefix=(\d+)",
        re.S,
    )
    blocks: list[LegacyBlockSpec] = []
    for origin_str, dim_str, name, tag_prefix in pattern.findall(text):
        origin_xyz = tuple(float(v.strip()) for v in origin_str.split(","))
        dims_xyz = tuple(float(v.strip()) for v in dim_str.split(","))
        blocks.append(
            LegacyBlockSpec(
                name=name,
                origin=(origin_xyz[0], origin_xyz[1]),
                dimensions=(dims_xyz[0], dims_xyz[1]),
                tag_prefix=int(tag_prefix),
            )
        )
    if len(blocks) != 2:
        raise ValueError(f"Expected 2 blocks in {script_path}, found {len(blocks)}")

    indexed = {block.name: block for block in blocks}
    return indexed["moving-block"], indexed["stationary-block"]


def load_legacy_materials(material_path: Path) -> tuple[dict[str, LegacyMaterial], LegacyFriction]:
    text = _read_text(material_path)

    def parse_key_values(block: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
        return values

    materials: dict[str, LegacyMaterial] = {}
    for block in re.findall(r"material elastic \[(.*?)\]", text, re.S):
        data = parse_key_values(block)
        materials[data["name"]] = LegacyMaterial(
            name=data["name"],
            rho=float(data["rho"]),
            E=float(data["E"]),
            nu=float(data["nu"]),
        )

    friction_blocks = re.findall(
        r"friction linear_slip_weakening no_regularisation \[(.*?)\]",
        text,
        re.S,
    )
    if not friction_blocks:
        raise ValueError(f"Could not parse friction law from {material_path}")
    friction_data = parse_key_values(friction_blocks[0])
    friction = LegacyFriction(
        mu_s=float(friction_data["mu_s"]),
        mu_k=float(friction_data["mu_k"]),
        d_c=float(friction_data["d_c"]),
    )

    if {"moving-block", "stationary-block"} - set(materials):
        raise ValueError(f"Missing moving/stationary material in {material_path}")

    return materials, friction


def load_legacy_simulation(source_path: Path) -> LegacySimulation:
    text = _read_text(source_path)

    def extract_float(pattern: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Missing pattern {pattern!r} in {source_path}")
        return float(match.group(1))

    time_factor = extract_float(r"TIME_FACTOR = ([0-9eE.+-]+);")
    normal_stress = extract_float(r"NORMAL_STRESS = ([0-9eE.+-]+);")
    rise_fraction = extract_float(r"riseEnd = ([0-9eE.+-]+);")
    tau_k_start_fraction = extract_float(r"TAU_K_START_STEP = PRESS_STEPS \* ([0-9eE.+-]+);")
    sim_ms = extract_float(r"SIMULATION_TIME = ([0-9eE.+-]+) \* ms;")
    normal_dir = int(extract_float(r"normal_dir = ([0-9eE.+-]+);"))

    slave = re.search(r'slave_surface = "([^"]+)";', text)
    master = re.search(r'master_surface = "([^"]+)";', text)
    if slave is None or master is None:
        raise ValueError(f"Could not parse contact surface names from {source_path}")

    return LegacySimulation(
        simulation_time=sim_ms * MS_TO_S,
        time_factor=time_factor,
        normal_stress=normal_stress,
        rise_fraction=rise_fraction,
        tau_k_start_fraction=tau_k_start_fraction,
        normal_dir=normal_dir,
        slave_surface=slave.group(1),
        master_surface=master.group(1),
    )


def load_legacy_case(root: Path) -> LegacyCase:
    moving, stationary = load_legacy_geometry(
        root / "NTU-Dynamic-Strain" / "Gmsh" / "Block-Assembly2D.py"
    )
    materials, friction = load_legacy_materials(
        root / "NTU-Dynamic-Strain" / "Materials" / "NTN-LSW.dat"
    )
    simulation = load_legacy_simulation(
        root
        / "NTU-Dynamic-Strain"
        / "Akantu"
        / "Velocity-weakening"
        / "Velocity-weakening-LSW.cc"
    )
    return LegacyCase(
        moving=moving,
        stationary=stationary,
        materials=materials,
        friction=friction,
        simulation=simulation,
    )


def _node_id(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


def create_structured_tri_block(
    spec: LegacyBlockSpec, mesh_size: float, dtype: jnp.dtype
) -> tuple[Mesh, dict[str, jax.Array], dict[str, jax.Array]]:
    x0, y0 = spec.origin
    lx, ly = spec.dimensions
    nx = int(round(lx / mesh_size))
    ny = int(round(ly / mesh_size))
    if not math.isclose(nx * mesh_size, lx, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{spec.name}: mesh_size={mesh_size} does not divide lx={lx}")
    if not math.isclose(ny * mesh_size, ly, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{spec.name}: mesh_size={mesh_size} does not divide ly={ly}")

    x_vals = jnp.linspace(x0, x0 + lx, nx + 1, dtype=dtype)
    y_vals = jnp.linspace(y0, y0 + ly, ny + 1, dtype=dtype)
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing="xy")
    coords = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

    ix, iy = jnp.meshgrid(
        jnp.arange(nx, dtype=jnp.int32),
        jnp.arange(ny, dtype=jnp.int32),
        indexing="xy",
    )
    n00 = (iy * (nx + 1) + ix).reshape(-1)
    n10 = n00 + 1
    n01 = n00 + (nx + 1)
    n11 = n01 + 1
    tri1 = jnp.stack([n00, n10, n11], axis=-1)
    tri2 = jnp.stack([n00, n11, n01], axis=-1)
    elements = jnp.concatenate([tri1, tri2], axis=0)

    def make_edge_nodes(which: str) -> jax.Array:
        if which == "back":
            return jnp.arange(0, (ny + 1) * (nx + 1), nx + 1, dtype=jnp.int32)
        if which == "front":
            return jnp.arange(nx, (ny + 1) * (nx + 1) + nx, nx + 1, dtype=jnp.int32)
        if which == "right":
            return jnp.arange(nx + 1, dtype=jnp.int32)
        if which == "left":
            return jnp.arange(ny * (nx + 1), (ny + 1) * (nx + 1), dtype=jnp.int32)
        raise ValueError(which)

    boundary_nodes = {
        f"{spec.name}-back": make_edge_nodes("back"),
        f"{spec.name}-front": make_edge_nodes("front"),
        f"{spec.name}-right": make_edge_nodes("right"),
        f"{spec.name}-left": make_edge_nodes("left"),
    }
    boundary_segments = {
        name: jnp.stack([nodes[:-1], nodes[1:]], axis=-1)
        for name, nodes in boundary_nodes.items()
    }

    return (
        Mesh(
            coords=coords,
            elements=elements,
        ),
        boundary_nodes,
        boundary_segments,
    )


def make_line_operator(mesh: Mesh, segments: jax.Array) -> Operator:
    return Operator(mesh._replace(elements=segments), Line2())


def boundary_weights(mesh: Mesh, segments: jax.Array, dtype: jnp.dtype) -> jax.Array:
    line_op = make_line_operator(mesh, segments)
    zeros = jnp.zeros(mesh.coords.shape[0], dtype=dtype)
    return jax.jacrev(lambda q: line_op.integrate(q))(zeros)


def build_block_model(
    spec: LegacyBlockSpec, mesh_size: float, dtype: jnp.dtype
) -> BlockModel:
    mesh, boundary_nodes, boundary_segments = create_structured_tri_block(
        spec, mesh_size, dtype
    )
    operator = Operator(mesh, Tri3())
    weights = {
        name: boundary_weights(mesh, segments, dtype)
        for name, segments in boundary_segments.items()
    }
    return BlockModel(
        spec=spec,
        mesh=mesh,
        operator=operator,
        boundary_nodes=boundary_nodes,
        boundary_segments=boundary_segments,
        boundary_weights=weights,
    )


def lumped_mass(operator: Operator, density: float, dtype: jnp.dtype) -> jax.Array:
    # `jacrev(op.integrate)` gives the nodal integration weights.
    # For a constant density field, the lumped mass is density times those weights.
    weights = jax.jacrev(lambda q: operator.integrate(q))(
        jnp.zeros(operator.mesh.coords.shape[0], dtype=dtype)
    )
    return weights * jnp.asarray(density, dtype=dtype)


def make_global_dof_indices(nodes: jax.Array | np.ndarray, offset: int, component: int) -> jax.Array:
    return offset + 2 * nodes + component


def make_dirichlet_dofs(stationary: BlockModel, moving_offset: int) -> jax.Array:
    front_nodes = stationary.boundary_nodes["stationary-block-front"]
    top_nodes = stationary.boundary_nodes["stationary-block-left"]
    dofs = jnp.concatenate(
        [
            make_global_dof_indices(front_nodes, moving_offset, 0),
            make_global_dof_indices(top_nodes, moving_offset, 1),
        ]
    )
    return jnp.unique(dofs.astype(jnp.int32))


def match_interface_nodes(
    master: BlockModel, slave: BlockModel, master_surface: str, slave_surface: str
) -> tuple[jax.Array, jax.Array]:
    master_nodes = master.boundary_nodes[master_surface]
    slave_nodes = slave.boundary_nodes[slave_surface]
    master_y = master.mesh.coords[master_nodes, 1]
    slave_y = slave.mesh.coords[slave_nodes, 1]
    match = jnp.isclose(master_y[:, None], slave_y[None, :], atol=1e-6)
    matched = jnp.any(match, axis=1)
    if not bool(np.asarray(jnp.any(matched))):
        raise ValueError("No overlapping interface nodes were found.")
    slave_match_idx = jnp.argmax(match, axis=1)
    master_idx = jnp.where(matched)[0]
    return master_nodes[master_idx], slave_nodes[slave_match_idx[master_idx]]


def triangle_min_edge_length(mesh: Mesh) -> float:
    pts = mesh.coords[mesh.elements]
    edges = jnp.concatenate(
        [
            pts[:, 1] - pts[:, 0],
            pts[:, 2] - pts[:, 1],
            pts[:, 0] - pts[:, 2],
        ],
        axis=0,
    )
    return float(jnp.min(jnp.linalg.norm(edges, axis=1)))


def build_case_model(case: LegacyCase, config: RunConfig) -> dict[str, Any]:
    dtype = jnp.float32 if config.dtype == "float32" else jnp.float64

    moving = build_block_model(case.moving, config.mesh_size, dtype)
    stationary = build_block_model(case.stationary, config.mesh_size, dtype)

    moving_material = case.materials["moving-block"]
    stationary_material = case.materials["stationary-block"]

    moving_mass = lumped_mass(moving.operator, moving_material.rho, dtype)
    stationary_mass = lumped_mass(stationary.operator, stationary_material.rho, dtype)

    moving_n = moving.n_nodes
    stationary_n = stationary.n_nodes
    moving_offset = 2 * moving_n
    total_dofs = 2 * (moving_n + stationary_n)

    fixed_dofs = make_dirichlet_dofs(stationary, moving_offset)
    moving_shear_edge_dofs = make_global_dof_indices(
        moving.boundary_nodes["moving-block-right"],
        0,
        1,
    ).astype(jnp.int32)

    mass_flat = jnp.concatenate(
        [
            jnp.repeat(moving_mass, 2),
            jnp.repeat(stationary_mass, 2),
        ]
    )

    force_normal = jnp.zeros(total_dofs, dtype=dtype).at[
        make_global_dof_indices(
            jnp.arange(moving_n, dtype=jnp.int32),
            0,
            0,
        )
    ].add(
        moving.boundary_weights["moving-block-back"]
        * jnp.asarray(case.simulation.normal_stress, dtype=dtype)
    )

    force_shear_unit = jnp.zeros(total_dofs, dtype=dtype).at[
        make_global_dof_indices(
            jnp.arange(moving_n, dtype=jnp.int32),
            0,
            1,
        )
    ].add(moving.boundary_weights["moving-block-right"])

    master_nodes, slave_nodes = match_interface_nodes(
        moving,
        stationary,
        case.simulation.master_surface,
        case.simulation.slave_surface,
    )
    interface_weights = moving.boundary_weights[case.simulation.master_surface][master_nodes]

    penalty_n = (
        config.normal_penalty
        if config.normal_penalty is not None
        else 10.0 * moving_material.E / config.mesh_size
    )
    penalty_t = (
        config.tangential_penalty
        if config.tangential_penalty is not None
        else penalty_n * 0.1
    )

    hmin = min(triangle_min_edge_length(moving.mesh), triangle_min_edge_length(stationary.mesh))
    cp = max(moving_material.cp, stationary_material.cp)
    dt_bulk = config.cfl * hmin / cp
    min_mass = float(jnp.min(mass_flat[jnp.setdiff1d(jnp.arange(total_dofs), fixed_dofs)]))
    max_interface_weight = float(jnp.max(interface_weights))
    dt_contact = 0.25 * math.sqrt(min_mass / max(penalty_n, penalty_t) / max_interface_weight)
    dt = min(dt_bulk, dt_contact)

    pressure_time = (
        config.normal_phase_time
        if config.normal_phase_time is not None
        else config.simulation_time
    )
    shear_time = (
        config.shear_phase_time
        if config.shear_phase_time is not None
        else config.simulation_time
    )

    pressure_steps = max(1, int(math.ceil(pressure_time / dt)))
    shear_steps = max(1, int(math.ceil(shear_time / dt)))
    normal_ramp_time = (
        0.0
        if config.normal_ramp_time is None
        else min(max(float(config.normal_ramp_time), 0.0), pressure_time)
    )
    normal_ramp_steps = (
        min(pressure_steps, int(math.ceil(normal_ramp_time / dt)))
        if normal_ramp_time > 0.0
        else 0
    )
    rise_steps = max(1, int(math.ceil(case.simulation.rise_fraction * shear_steps)))
    tau_k_start_step = min(
        pressure_steps - 1,
        max(0, int(math.floor(case.simulation.tau_k_start_fraction * pressure_steps))),
    )
    shear_ratio = case.moving.dimensions[1] / case.stationary.dimensions[0]
    tau_scale = max(float(config.shear_scale), 0.0)
    tau_k = tau_scale * shear_ratio * case.friction.mu_k * case.simulation.normal_stress
    tau_s = tau_scale * shear_ratio * case.friction.mu_s * case.simulation.normal_stress
    dtau = (tau_s - tau_k) / rise_steps

    scalar_dtype = np.float32 if dtype == jnp.float32 else np.float64
    pressure_schedule = np.zeros(pressure_steps, dtype=scalar_dtype)
    pressure_schedule[tau_k_start_step:] = tau_k
    shear_schedule = np.full(shear_steps, tau_k, dtype=scalar_dtype)
    for i in range(shear_steps):
        if i < rise_steps:
            shear_schedule[i] = tau_k + (i + 1) * dtau
        else:
            shear_schedule[i] = tau_s

    normal_schedule_pressure = np.ones(pressure_steps, dtype=scalar_dtype)
    if normal_ramp_steps > 0:
        normal_schedule_pressure[:normal_ramp_steps] = (
            np.arange(1, normal_ramp_steps + 1, dtype=scalar_dtype) / normal_ramp_steps
        )
    normal_schedule_shear = np.ones(shear_steps, dtype=scalar_dtype)

    return {
        "dtype": dtype,
        "moving": moving,
        "stationary": stationary,
        "moving_material": moving_material,
        "stationary_material": stationary_material,
        "fixed_dofs": fixed_dofs,
        "moving_shear_edge_dofs": moving_shear_edge_dofs,
        "force_normal": force_normal,
        "force_shear_unit": force_shear_unit,
        "mass_flat": mass_flat,
        "master_nodes": master_nodes,
        "slave_nodes": slave_nodes,
        "interface_weights": interface_weights,
        "penalty_n": jnp.asarray(penalty_n, dtype=dtype),
        "penalty_t": jnp.asarray(penalty_t, dtype=dtype),
        "dt": float(dt),
        "pressure_schedule": jnp.asarray(pressure_schedule, dtype=dtype),
        "shear_schedule": jnp.asarray(shear_schedule, dtype=dtype),
        "normal_schedule_pressure": jnp.asarray(normal_schedule_pressure, dtype=dtype),
        "normal_schedule_shear": jnp.asarray(normal_schedule_shear, dtype=dtype),
        "tau_k": float(tau_k),
        "tau_s": float(tau_s),
        "shear_scale": tau_scale,
        "pressure_time": float(pressure_time),
        "shear_time": float(shear_time),
        "normal_ramp_time": float(normal_ramp_time),
        "normal_ramp_steps": normal_ramp_steps,
        "rise_steps": rise_steps,
        "pressure_steps": pressure_steps,
        "shear_steps": shear_steps,
        "moving_offset": moving_offset,
        "total_dofs": total_dofs,
    }


def run_simulation(case: LegacyCase, config: RunConfig) -> dict[str, Any]:
    model = build_case_model(case, config)

    dtype = model["dtype"]
    moving = model["moving"]
    stationary = model["stationary"]
    moving_material = model["moving_material"]
    stationary_material = model["stationary_material"]
    base_fixed_dofs = model["fixed_dofs"]
    moving_shear_edge_dofs = model["moving_shear_edge_dofs"]
    force_normal = model["force_normal"]
    force_shear_unit = model["force_shear_unit"]
    mass_flat = model["mass_flat"]
    master_nodes = model["master_nodes"]
    slave_nodes = model["slave_nodes"]
    interface_weights = model["interface_weights"]
    penalty_n = model["penalty_n"]
    penalty_t = model["penalty_t"]
    dt = model["dt"]
    moving_offset = model["moving_offset"]
    total_dofs = model["total_dofs"]

    n_moving = moving.n_nodes
    n_stationary = stationary.n_nodes
    friction = case.friction

    if config.lock_shear_edge_during_normal:
        normal_phase_fixed_dofs = jnp.unique(
            jnp.concatenate([base_fixed_dofs, moving_shear_edge_dofs])
        )
    else:
        normal_phase_fixed_dofs = base_fixed_dofs
    shear_phase_fixed_dofs = base_fixed_dofs

    def apply_dirichlet(vec: jax.Array, active_fixed_dofs: jax.Array) -> jax.Array:
        return vec.at[active_fixed_dofs].set(0.0)

    def split_u(u_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            u_flat[: 2 * n_moving].reshape(n_moving, 2),
            u_flat[2 * n_moving :].reshape(n_stationary, 2),
        )

    def elastic_energy_total(u_flat: jax.Array) -> jax.Array:
        u_moving, u_stationary = split_u(u_flat)
        eps_moving = moving.operator.grad(u_moving)
        eps_stationary = stationary.operator.grad(u_stationary)

        def strain_energy_density(grad_u: jax.Array, mat: LegacyMaterial) -> jax.Array:
            eps = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
            return mat.mu * jnp.einsum("...ij,...ij->...", eps, eps) + 0.5 * mat.lmbda * jnp.trace(
                eps, axis1=-2, axis2=-1
            ) ** 2

        return moving.operator.integrate(
            strain_energy_density(eps_moving, moving_material)
        ) + stationary.operator.integrate(
            strain_energy_density(eps_stationary, stationary_material)
        )

    elastic_energy_and_force = jax.jit(jax.value_and_grad(elastic_energy_total))

    total_interface_length = jnp.sum(interface_weights)
    moving_iface_x = 2 * master_nodes
    moving_iface_y = 2 * master_nodes + 1
    stationary_iface_x = moving_offset + 2 * slave_nodes
    stationary_iface_y = moving_offset + 2 * slave_nodes + 1

    def contact_response(
        u_flat: jax.Array, plastic_slip: jax.Array, cum_slip: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        u_moving, u_stationary = split_u(u_flat)
        rel_normal = u_moving[master_nodes, 0] - u_stationary[slave_nodes, 0]
        penetration = jnp.maximum(rel_normal, 0.0)
        in_contact = penetration > 0.0

        rel_tangent = u_moving[master_nodes, 1] - u_stationary[slave_nodes, 1]
        trial_tau = penalty_t * (rel_tangent - plastic_slip)
        normal_traction = penalty_n * penetration
        mu_eff = jnp.maximum(
            friction.mu_k,
            friction.mu_s
            - (friction.mu_s - friction.mu_k)
            * jnp.minimum(cum_slip / friction.d_c, 1.0),
        )
        yield_tau = mu_eff * normal_traction
        sliding = in_contact & (jnp.abs(trial_tau) > yield_tau)
        tau = jnp.where(
            in_contact,
            jnp.where(sliding, jnp.sign(trial_tau) * yield_tau, trial_tau),
            0.0,
        )
        new_plastic = jnp.where(sliding, rel_tangent - tau / penalty_t, plastic_slip)
        new_cum = cum_slip + jnp.where(sliding, jnp.abs(new_plastic - plastic_slip), 0.0)

        forces = jnp.zeros(total_dofs, dtype=dtype)
        forces = forces.at[moving_iface_x].add(interface_weights * normal_traction)
        forces = forces.at[stationary_iface_x].add(-interface_weights * normal_traction)
        forces = forces.at[moving_iface_y].add(interface_weights * tau)
        forces = forces.at[stationary_iface_y].add(-interface_weights * tau)

        elastic_gap = jnp.where(in_contact, rel_tangent - new_plastic, 0.0)
        interface_energy = jnp.sum(
            interface_weights
            * (
                0.5 * penalty_n * penetration**2
                + 0.5 * penalty_t * elastic_gap**2
            )
        )
        diagnostics = {
            "avg_tau": jnp.sum(interface_weights * tau) / total_interface_length,
            "avg_sigma_n": jnp.sum(interface_weights * normal_traction) / total_interface_length,
            "max_penetration": jnp.max(penetration),
            "max_slip": jnp.max(new_cum),
            "mu_eff_mean": jnp.sum(interface_weights * mu_eff) / total_interface_length,
            "interface_energy": interface_energy,
        }
        return forces, new_plastic, new_cum, diagnostics

    def acceleration(
        u_flat: jax.Array,
        plastic_slip: jax.Array,
        cum_slip: jax.Array,
        normal_scale: jax.Array,
        shear_traction: jax.Array,
        active_fixed_dofs: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        elastic_energy, elastic_force = elastic_energy_and_force(u_flat)
        contact_force, plastic_new, cum_new, contact_diag = contact_response(
            u_flat, plastic_slip, cum_slip
        )
        force_ext = normal_scale * force_normal + shear_traction * force_shear_unit
        accel = apply_dirichlet(
            (force_ext - elastic_force - contact_force) / mass_flat,
            active_fixed_dofs,
        )
        diag = {
            "elastic_energy": elastic_energy,
            "kinetic_energy": jnp.array(0.0, dtype=dtype),
            **contact_diag,
        }
        diag["plastic_slip"] = plastic_new
        diag["cum_slip"] = cum_new
        return accel, diag

    u0 = jnp.zeros(total_dofs, dtype=dtype)
    plastic0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    cum0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    a0, diag0 = acceleration(
        u0,
        plastic0,
        cum0,
        model["normal_schedule_pressure"][0],
        model["pressure_schedule"][0],
        normal_phase_fixed_dofs,
    )
    v_half0 = apply_dirichlet(0.5 * dt * a0, normal_phase_fixed_dofs)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        loading: jax.Array,
        active_fixed_dofs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        u_flat, v_half, plastic_slip, cum_slip, time_now = carry
        normal_scale = loading[0]
        shear_traction = loading[1]
        u_new = apply_dirichlet(u_flat + dt * v_half, active_fixed_dofs)
        accel, diag = acceleration(
            u_new,
            plastic_slip,
            cum_slip,
            normal_scale,
            shear_traction,
            active_fixed_dofs,
        )
        v_half_new = apply_dirichlet(v_half + dt * accel, active_fixed_dofs)
        kinetic = 0.5 * jnp.sum(mass_flat * v_half_new**2)
        output = jnp.array(
            [
                time_now + dt,
                shear_traction,
                diag["avg_tau"],
                diag["avg_sigma_n"],
                diag["max_penetration"],
                diag["max_slip"],
                diag["mu_eff_mean"],
                diag["elastic_energy"],
                diag["interface_energy"],
                kinetic,
            ],
            dtype=dtype,
        )
        return (
            u_new,
            v_half_new,
            diag["plastic_slip"],
            diag["cum_slip"],
            time_now + dt,
        ), output

    def run_phase(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        schedule: jax.Array,
        active_fixed_dofs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        return jax.jit(
            lambda c, s: jax.lax.scan(
                lambda carry_state, loading: step(carry_state, loading, active_fixed_dofs),
                c,
                s,
            )
        )(carry, schedule)

    carry0 = (u0, v_half0, diag0["plastic_slip"], diag0["cum_slip"], jnp.array(0.0, dtype=dtype))
    carry1, hist_pressure = run_phase(
        carry0,
        jnp.stack([model["normal_schedule_pressure"], model["pressure_schedule"]], axis=1),
        normal_phase_fixed_dofs,
    )
    carry2, hist_shear = run_phase(
        carry1,
        jnp.stack([model["normal_schedule_shear"], model["shear_schedule"]], axis=1),
        shear_phase_fixed_dofs,
    )
    jax.block_until_ready(hist_shear)

    history = np.vstack([np.asarray(hist_pressure), np.asarray(hist_shear)])
    column_names = [
        "time",
        "applied_shear",
        "avg_tau",
        "avg_sigma_n",
        "max_penetration",
        "max_slip",
        "mu_eff_mean",
        "elastic_energy",
        "interface_energy",
        "kinetic_energy",
    ]
    final_u, final_v, final_plastic, final_cum, final_time = carry2

    summary = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(np.asarray(history).dtype),
        "mesh_size": config.mesh_size,
        "dt": dt,
        "pressure_time": model["pressure_time"],
        "shear_time": model["shear_time"],
        "normal_ramp_time": model["normal_ramp_time"],
        "normal_ramp_steps": model["normal_ramp_steps"],
        "pressure_steps": model["pressure_steps"],
        "shear_steps": model["shear_steps"],
        "tau_k": model["tau_k"],
        "tau_s": model["tau_s"],
        "shear_scale": model["shear_scale"],
        "normal_penalty": float(penalty_n),
        "tangential_penalty": float(penalty_t),
        "lock_shear_edge_during_normal": bool(config.lock_shear_edge_during_normal),
        "moving_nodes": n_moving,
        "stationary_nodes": n_stationary,
        "moving_elements": int(moving.mesh.elements.shape[0]),
        "stationary_elements": int(stationary.mesh.elements.shape[0]),
        "final_time": float(final_time),
        "final_max_slip": float(jnp.max(final_cum)),
        "final_max_penetration": float(history[-1, 4]),
        "final_avg_tau": float(history[-1, 2]),
        "final_avg_sigma_n": float(history[-1, 3]),
        "legacy": {
            "materials": {name: asdict(mat) for name, mat in case.materials.items()},
            "friction": asdict(case.friction),
            "simulation": asdict(case.simulation),
            "geometry": {
                "moving": asdict(case.moving),
                "stationary": asdict(case.stationary),
            },
        },
    }

    if config.output_prefix:
        prefix = Path(config.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            prefix.with_suffix(".npz"),
            history=history,
            columns=np.asarray(column_names, dtype=object),
            final_u=np.asarray(final_u),
            final_v_half=np.asarray(final_v),
            final_plastic_slip=np.asarray(final_plastic),
            final_cum_slip=np.asarray(final_cum),
        )
        prefix.with_suffix(".json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return {"summary": summary, "history": history, "columns": column_names}


def run_simulation_dumped(
    case: LegacyCase,
    config: RunConfig,
    data_path: Path,
    *,
    frames_per_phase: int = 2400,
    shear_frames_per_phase: int | None = None,
    compression: str = "lzf",
    include_initial_frame: bool = True,
) -> dict[str, Any]:
    import h5py

    model = build_case_model(case, config)

    dtype = model["dtype"]
    moving = model["moving"]
    stationary = model["stationary"]
    moving_material = model["moving_material"]
    stationary_material = model["stationary_material"]
    base_fixed_dofs = model["fixed_dofs"]
    moving_shear_edge_dofs = model["moving_shear_edge_dofs"]
    force_normal = model["force_normal"]
    force_shear_unit = model["force_shear_unit"]
    mass_flat = model["mass_flat"]
    master_nodes = model["master_nodes"]
    slave_nodes = model["slave_nodes"]
    interface_weights = model["interface_weights"]
    penalty_n = model["penalty_n"]
    penalty_t = model["penalty_t"]
    dt = model["dt"]
    moving_offset = model["moving_offset"]
    total_dofs = model["total_dofs"]

    n_moving = moving.n_nodes
    n_stationary = stationary.n_nodes
    friction = case.friction

    if config.lock_shear_edge_during_normal:
        normal_phase_fixed_dofs = jnp.unique(
            jnp.concatenate([base_fixed_dofs, moving_shear_edge_dofs])
        )
    else:
        normal_phase_fixed_dofs = base_fixed_dofs
    shear_phase_fixed_dofs = base_fixed_dofs

    def apply_dirichlet(vec: jax.Array, active_fixed_dofs: jax.Array) -> jax.Array:
        return vec.at[active_fixed_dofs].set(0.0)

    def split_flat(flat: jax.Array, n_nodes: int) -> jax.Array:
        return flat.reshape(n_nodes, 2)

    def split_u(u_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            split_flat(u_flat[: 2 * n_moving], n_moving),
            split_flat(u_flat[2 * n_moving :], n_stationary),
        )

    def compute_strain(grad_u: jax.Array) -> jax.Array:
        return 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))

    def compute_stress(eps: jax.Array, mat: LegacyMaterial) -> jax.Array:
        return 2.0 * mat.mu * eps + mat.lmbda * jnp.trace(
            eps, axis1=-2, axis2=-1
        )[..., None, None] * jnp.eye(2, dtype=eps.dtype)

    def collapse_element_field(field: jax.Array) -> jax.Array:
        return field[:, 0] if field.ndim == 4 else field

    def elastic_energy_total(u_flat: jax.Array) -> jax.Array:
        u_moving, u_stationary = split_u(u_flat)
        eps_moving = compute_strain(moving.operator.grad(u_moving))
        eps_stationary = compute_strain(stationary.operator.grad(u_stationary))
        return moving.operator.integrate(
            moving_material.mu * jnp.einsum("...ij,...ij->...", eps_moving, eps_moving)
            + 0.5
            * moving_material.lmbda
            * jnp.trace(eps_moving, axis1=-2, axis2=-1) ** 2
        ) + stationary.operator.integrate(
            stationary_material.mu
            * jnp.einsum("...ij,...ij->...", eps_stationary, eps_stationary)
            + 0.5
            * stationary_material.lmbda
            * jnp.trace(eps_stationary, axis1=-2, axis2=-1) ** 2
        )

    elastic_energy_and_force = jax.jit(jax.value_and_grad(elastic_energy_total))

    total_interface_length = jnp.sum(interface_weights)
    moving_iface_x = 2 * master_nodes
    moving_iface_y = 2 * master_nodes + 1
    stationary_iface_x = moving_offset + 2 * slave_nodes
    stationary_iface_y = moving_offset + 2 * slave_nodes + 1

    def contact_response(
        u_flat: jax.Array, plastic_slip: jax.Array, cum_slip: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        u_moving, u_stationary = split_u(u_flat)
        rel_normal = u_moving[master_nodes, 0] - u_stationary[slave_nodes, 0]
        penetration = jnp.maximum(rel_normal, 0.0)
        in_contact = penetration > 0.0

        rel_tangent = u_moving[master_nodes, 1] - u_stationary[slave_nodes, 1]
        trial_tau = penalty_t * (rel_tangent - plastic_slip)
        normal_traction = penalty_n * penetration
        mu_eff = jnp.maximum(
            friction.mu_k,
            friction.mu_s
            - (friction.mu_s - friction.mu_k)
            * jnp.minimum(cum_slip / friction.d_c, 1.0),
        )
        yield_tau = mu_eff * normal_traction
        sliding = in_contact & (jnp.abs(trial_tau) > yield_tau)
        tau = jnp.where(
            in_contact,
            jnp.where(sliding, jnp.sign(trial_tau) * yield_tau, trial_tau),
            0.0,
        )
        new_plastic = jnp.where(sliding, rel_tangent - tau / penalty_t, plastic_slip)
        new_cum = cum_slip + jnp.where(
            sliding, jnp.abs(new_plastic - plastic_slip), 0.0
        )

        forces = jnp.zeros(total_dofs, dtype=dtype)
        forces = forces.at[moving_iface_x].add(interface_weights * normal_traction)
        forces = forces.at[stationary_iface_x].add(-interface_weights * normal_traction)
        forces = forces.at[moving_iface_y].add(interface_weights * tau)
        forces = forces.at[stationary_iface_y].add(-interface_weights * tau)

        elastic_gap = jnp.where(in_contact, rel_tangent - new_plastic, 0.0)
        interface_energy = jnp.sum(
            interface_weights
            * (
                0.5 * penalty_n * penetration**2
                + 0.5 * penalty_t * elastic_gap**2
            )
        )
        diagnostics = {
            "avg_tau": jnp.sum(interface_weights * tau) / total_interface_length,
            "avg_sigma_n": jnp.sum(interface_weights * normal_traction)
            / total_interface_length,
            "max_penetration": jnp.max(penetration),
            "max_slip": jnp.max(new_cum),
            "mu_eff_mean": jnp.sum(interface_weights * mu_eff) / total_interface_length,
            "interface_energy": interface_energy,
        }
        return forces, new_plastic, new_cum, diagnostics

    def acceleration(
        u_flat: jax.Array,
        plastic_slip: jax.Array,
        cum_slip: jax.Array,
        normal_scale: jax.Array,
        shear_traction: jax.Array,
        active_fixed_dofs: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        elastic_energy, elastic_force = elastic_energy_and_force(u_flat)
        contact_force, plastic_new, cum_new, contact_diag = contact_response(
            u_flat, plastic_slip, cum_slip
        )
        force_ext = normal_scale * force_normal + shear_traction * force_shear_unit
        accel = apply_dirichlet(
            (force_ext - elastic_force - contact_force) / mass_flat,
            active_fixed_dofs,
        )
        diag = {
            "elastic_energy": elastic_energy,
            "kinetic_energy": jnp.array(0.0, dtype=dtype),
            **contact_diag,
        }
        diag["plastic_slip"] = plastic_new
        diag["cum_slip"] = cum_new
        return accel, diag

    def make_row(
        time_now: jax.Array,
        shear_traction: jax.Array,
        diag: dict[str, jax.Array],
        kinetic: jax.Array,
    ) -> jax.Array:
        return jnp.array(
            [
                time_now,
                shear_traction,
                diag["avg_tau"],
                diag["avg_sigma_n"],
                diag["max_penetration"],
                diag["max_slip"],
                diag["mu_eff_mean"],
                diag["elastic_energy"],
                diag["interface_energy"],
                kinetic,
            ],
            dtype=dtype,
        )

    u0 = jnp.zeros(total_dofs, dtype=dtype)
    plastic0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    cum0 = jnp.zeros(master_nodes.shape[0], dtype=dtype)
    a0, diag0 = acceleration(
        u0,
        plastic0,
        cum0,
        model["normal_schedule_pressure"][0],
        jnp.asarray(0.0, dtype=dtype),
        normal_phase_fixed_dofs,
    )
    v_half0 = apply_dirichlet(0.5 * dt * a0, normal_phase_fixed_dofs)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        loading: jax.Array,
        active_fixed_dofs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        u_flat, v_half, plastic_slip, cum_slip, time_now = carry
        normal_scale = loading[0]
        shear_traction = loading[1]
        u_new = apply_dirichlet(u_flat + dt * v_half, active_fixed_dofs)
        accel, diag = acceleration(
            u_new,
            plastic_slip,
            cum_slip,
            normal_scale,
            shear_traction,
            active_fixed_dofs,
        )
        v_half_new = apply_dirichlet(v_half + dt * accel, active_fixed_dofs)
        kinetic = 0.5 * jnp.sum(mass_flat * v_half_new**2)
        output = make_row(time_now + dt, shear_traction, diag, kinetic)
        return (
            u_new,
            v_half_new,
            diag["plastic_slip"],
            diag["cum_slip"],
            time_now + dt,
        ), output

    @jax.jit
    def observe_fields(
        u_flat: jax.Array, v_flat: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        u_moving, u_stationary = split_u(u_flat)
        v_moving, v_stationary = split_u(v_flat)
        eps_moving = collapse_element_field(compute_strain(moving.operator.grad(u_moving)))
        eps_stationary = collapse_element_field(
            compute_strain(stationary.operator.grad(u_stationary))
        )
        sigma_moving = compute_stress(eps_moving, moving_material)
        sigma_stationary = compute_stress(eps_stationary, stationary_material)
        return (
            u_moving,
            u_stationary,
            v_moving,
            v_stationary,
            eps_moving,
            eps_stationary,
            sigma_moving,
            sigma_stationary,
        )

    chunk_runners: dict[tuple[str, int], Any] = {}

    def advance_chunk(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        schedule_chunk: jax.Array,
        active_fixed_dofs: jax.Array,
        phase_label: str,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        length = int(schedule_chunk.shape[0])
        key = (phase_label, length)
        if key not in chunk_runners:
            chunk_runners[key] = jax.jit(
                lambda c, s: jax.lax.scan(
                    lambda carry_state, loading: step(
                        carry_state, loading, active_fixed_dofs
                    ),
                    c,
                    s,
                )
            )
        return chunk_runners[key](carry, schedule_chunk)

    shear_frames_per_phase = (
        frames_per_phase if shear_frames_per_phase is None else shear_frames_per_phase
    )
    save_every_pressure = max(1, math.ceil(model["pressure_steps"] / frames_per_phase))
    save_every_shear = max(1, math.ceil(model["shear_steps"] / shear_frames_per_phase))
    n_press_frames = len(range(0, model["pressure_steps"], save_every_pressure))
    n_shear_frames = len(range(0, model["shear_steps"], save_every_shear))
    total_frames = (1 if include_initial_frame else 0) + n_press_frames + n_shear_frames

    history_columns = [
        "time",
        "applied_shear",
        "avg_tau",
        "avg_sigma_n",
        "max_penetration",
        "max_slip",
        "mu_eff_mean",
        "elastic_energy",
        "interface_energy",
        "kinetic_energy",
    ]

    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    def _create_group_datasets(h5: h5py.File, name: str, n_frames: int, n_nodes: int, n_elem: int):
        grp = h5.create_group(name)
        grp.create_dataset("coords", data=np.asarray(moving.mesh.coords if name == "moving" else stationary.mesh.coords))
        grp.create_dataset("elements", data=np.asarray(moving.mesh.elements if name == "moving" else stationary.mesh.elements))
        kwargs = dict(compression=compression, chunks=(1, n_nodes, 2))
        grp.create_dataset("displacement", shape=(n_frames, n_nodes, 2), dtype="f4", **kwargs)
        grp.create_dataset("velocity", shape=(n_frames, n_nodes, 2), dtype="f4", **kwargs)
        elem_kwargs = dict(compression=compression, chunks=(1, n_elem, 2, 2))
        grp.create_dataset("strain", shape=(n_frames, n_elem, 2, 2), dtype="f4", **elem_kwargs)
        grp.create_dataset("stress", shape=(n_frames, n_elem, 2, 2), dtype="f4", **elem_kwargs)
        return grp

    def save_frame(
        h5: h5py.File,
        frame_idx: int,
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        history_row: np.ndarray,
        *,
        phase_id: int,
        step_id: int,
    ) -> None:
        u_flat, v_half, plastic_slip, cum_slip, _time_now = carry
        (
            u_moving,
            u_stationary,
            v_moving,
            v_stationary,
            eps_moving,
            eps_stationary,
            sigma_moving,
            sigma_stationary,
        ) = observe_fields(u_flat, v_half)
        h5["moving/displacement"][frame_idx] = np.asarray(u_moving, dtype=np.float32)
        h5["moving/velocity"][frame_idx] = np.asarray(v_moving, dtype=np.float32)
        h5["moving/strain"][frame_idx] = np.asarray(eps_moving, dtype=np.float32)
        h5["moving/stress"][frame_idx] = np.asarray(sigma_moving, dtype=np.float32)
        h5["stationary/displacement"][frame_idx] = np.asarray(u_stationary, dtype=np.float32)
        h5["stationary/velocity"][frame_idx] = np.asarray(v_stationary, dtype=np.float32)
        h5["stationary/strain"][frame_idx] = np.asarray(eps_stationary, dtype=np.float32)
        h5["stationary/stress"][frame_idx] = np.asarray(sigma_stationary, dtype=np.float32)
        h5["history"][frame_idx] = history_row.astype(np.float32)
        h5["phase_id"][frame_idx] = phase_id
        h5["step_id"][frame_idx] = step_id
        h5["interface/plastic_slip"][frame_idx] = np.asarray(plastic_slip, dtype=np.float32)
        h5["interface/cumulative_slip"][frame_idx] = np.asarray(cum_slip, dtype=np.float32)

    kinetic0 = 0.5 * jnp.sum(mass_flat * v_half0**2)
    initial_row = np.asarray(
        make_row(jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), diag0, kinetic0)
    )
    carry = (u0, v_half0, diag0["plastic_slip"], diag0["cum_slip"], jnp.asarray(0.0, dtype=dtype))

    with h5py.File(data_path, "w") as h5:
        h5.attrs["backend"] = jax.default_backend()
        h5.attrs["dt"] = dt
        h5.attrs["save_every_pressure"] = save_every_pressure
        h5.attrs["save_every_shear"] = save_every_shear
        h5.attrs["pressure_steps"] = model["pressure_steps"]
        h5.attrs["shear_steps"] = model["shear_steps"]
        h5.attrs["pressure_frames_target"] = frames_per_phase
        h5.attrs["shear_frames_target"] = shear_frames_per_phase
        h5.attrs["include_initial_frame"] = int(include_initial_frame)
        h5.attrs["normal_ramp_time"] = model["normal_ramp_time"]
        h5.attrs["normal_ramp_steps"] = model["normal_ramp_steps"]
        h5.attrs["lock_shear_edge_during_normal"] = int(
            config.lock_shear_edge_during_normal
        )
        h5.create_dataset("history_columns", data=np.asarray(history_columns, dtype="S"))
        h5.create_dataset("phase_id", shape=(total_frames,), dtype="i4")
        h5.create_dataset("step_id", shape=(total_frames,), dtype="i4")
        h5.create_dataset(
            "history",
            shape=(total_frames, len(history_columns)),
            dtype="f4",
            compression=compression,
            chunks=(min(256, total_frames), len(history_columns)),
        )
        moving_grp = _create_group_datasets(
            h5, "moving", total_frames, n_moving, int(moving.mesh.elements.shape[0])
        )
        stationary_grp = _create_group_datasets(
            h5,
            "stationary",
            total_frames,
            n_stationary,
            int(stationary.mesh.elements.shape[0]),
        )
        iface = h5.create_group("interface")
        iface.attrs["mu_static"] = friction.mu_s
        iface.attrs["mu_kinetic"] = friction.mu_k
        iface.attrs["critical_slip"] = friction.d_c
        iface.create_dataset("master_nodes", data=np.asarray(master_nodes))
        iface.create_dataset("slave_nodes", data=np.asarray(slave_nodes))
        iface.create_dataset(
            "contact_line_y",
            data=np.asarray(moving.mesh.coords[np.asarray(master_nodes), 1], dtype=np.float32),
        )
        iface.create_dataset(
            "plastic_slip",
            shape=(total_frames, int(master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(master_nodes.shape[0])),
        )
        iface.create_dataset(
            "cumulative_slip",
            shape=(total_frames, int(master_nodes.shape[0])),
            dtype="f4",
            compression=compression,
            chunks=(1, int(master_nodes.shape[0])),
        )

        if include_initial_frame:
            save_frame(h5, 0, carry, initial_row, phase_id=0, step_id=0)
            frame_count = 1
        else:
            frame_count = 0

        phases = [
            (
                1,
                "normal",
                np.column_stack(
                    [
                        np.asarray(model["normal_schedule_pressure"]),
                        np.asarray(model["pressure_schedule"]),
                    ]
                ),
                save_every_pressure,
                normal_phase_fixed_dofs,
            ),
            (
                2,
                "shear",
                np.column_stack(
                    [
                        np.asarray(model["normal_schedule_shear"]),
                        np.asarray(model["shear_schedule"]),
                    ]
                ),
                save_every_shear,
                shear_phase_fixed_dofs,
            ),
        ]
        for phase_id, phase_label, schedule, save_every, active_fixed_dofs in phases:
            schedule_np = np.asarray(schedule)
            phase_steps = schedule_np.shape[0]
            for start in range(0, phase_steps, save_every):
                stop = min(start + save_every, phase_steps)
                chunk = jnp.asarray(schedule_np[start:stop], dtype=dtype)
                carry, outputs = advance_chunk(
                    carry,
                    chunk,
                    active_fixed_dofs,
                    phase_label,
                )
                row = np.asarray(outputs[-1])
                save_frame(h5, frame_count, carry, row, phase_id=phase_id, step_id=stop)
                frame_count += 1

        h5.attrs["saved_frames"] = frame_count

    final_u, final_v, final_plastic, final_cum, final_time = carry
    with h5py.File(data_path, "r") as h5:
        history = np.asarray(h5["history"][:frame_count])

    summary = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(history.dtype),
        "mesh_size": config.mesh_size,
        "dt": dt,
        "pressure_time": model["pressure_time"],
        "shear_time": model["shear_time"],
        "normal_ramp_time": model["normal_ramp_time"],
        "normal_ramp_steps": model["normal_ramp_steps"],
        "pressure_steps": model["pressure_steps"],
        "shear_steps": model["shear_steps"],
        "tau_k": model["tau_k"],
        "tau_s": model["tau_s"],
        "shear_scale": model["shear_scale"],
        "normal_penalty": float(penalty_n),
        "tangential_penalty": float(penalty_t),
        "lock_shear_edge_during_normal": bool(config.lock_shear_edge_during_normal),
        "moving_nodes": n_moving,
        "stationary_nodes": n_stationary,
        "moving_elements": int(moving.mesh.elements.shape[0]),
        "stationary_elements": int(stationary.mesh.elements.shape[0]),
        "final_time": float(final_time),
        "final_max_slip": float(jnp.max(final_cum)),
        "final_max_penetration": float(history[-1, 4]),
        "final_avg_tau": float(history[-1, 2]),
        "final_avg_sigma_n": float(history[-1, 3]),
        "saved_frames": int(frame_count),
        "data_path": str(data_path),
        "legacy": {
            "materials": {name: asdict(mat) for name, mat in case.materials.items()},
            "friction": asdict(case.friction),
            "simulation": asdict(case.simulation),
            "geometry": {
                "moving": asdict(case.moving),
                "stationary": asdict(case.stationary),
            },
        },
    }

    return {
        "summary": summary,
        "history": history,
        "columns": history_columns,
        "final_u": np.asarray(final_u),
        "final_v_half": np.asarray(final_v),
        "final_plastic_slip": np.asarray(final_plastic),
        "final_cum_slip": np.asarray(final_cum),
    }


def save_history_plots(
    result: dict[str, Any],
    plot_dir: Path,
    *,
    prefix: str = "velocity_weakening_tatva",
    extension: str = ".pdf",
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plot_dir.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result["history"])
    columns = list(result["columns"])
    col = {name: idx for idx, name in enumerate(columns)}
    time_ms = history[:, col["time"]] * 1e3

    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_ms, history[:, col["applied_shear"]], label="Applied shear", lw=2)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Applied shear")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(
        time_ms,
        history[:, col["avg_tau"]],
        label="Average interface shear",
        lw=2,
        color="tab:orange",
    )
    ax2.plot(
        time_ms,
        history[:, col["avg_sigma_n"]],
        label="Average interface normal",
        lw=2,
        color="tab:green",
    )
    ax2.set_ylabel("Interface traction")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper center")
    path = plot_dir / f"{prefix}_tractions{extension}"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.0), sharex=True)
    axes[0].plot(time_ms, history[:, col["max_slip"]], label="Max cumulative slip", lw=2)
    axes[0].plot(time_ms, history[:, col["max_penetration"]], label="Max penetration", lw=2)
    axes[0].set_ylabel("Slip / penetration")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(
        time_ms,
        history[:, col["mu_eff_mean"]],
        label="Mean effective friction",
        lw=2,
        color="tab:green",
    )
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Effective friction")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    path = plot_dir / f"{prefix}_interface_state{extension}"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_ms, history[:, col["elastic_energy"]], label="Elastic energy", lw=2)
    ax.plot(time_ms, history[:, col["interface_energy"]], label="Interface energy", lw=2)
    ax.plot(time_ms, history[:, col["kinetic_energy"]], label="Kinetic energy", lw=2)
    ax.plot(
        time_ms,
        history[:, col["elastic_energy"]]
        + history[:, col["interface_energy"]]
        + history[:, col["kinetic_energy"]],
        label="Total tracked energy",
        lw=2,
        ls="--",
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = plot_dir / f"{prefix}_energies{extension}"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite of the Akantu velocity-weakening example in tatva."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing NTU-Dynamic-Strain.",
    )
    parser.add_argument("--mesh-size", type=float, default=25.0)
    parser.add_argument("--simulation-time", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--normal-penalty", type=float, default=None)
    parser.add_argument("--tangential-penalty", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = load_legacy_case(args.root)
    config = RunConfig(
        mesh_size=args.mesh_size,
        simulation_time=(
            case.simulation.simulation_time
            if args.simulation_time is None
            else args.simulation_time
        ),
        cfl=args.cfl,
        dtype=args.dtype,
        normal_penalty=args.normal_penalty,
        tangential_penalty=args.tangential_penalty,
        output_prefix=args.output_prefix,
    )
    result = run_simulation(case, config)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
