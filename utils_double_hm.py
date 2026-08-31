from config_double_hm import (
    MASSFLOW_KEY,
    DISPLACEMENT_KEY,
    PRJ_TEMPLATE,
    OGS_BIN,
    OUT_ROOT,
    BOTTOM_NODE_IDS,
    TOP_NODE_ID,
    REF_CSV,
    LOG_CSV,
    OPTIMIZER,
    LOSS_FUNCTION,
    VTU_PATTERN,
    VERTICAL_COMPONENT,
    SETTLEMENT_POSITIVE_DOWNWARD,
    TARGET_WEIGHTS,
)

import os
import csv
import shutil
import numpy as np
import pandas as pd
import meshio
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ─── Extract time step from VTU file name ───────────────────────────────────
def extract_time_from_filename(p: Path) -> float:
    """Extracts the time value from a VTU filename."""
    try:
        return float(p.stem.split("_t_")[-1])
    except ValueError:
        return np.nan


# ─── Read HM benchmark targets from VTU files ───────────────────────────────
def read_hm_targets(folder: Path):
    """
    Extracts both calibration targets from VTU files:

    1. MassFlowRate summed over bottom boundary nodes
    2. Settlement at the top-center node
    """
    files = sorted(folder.glob(VTU_PATTERN), key=extract_time_from_filename)

    times, flow_vals, settlement_vals = [], [], []
    bottom_nodes = np.asarray(BOTTOM_NODE_IDS, dtype=int)

    for f in files:
        try:
            mesh = meshio.read(f)
            t = extract_time_from_filename(f)

            mass_flow = np.asarray(mesh.point_data[MASSFLOW_KEY], dtype=float)
            displacement = np.asarray(mesh.point_data[DISPLACEMENT_KEY], dtype=float)

            q_bottom = float(np.sum(mass_flow[bottom_nodes]))
            uy_top = float(displacement[TOP_NODE_ID, VERTICAL_COMPONENT])

            if SETTLEMENT_POSITIVE_DOWNWARD:
                settlement_top = -uy_top
            else:
                settlement_top = uy_top

            times.append(t)
            flow_vals.append(q_bottom)
            settlement_vals.append(settlement_top)

        except Exception:
            continue

    times = np.asarray(times, dtype=float)
    flow_vals = np.asarray(flow_vals, dtype=float)
    settlement_vals = np.asarray(settlement_vals, dtype=float)

    order = np.argsort(times)

    return times[order], flow_vals[order], settlement_vals[order]


# ─── Standard loss calculation ──────────────────────────────────────────────
def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """Basic metrics: rmse, mse, mae, r2."""
    if method == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif method == "mse":
        return mean_squared_error(y_true, y_pred)
    elif method == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif method == "r2":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown loss function: {method}")


# ─── Normalization scale for each target ────────────────────────────────────
def normalization_scale(y_true: np.ndarray) -> float:
    """
    Returns a normalization scale for one target.

    This is needed because MassFlowRate and settlement have different units.
    """
    scale = np.ptp(y_true)

    if scale > 1e-30:
        return scale

    scale = np.max(np.abs(y_true))

    if scale > 1e-30:
        return scale

    return 1.0


# ─── Normalized loss: only exact timestamp matches ──────────────────────────
def normalized_loss_on_exact_matches(t_ref, y_ref, t_sim, y_sim, method: str):
    """
    Compute normalized loss using ONLY samples whose timestamps match EXACTLY
    in both reference and simulation.
    """
    t_ref = np.asarray(t_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    t_sim = np.asarray(t_sim, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)

    # Sort both
    ref_ord = np.argsort(t_ref)
    t_ref = t_ref[ref_ord]
    y_ref = y_ref[ref_ord]

    sim_ord = np.argsort(t_sim)
    t_sim = t_sim[sim_ord]
    y_sim = y_sim[sim_ord]

    # Find EXACT common timestamps
    common = np.intersect1d(t_ref, t_sim, assume_unique=False)

    if common.size == 0:
        raise RuntimeError("No EXACT matching timestamps between reference and simulation.")

    # Build dicts to map time -> value
    ref_map = dict(zip(t_ref, y_ref))
    sim_map = dict(zip(t_sim, y_sim))

    y_true = np.array([ref_map[t] for t in common], dtype=float)
    y_pred = np.array([sim_map[t] for t in common], dtype=float)

    # Guard for degenerate R²
    if method == "r2":
        if len(y_true) < 2 or np.allclose(y_true, y_true.mean()):
            raw_metric = 0.0
        else:
            raw_metric = compute_loss(y_true, y_pred, method)

        objective_part = -raw_metric
        scale = 1.0

    else:
        raw_metric = compute_loss(y_true, y_pred, method)
        scale = normalization_scale(y_true)

        if method == "mse":
            objective_part = raw_metric / (scale ** 2)
        else:
            objective_part = raw_metric / scale

    return objective_part, raw_metric, scale, common.size


# ─── Class to Modify and Write OGS Project File ─────────────────────────────
class OGSProject:
    """Represents and modifies an OpenGeoSys project file (.prj)."""

    def __init__(self, template: Path):
        self.tree = ET.parse(template)
        self.root = self.tree.getroot()
        self.base_dir = template.parent

    def set_permeability(self, k: float):
        node = self.root.find(".//media//property[name='permeability']/value")
        if node is None:
            raise RuntimeError("No <permeability> element found")
        node.text = f"{k:.6e}"

    def set_youngs_modulus(self, E: float):
        node = self.root.find(".//parameter[name='E']/value")
        if node is None:
            raise RuntimeError("No <E> element found")
        node.text = f"{E:.6e}"

    def set_poisson_ratio(self, nu: float):
        node = self.root.find(".//parameter[name='nu']/value")
        if node is None:
            raise RuntimeError("No <nu> element found")
        node.text = f"{nu:.6e}"

    def set_porosity(self, phi: float):
        node = self.root.find(".//media//property[name='porosity']/value")
        if node is None:
            raise RuntimeError("No <porosity> element found")
        node.text = f"{phi:.6e}"

        self.set_initial_stress_porosity(phi)

    def set_initial_stress_porosity(self, phi: float):
        initial_stress = None

        for param in self.root.findall(".//parameters/parameter"):
            name = param.find("name")
            if name is not None and (name.text or "").strip() == "Initial_stress":
                initial_stress = param
                break

        if initial_stress is None:
            raise RuntimeError("No <Initial_stress> parameter found")

        expressions = initial_stress.findall("expression")

        if len(expressions) < 2:
            raise RuntimeError("Initial_stress must contain at least two <expression> entries")

        expressions[1].text = (
            f"-((1-{phi:.10g})*2e3+{phi:.10g}*1e3)*9.81*(1-y)"
        )

    def set_bishop_exponent(self, bishop_exponent: float):
        node = self.root.find(
            ".//media//property[name='bishops_effective_stress']/exponent"
        )
        if node is None:
            raise RuntimeError("No BishopsPowerLaw <exponent> element found")
        node.text = f"{bishop_exponent:.6e}"

    def mesh_sources(self):
        meshes = []
        for elem in self.root.findall(".//meshes/mesh"):
            txt = (elem.text or "").strip()
            if txt:
                p = Path(txt)
                meshes.append(p if p.is_absolute() else self.base_dir / p)
        return meshes

    def write(self, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.tree.write(destination, encoding="ISO-8859-1")


# ─── Run OGS Simulation ─────────────────────────────────────────────────────
def run_simulation(
    k_perm: float,
    E: float,
    nu: float,
    phi: float,
    bishop_exponent: float,
    run_dir: Path,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    prj = OGSProject(PRJ_TEMPLATE)
    prj.set_permeability(k_perm)
    prj.set_youngs_modulus(E)
    prj.set_poisson_ratio(nu)
    prj.set_porosity(phi)
    prj.set_bishop_exponent(bishop_exponent)

    prj_file = run_dir / PRJ_TEMPLATE.name
    prj.write(prj_file)

    for mesh_path in prj.mesh_sources():
        dst = run_dir / mesh_path.name
        if not dst.exists():
            try:
                dst.symlink_to(mesh_path)
            except OSError:
                shutil.copy2(mesh_path, dst)

    ret = os.system(f"{OGS_BIN} {prj_file} -o {run_dir}")
    if ret != 0:
        raise RuntimeError(f"OGS exited with status {ret}")


# ─── Objective Function ─────────────────────────────────────────────────────
def objective(params) -> float:
    """
    1. Run OGS simulation
    2. Extract MassFlowRate summed over bottom boundary nodes
    3. Extract settlement at TOP_NODE_ID
    4. Compare both targets with reference data at EXACT matching timestamps
    5. Return weighted normalized double-target loss
    """
    k_perm = np.nan
    E = np.nan
    nu = np.nan
    phi = np.nan
    bishop_exponent = np.nan

    try:
        k_perm, E, nu, phi, bishop_exponent = [float(v) for v in params]

        run_dir = OUT_ROOT / (
            f"run_k_{k_perm:.2e}"
            f"_E_{E:.2e}"
            f"_nu_{nu:.3f}"
            f"_phi_{phi:.3f}"
            f"_bishop_exponent_{bishop_exponent:.3f}"
        )

        run_simulation(k_perm, E, nu, phi, bishop_exponent, run_dir)

        t_sim, q_sim, settlement_sim = read_hm_targets(run_dir)

        if t_sim.size == 0:
            raise RuntimeError("No simulation data found")

        ref = pd.read_csv(REF_CSV)

        t_ref = ref["time"].to_numpy(dtype=float)
        q_ref = ref[MASSFLOW_KEY].to_numpy(dtype=float)
        settlement_ref = ref["settlement_top"].to_numpy(dtype=float)

        flow_loss, flow_raw, flow_scale, n_flow = normalized_loss_on_exact_matches(
            t_ref=t_ref,
            y_ref=q_ref,
            t_sim=t_sim,
            y_sim=q_sim,
            method=LOSS_FUNCTION,
        )

        settlement_loss, settlement_raw, settlement_scale, n_settlement = (
            normalized_loss_on_exact_matches(
                t_ref=t_ref,
                y_ref=settlement_ref,
                t_sim=t_sim,
                y_sim=settlement_sim,
                method=LOSS_FUNCTION,
            )
        )

        w_flow = TARGET_WEIGHTS["MassFlowRate"]
        w_settlement = TARGET_WEIGHTS["settlement_top"]
        w_sum = w_flow + w_settlement

        if w_sum <= 0:
            raise RuntimeError("The sum of TARGET_WEIGHTS must be positive")

        w_flow = w_flow / w_sum
        w_settlement = w_settlement / w_sum

        metric_val = w_flow * flow_loss + w_settlement * settlement_loss

        status, note = "ok", ""

    except Exception as e:
        metric_val = 1e9

        flow_loss = np.nan
        settlement_loss = np.nan

        flow_raw = np.nan
        settlement_raw = np.nan

        flow_scale = np.nan
        settlement_scale = np.nan

        n_flow = 0
        n_settlement = 0

        status, note = "fail", str(e)

    # Log result
    with LOG_CSV.open("a", newline="") as f:
        w = csv.writer(f)

        if f.tell() == 0:
            w.writerow([
                "algorithm",

                "permeability",
                "youngs_modulus",
                "poisson_ratio",
                "porosity",
                "bishop_exponent",

                "total_loss",
                "metric",

                "flow_loss",
                "settlement_loss",

                "flow_raw_metric",
                "settlement_raw_metric",

                "flow_scale",
                "settlement_scale",

                "n_flow",
                "n_settlement",

                "timestamp",
                "status",
                "note",
            ])

        w.writerow([
            OPTIMIZER,

            k_perm,
            E,
            nu,
            phi,
            bishop_exponent,

            metric_val,
            LOSS_FUNCTION,

            flow_loss,
            settlement_loss,

            flow_raw,
            settlement_raw,

            flow_scale,
            settlement_scale,

            n_flow,
            n_settlement,

            datetime.now().isoformat(),
            status,
            note,
        ])

    return metric_val
