from config_single_hm import (
    MASSFLOW_KEY,
    PRJ_TEMPLATE,
    OGS_BIN,
    OUT_ROOT,
    NODE_ID,
    REF_CSV,
    LOG_CSV,
    OPTIMIZER,
    LOSS_FUNCTION,
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


# ─── Read mass flow rate from VTU files ─────────────────────────────────────
def read_mass_flow(folder: Path, node: int, key: str):
    """
    Extracts a time series of a specific point data field (e.g., MassFlowRate)
    from multiple VTU files at a specific node index.
    """
    files = sorted(folder.glob("*.vtu"), key=extract_time_from_filename)
    times, vals = [], []
    for f in files:
        try:
            mesh = meshio.read(f)
            t = extract_time_from_filename(f)
            q = float(mesh.point_data[key][node])
            times.append(t)
            vals.append(q)
        except Exception:
            continue

    times = np.asarray(times, dtype=float)
    vals = np.asarray(vals, dtype=float)

    order = np.argsort(times)

    return times[order], vals[order]


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


# ─── Simplest loss: only exact timestamp matches ────────────────────────────
def loss_on_exact_matches(t_ref, y_ref, t_sim, y_sim, method: str) -> float:
    """
    Compute loss using ONLY samples whose timestamps match EXACTLY in both series.
    No interpolation, no time-weighting.
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
            return 0.0

    return compute_loss(y_true, y_pred, method)


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
    run_dir: Path,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    prj = OGSProject(PRJ_TEMPLATE)
    prj.set_permeability(k_perm)
    prj.set_youngs_modulus(E)
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
    2. Extract MassFlowRate at NODE_ID
    3. Compare with reference data at EXACT matching timestamps only
    4. Return metric (minimized, or -R² if maximization)
    """
    k_perm = np.nan
    E = np.nan

    try:
        k_perm, E = [float(v) for v in params]

        run_dir = OUT_ROOT / (
            f"run_k_{k_perm:.2e}"
            f"_E_{E:.2e}"
        )

        run_simulation(k_perm, E, run_dir)

        t_sim, q_sim = read_mass_flow(run_dir, NODE_ID, MASSFLOW_KEY)

        if t_sim.size == 0:
            raise RuntimeError("No simulation data found")

        ref = pd.read_csv(REF_CSV)

        t_ref = ref["time"].to_numpy(dtype=float)
        y_ref = ref[MASSFLOW_KEY].to_numpy(dtype=float)

        metric_val = loss_on_exact_matches(
            t_ref=t_ref,
            y_ref=y_ref,
            t_sim=t_sim,
            y_sim=q_sim,
            method=LOSS_FUNCTION,
        )

        status, note = "ok", ""

    except Exception as e:
        metric_val = -np.inf if LOSS_FUNCTION == "r2" else 1e9
        status, note = "fail", str(e)

    # Ensure minimization: rmse/mse/mae → minimize directly, r2 → maximize via negative
    to_minimize = -metric_val if LOSS_FUNCTION == "r2" else metric_val

    # Log result
    with LOG_CSV.open("a", newline="") as f:
        w = csv.writer(f)

        if f.tell() == 0:
            w.writerow([
                "algorithm",
                "permeability",
                "youngs_modulus",
                "loss",
                "metric",
                "timestamp",
                "status",
                "note",
            ])

        w.writerow([
            OPTIMIZER,
            k_perm,
            E,
            metric_val,
            LOSS_FUNCTION,
            datetime.now().isoformat(),
            status,
            note,
        ])

    return to_minimize
