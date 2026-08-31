from optimizers_double_hm import (
    bayes_optimize,
    random_search_optimize,
    grid_search_optimize,
    ga_optimize,
    pso_optimize,
    gwo_optimize,
    woa_optimize,
    acor_optimize,
    sa_optimize,
    nelder_mead_optimize,
    memetic_optimize,
    pso_gd_optimize,
)

from config_double_hm import (
    OPTIMIZER,
    LOG_CSV,
    LOSS_FUNCTION,
    BOUNDS,
)

import csv
from datetime import datetime


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER NAMES
# ═════════════════════════════════════════════════════════════════════════════

PARAMETER_NAMES = [
    b.name if getattr(b, "name", None) is not None else f"param_{i}"
    for i, b in enumerate(BOUNDS)
]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Define optimizer dispatch dictionary ──
    # Each key maps to a function that runs a specific optimization algorithm
    dispatch = {
        "bayesian_optimization":        bayes_optimize,
        "random_search":               random_search_optimize,
        "genetic_algorithm":           ga_optimize,
        "particle_swarm_optimization": pso_optimize,
        "simulated_annealing":         sa_optimize,
        "grey_wolf_optimizer":         gwo_optimize,
        "grid_search":                 grid_search_optimize,
        "whale_optimization":          woa_optimize,
        "ant_colony_optimization":     acor_optimize,
        "memetic_algorithm":           memetic_optimize,
        "pso_with_gradient_descent":   pso_gd_optimize,
        "nelder_mead":                 nelder_mead_optimize,
    }

    # ── Error handling if optimizer name is invalid ──
    if OPTIMIZER not in dispatch:
        raise ValueError(f"Unknown optimizer '{OPTIMIZER}'")

    # ── Run the selected optimization method ──
    best_params, best_obj = dispatch[OPTIMIZER]()
    best_params = list(best_params)

    # ── Check number of optimized parameters ──
    if len(best_params) != len(PARAMETER_NAMES):
        raise RuntimeError(
            f"Expected {len(PARAMETER_NAMES)} parameters, "
            f"but got {len(best_params)}: {best_params}"
        )

    # ── Extract best parameters ──
    best_k, best_E, best_nu, best_phi, best_bishop = best_params

    # ── Display best result in terminal ──
    if LOSS_FUNCTION == "r2":
        # For R², negate the stored value because optimization minimized -R²
        print(
            f"→ Best k = {best_k:.2e}, "
            f"E = {best_E:.2e}, "
            f"nu = {best_nu:.3f}, "
            f"phi = {best_phi:.3f}, "
            f"bishop = {best_bishop:.3f} | "
            f"R2 = {-best_obj:.3f}"
        )
    else:
        # For loss functions like RMSE, MSE, MAE
        print(
            f"→ Best k = {best_k:.2e}, "
            f"E = {best_E:.2e}, "
            f"nu = {best_nu:.3f}, "
            f"phi = {best_phi:.3f}, "
            f"bishop = {best_bishop:.3f} | "
            f"{LOSS_FUNCTION.upper()} = {best_obj:.3e}"
        )

    # ── Log final selected best parameters to CSV ──
    with LOG_CSV.open("a", newline="") as f:
        csv.writer(f).writerow([
            OPTIMIZER,                            # Optimization method used

            best_k,                               # Best intrinsic permeability [m²]
            best_E,                               # Best Young's modulus [Pa]
            best_nu,                              # Best Poisson's ratio [-]
            best_phi,                             # Best porosity [-]
            best_bishop,                          # Best Bishop exponent [-]

            -best_obj if LOSS_FUNCTION == "r2" else best_obj,  # Store R² as positive
            LOSS_FUNCTION,                        # Loss metric used

            "",                                   # Flow loss not available for final optimizer row
            "",                                   # Settlement loss not available for final optimizer row

            "",                                   # Flow raw metric not available for final optimizer row
            "",                                   # Settlement raw metric not available for final optimizer row

            "",                                   # Flow scale not available for final optimizer row
            "",                                   # Settlement scale not available for final optimizer row

            "",                                   # Number of flow matches not available for final optimizer row
            "",                                   # Number of settlement matches not available for final optimizer row

            datetime.now().isoformat(),           # Timestamp of result

            "best",                               # Status tag
            "Final selected parameters"           # Description
        ])


# ── Entry point: run main() if this script is executed directly ──
if __name__ == "__main__":
    main()
