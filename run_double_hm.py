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

from run_records import best_successful_record, append_best_record, require_fresh_log


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

    require_fresh_log(LOG_CSV)

    # ── Run the selected optimization method ──
    best_params, best_obj = dispatch[OPTIMIZER]()
    # Include every successful objective call, including local-search evaluations.
    selected = best_successful_record(LOG_CSV)
    best_params = [float(selected[name]) for name in PARAMETER_NAMES]
    loss_key = "total_loss" if "total_loss" in selected else "loss"
    best_obj = float(selected[loss_key])
    if loss_key == "loss" and LOSS_FUNCTION == "r2":
        best_obj = -best_obj

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
    append_best_record(LOG_CSV)


# ── Entry point: run main() if this script is executed directly ──
if __name__ == "__main__":
    main()
