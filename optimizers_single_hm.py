import numpy as np
import random
from math import log10
from sklearn.model_selection import ParameterGrid
from scipy.optimize import dual_annealing, minimize
from skopt import gp_minimize, dummy_minimize
from deap import base, creator, tools
from pyswarms.single import GlobalBestPSO
import pygad
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy.swarm_based.ACOR import OriginalACOR

from utils_single_hm import objective

from config_single_hm import (
    BOUNDS,
    RANDOM_STATE,
    N_CALLS,
    N_INITIAL_POINTS,
    ACQ_FUNC,
    RS_N_CALLS,
    GRID_N_CALLS,
    GA_POP_SIZE,
    GA_N_GEN,
    GA_CXPB,
    GA_MUTPB,
    PSO_SWARM_SIZE,
    PSO_N_ITERS,
    PSO_C1,
    PSO_C2,
    PSO_W,
    POP_EPOCHS,
    POP_SIZE,
    WOA_EPOCHS,
    WOA_POP_SIZE,
    ANTOA_EPOCHS,
    ANTOA_POP_SIZE,
    SA_N_ITER,
    SA_MAXFUN,
    SA_T0,
    SA_RESTART_TEMP_RATIO,
    SA_VISIT,
    SA_ACCEPT,
    SA_NO_LOCAL_SEARCH,
    NM_MAX_ITER,
    NM_XTOL,
    NM_FTOL,
    MEMETIC_POP_SIZE,
    MEMETIC_N_GEN,
    MEMETIC_LS_ENABLED,
    MEMETIC_LS_METHOD,
    MEMETIC_LS_FREQ,
    MEMETIC_LS_MAXITER,
    HYBRID_PSO_SWARM_SIZE,
    HYBRID_PSO_N_ITERS,
    HYBRID_PSO_C1,
    HYBRID_PSO_C2,
    HYBRID_PSO_W,
    GD_METHOD,
    GD_MAX_ITER,
    GD_TOL,
)
# ═════════════════════════════════════════════════════════════════════════════
# OPTIMISERS
# ═════════════════════════════════════════════════════════════════════════════

# ─── Bayesian Optimization ──────────────────────────────────────────────────
def bayes_optimize():
    # Run Bayesian Optimization using Gaussian Processes with scikit-optimize.

    res = gp_minimize(
        func=objective,                    # The function to minimize
        dimensions=BOUNDS,                 # Search space for 2 parameters
        n_calls=N_CALLS,                   # Total number of function evaluations
        n_initial_points=N_INITIAL_POINTS, # Number of random evaluations before GP model
        acq_func=ACQ_FUNC,                 # Acquisition function
        acq_optimizer="auto",              # Optimizer used for acquisition function
        random_state=RANDOM_STATE          # Seed for reproducibility
    )

    return res.x, res.fun


# ─── Random Search ──────────────────────────────────────────────────────────
def random_search_optimize():
    # Perform Random Search Optimization using scikit-optimize's dummy_minimize.

    res = dummy_minimize(
        func=objective,             # The objective function to minimize
        dimensions=BOUNDS,          # Parameter search space
        n_calls=RS_N_CALLS,         # Number of random trials
        random_state=RANDOM_STATE,  # Seed for reproducibility
        verbose=False
    )

    return res.x, res.fun


# ─── Grid Search ────────────────────────────────────────────────────────────
def grid_search_optimize():
    # Perform Grid Search by evaluating the objective over a grid of parameter combinations.

    total_combinations = GRID_N_CALLS

    # For 2 parameters, use square root for a balanced grid
    n_points = int(np.ceil(np.sqrt(total_combinations)))
    n_points = max(n_points, 2)

    # Generate parameter values
    k_vals = np.logspace(np.log10(BOUNDS[0].low), np.log10(BOUNDS[0].high), num=n_points)
    E_vals = np.linspace(BOUNDS[1].low, BOUNDS[1].high, num=n_points)

    # Create the full grid of parameter combinations
    grid = list(ParameterGrid({
        "permeability": k_vals,
        "youngs_modulus": E_vals,
    }))

    # Subsample the full grid reproducibly if it exceeds the evaluation budget.
    if len(grid) > GRID_N_CALLS:
        rng = np.random.default_rng(RANDOM_STATE)
        selected_indices = rng.choice(len(grid), size=GRID_N_CALLS, replace=False)
        grid = [grid[i] for i in selected_indices]

    best_loss = float("inf")
    best_params = None

    # Evaluate each point in the grid
    for params in grid:
        k_perm = params["permeability"]
        E = params["youngs_modulus"]

        loss = objective([k_perm, E])

        if loss < best_loss:
            best_loss = loss
            best_params = [k_perm, E]

    return best_params, best_loss


# ─── Genetic Algorithm ──────────────────────────────────────────────────────
def ga_optimize():
    # Perform Genetic Algorithm Optimization using DEAP.
    random.seed(RANDOM_STATE)

    # Define lower and upper bounds
    low = [
        log10(BOUNDS[0].low),
        BOUNDS[1].low,
    ]

    high = [
        log10(BOUNDS[0].high),
        BOUNDS[1].high,
    ]

    # Create fitness and individual types if not already defined
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def decode_individual(ind):
        for i in range(len(ind)):
            ind[i] = min(max(ind[i], 0.0), 1.0)

        log_k = low[0] + ind[0] * (high[0] - low[0])
        E = low[1] + ind[1] * (high[1] - low[1])
        return [10**log_k, E]

    # Initialize DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_norm", random.random)

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_norm,
        n=len(low)
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_individual(ind):
        return (objective(decode_individual(ind)),)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create and evaluate initial population
    pop = toolbox.population(n=GA_POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hof = tools.HallOfFame(1)
    hof.update(pop)

    # Run the Genetic Algorithm
    for _ in range(GA_N_GEN):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # Apply crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < GA_MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate modified individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)

    # Select and return the best individual seen during the full GA run.
    best = hof[0]

    best_k, best_E = decode_individual(best)
    best_loss = best.fitness.values[0]

    return [best_k, best_E], best_loss


# ─── Particle Swarm Optimization ────────────────────────────────────────────
def pso_optimize():
    # Perform Particle Swarm Optimization using pyswarms' GlobalBestPSO.
    np.random.seed(RANDOM_STATE)

    low = np.array([
        log10(BOUNDS[0].low),
        BOUNDS[1].low,
    ])

    high = np.array([
        log10(BOUNDS[0].high),
        BOUNDS[1].high,
    ])

    options = {
        'c1': PSO_C1,
        'c2': PSO_C2,
        'w': PSO_W
    }

    def cost_fn(x):
        return np.array([
            objective([10**xi[0], xi[1]]) for xi in x
        ])

    opt = GlobalBestPSO(
        n_particles=PSO_SWARM_SIZE,
        dimensions=2,
        options=options,
        bounds=(low, high)
    )

    best_cost, best_pos = opt.optimize(cost_fn, iters=PSO_N_ITERS)

    best_k = 10**best_pos[0]
    best_E = best_pos[1]

    return [best_k, best_E], best_cost


# ─── Grey Wolf Optimizer ────────────────────────────────────────────────────
def gwo_optimize():
    # Perform optimization using Grey Wolf Optimizer from Mealpy.

    def fitness_func(solution):
        k_perm = 10**solution[0]
        E = solution[1]

        return objective([k_perm, E])

    bounds_var = FloatVar(
        lb=[
            log10(BOUNDS[0].low),
            BOUNDS[1].low,
        ],
        ub=[
            log10(BOUNDS[0].high),
            BOUNDS[1].high,
        ]
    )

    model = OriginalGWO(epoch=POP_EPOCHS, pop_size=POP_SIZE)

    best = model.solve({
        "obj_func": fitness_func,
        "bounds": bounds_var,
        "minmax": "min",
    }, seed=RANDOM_STATE)

    best_k = 10**best.solution[0]
    best_E = best.solution[1]
    best_fun = best.target.fitness

    print(
        f"▶ GWO best k = {best_k:.2e}, E = {best_E:.2e}, "
        f"obj = {best_fun:.3e}"
    )

    return [best_k, best_E], best_fun


# ─── Whale Optimization Algorithm ───────────────────────────────────────────
def woa_optimize():
    # Perform optimization using Whale Optimization Algorithm from Mealpy.

    def fitness_func(solution):
        k_perm = 10**solution[0]
        E = solution[1]

        return objective([k_perm, E])

    bounds_var = FloatVar(
        lb=[
            log10(BOUNDS[0].low),
            BOUNDS[1].low,
        ],
        ub=[
            log10(BOUNDS[0].high),
            BOUNDS[1].high,
        ]
    )

    model = OriginalWOA(epoch=WOA_EPOCHS, pop_size=WOA_POP_SIZE)

    best = model.solve({
        "obj_func": fitness_func,
        "bounds": bounds_var,
        "minmax": "min",
    }, seed=RANDOM_STATE)

    best_k = 10**best.solution[0]
    best_E = best.solution[1]
    best_fun = best.target.fitness

    print(
        f"▶ WOA best k = {best_k:.2e}, E = {best_E:.2e}, "
        f"obj = {best_fun:.3e}"
    )

    return [best_k, best_E], best_fun


# ─── Ant Colony Optimization (ACOR) ─────────────────────────────────────────
def acor_optimize():
    # Perform Ant Colony Optimization for continuous domains using Mealpy.

    def fitness_func(solution):
        k_perm = 10**solution[0]
        E = solution[1]

        return objective([k_perm, E])

    bounds_var = FloatVar(
        lb=[
            log10(BOUNDS[0].low),
            BOUNDS[1].low,
        ],
        ub=[
            log10(BOUNDS[0].high),
            BOUNDS[1].high,
        ]
    )

    model = OriginalACOR(
        epoch=ANTOA_EPOCHS,
        pop_size=ANTOA_POP_SIZE,
        sample_count=9,
        intent_factor=0.5,
        zeta=1.0
    )

    best = model.solve({
        "obj_func": fitness_func,
        "bounds": bounds_var,
        "minmax": "min",
    }, seed=RANDOM_STATE)

    best_k = 10**best.solution[0]
    best_E = best.solution[1]
    best_fun = best.target.fitness

    print(
        f"▶ ACOR best k = {best_k:.2e}, E = {best_E:.2e}, "
        f"obj = {best_fun:.3e}"
    )

    return [best_k, best_E], best_fun


# ─── Simulated Annealing ────────────────────────────────────────────────────
def sa_optimize():
    # Perform Simulated Annealing optimization using SciPy's dual_annealing.

    low_k = log10(BOUNDS[0].low)
    high_k = log10(BOUNDS[0].high)

    bounds = [
        (low_k, high_k),
        (BOUNDS[1].low, BOUNDS[1].high),
    ]

    res = dual_annealing(
        func=lambda x: objective([10**x[0], x[1]]),
        bounds=bounds,
        maxiter=SA_N_ITER,
        maxfun=SA_MAXFUN,
        initial_temp=SA_T0,
        visit=SA_VISIT,
        accept=SA_ACCEPT,
        restart_temp_ratio=SA_RESTART_TEMP_RATIO,
        no_local_search=SA_NO_LOCAL_SEARCH,
        seed=RANDOM_STATE
    )

    best_k = 10**res.x[0]
    best_E = res.x[1]
    best_loss = res.fun

    return [best_k, best_E], best_loss

# ─── Nelder-Mead Simplex ────────────────────────────────────────────────────
def nelder_mead_optimize():

    def wrapped_objective(x):
        k_perm = 10**x[0]
        E = x[1]

        return objective([k_perm, E])

    x0 = [
        (log10(BOUNDS[0].low) + log10(BOUNDS[0].high)) / 2,
        (BOUNDS[1].low + BOUNDS[1].high) / 2,
    ]

    bounds = [
        (log10(BOUNDS[0].low), log10(BOUNDS[0].high)),
        (BOUNDS[1].low, BOUNDS[1].high),
    ]

    result = minimize(
        fun=wrapped_objective,
        x0=x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'maxiter': NM_MAX_ITER,
            'maxfev': 100,
            'xatol': NM_XTOL,
            'fatol': NM_FTOL,
            'disp': True
        }
    )

    best_k = 10**result.x[0]
    best_E = result.x[1]
    best_fun = result.fun

    print(
        f"▶ Nelder-Mead best k = {best_k:.2e}, E = {best_E:.2e}, "
        f"obj = {best_fun:.3e}"
    )

    return [best_k, best_E], best_fun


# ─── Memetic Algorithm Local Search ─────────────────────────────────────────
def local_search(solution, fitness_value):
    """
    Local refinement using SciPy's dual_annealing as a local optimizer.
    Called periodically during the Memetic Algorithm to improve the best solution.
    """
    from scipy.optimize import dual_annealing

    def local_obj(x):
        k_perm = 10**x[0]
        E = x[1]

        return objective([k_perm, E])

    bounds = [
        (log10(BOUNDS[0].low), log10(BOUNDS[0].high)),
        (BOUNDS[1].low, BOUNDS[1].high),
    ]

    if MEMETIC_LS_METHOD == "dual_annealing":
        result = dual_annealing(
            local_obj,
            bounds,
            x0=solution,
            maxiter=MEMETIC_LS_MAXITER,
            seed=RANDOM_STATE
        )
    else:
        result = minimize(
            local_obj,
            solution,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": MEMETIC_LS_MAXITER,
                "disp": False
            }
        )

    return result.x


# ─── Memetic Algorithm (GA + Local Search) ─────────────────────────────────
def memetic_optimize():
    # Perform Memetic Optimization using PyGAD and local search.

    def fitness_func(ga_instance, solution, _):
        k_perm = 10**solution[0]
        E = solution[1]

        return -objective([k_perm, E])

    def on_generation(ga_instance):
        if MEMETIC_LS_ENABLED and ga_instance.generations_completed % MEMETIC_LS_FREQ == 0:
            best_solution, best_fitness, _ = ga_instance.best_solution()
            improved = local_search(best_solution, best_fitness)
            ga_instance.population[0] = improved

    ga_instance = pygad.GA(
        num_generations=MEMETIC_N_GEN,
        num_parents_mating=MEMETIC_POP_SIZE // 2,
        fitness_func=fitness_func,
        sol_per_pop=MEMETIC_POP_SIZE,
        num_genes=2,

        init_range_low=[
            log10(BOUNDS[0].low),
            BOUNDS[1].low,
        ],

        init_range_high=[
            log10(BOUNDS[0].high),
            BOUNDS[1].high,
        ],

        mutation_percent_genes=20,
        mutation_type="random",
        crossover_type="single_point",

        gene_space=[
            {"low": log10(BOUNDS[0].low), "high": log10(BOUNDS[0].high)},
            {"low": BOUNDS[1].low, "high": BOUNDS[1].high},
        ],

        on_generation=on_generation,
        stop_criteria=["saturate_10"],
        random_seed=RANDOM_STATE
    )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()

    best_k = 10**solution[0]
    best_E = solution[1]
    best_fun = -solution_fitness

    print(
        f"▶ Memetic best k = {best_k:.2e}, E = {best_E:.2e}, "
        f"obj = {best_fun:.3e}"
    )

    return [best_k, best_E], best_fun


# ─── Hybrid PSO + Gradient Descent ──────────────────────────────────────────
def pso_gd_optimize():
    np.random.seed(RANDOM_STATE)

    low = np.array([
        log10(BOUNDS[0].low),
        BOUNDS[1].low,
    ])

    high = np.array([
        log10(BOUNDS[0].high),
        BOUNDS[1].high,
    ])

    options = {
        'c1': HYBRID_PSO_C1,
        'c2': HYBRID_PSO_C2,
        'w': HYBRID_PSO_W
    }

    def cost_fn(x):
        return np.array([
            objective([10**xi[0], xi[1]]) for xi in x
        ])

    opt = GlobalBestPSO(
        n_particles=HYBRID_PSO_SWARM_SIZE,
        dimensions=2,
        options=options,
        bounds=(low, high)
    )

    best_cost, best_pos = opt.optimize(cost_fn, iters=HYBRID_PSO_N_ITERS)

    def local_cost(x):
        return objective([10**x[0], x[1]])

    result = minimize(
        local_cost,
        best_pos,
        method=GD_METHOD,
        bounds=[
            (low[0], high[0]),
            (low[1], high[1]),
        ],
        options={
            "maxiter": GD_MAX_ITER,
            "gtol": GD_TOL,
            "disp": False
        }
    )

    best_k = 10**result.x[0]
    best_E = result.x[1]
    best_fun = result.fun

    print(
        f"▶ Hybrid best k = {best_k:.2e}, E = {best_E:.2e}, "
        f"obj = {best_fun:.3e}"
    )

    return [best_k, best_E], best_fun
