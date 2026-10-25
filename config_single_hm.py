import os
from pathlib import Path
from skopt.space import Real

PROJECT_DIR = Path(__file__).resolve().parent

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# ─── Input/Output File Paths ────────────────────────────────────────────────
MASSFLOW_KEY        = "MassFlowRate"                         # Field name to extract MassFlowRate from VTU
NODE_ID             = 64                                     # Node ID to extract MassFlowRate from VTU

PRJ_TEMPLATE        = PROJECT_DIR / "single_hm_template.prj" # Path to the template OGS project file to be modified
OGS_BIN             = Path("/home/reza/Opengeosys/build/release/bin/ogs")       # Path to the OGS executable binary

# ─── Optimization Selection ─────────────────────────────────────────────────
# Select which optimizer to use:
# bayesian_optimization                           # Surrogate-Based
# random_search | grid_search                     # Random/Exhaustive Search
# genetic_algorithm                               # Evolutionary Algorithms
# particle_swarm_optimization | grey_wolf_optimizer |
# whale_optimization | ant_colony_optimization    # Swarm Intelligence
# simulated_annealing                             # Physics-Based Metaheuristic
# nelder_mead                                     # Simplex-Based
# memetic_algorithm | pso_with_gradient_descent   # Hybrid Method
OPTIMIZER = os.environ.get("OGS_OPTIMIZER", "simulated_annealing")

# ─── Loss Function Selection ────────────────────────────────────────────────
# Select loss function to guide optimization:
# rmse → Root Mean Squared Error
# mse  → Mean Squared Error
# mae  → Mean Absolute Error
# r2   → Coefficient of Determination
LOSS_FUNCTION = "rmse"

# ─── Output Management ──────────────────────────────────────────────────────
BASE_OUT_ROOT   = PROJECT_DIR / "optimization_runs_single_hm"  # Root folder for optimization output folders
OUT_ROOT        = BASE_OUT_ROOT / OPTIMIZER                                    # Output folder specific to selected optimizer
LOG_CSV         = PROJECT_DIR / f"single_hm_optim_log_{OPTIMIZER}.csv"
# Path to log CSV file recording each trial (parameters, loss, etc.)
REF_CSV         = PROJECT_DIR / "reference_single_hm.csv"  # Reference measurements to be compared with simulations
# Required columns: time, MassFlowRate

# Each evaluation gets a unique run_id/run_dir, recorded in LOG_CSV.
# Plotting resolves that recorded path; rounded parameter names are not identities.

# ─── Parameter Search Space ──────────────────────────────────────────────────
BOUNDS = [
    Real(1e-23, 1e-16, prior="log-uniform", name="permeability"),       # log-uniform sampling for intrinsic permeability [m²]
    Real(1e7,   9e7,   prior="uniform",     name="youngs_modulus")      # uniform sampling for Young's modulus [Pa]
]

N_PARAMS        = len(BOUNDS)                         # Number of calibration parameters
PARAMETER_NAMES = [b.name for b in BOUNDS]             # Parameter names for logging and printing

# ─── Global Random Seed ─────────────────────────────────────────────────────
RANDOM_STATE    = 42               # Ensure reproducibility of stochastic optimizers (Bayesian, GA, PSO, etc.)

# ─── Bayesian Optimization ──────────────────────────────────────────────────
N_CALLS              = 100         # total number of evaluations
N_INITIAL_POINTS     = 20          # random points before GP model starts
ACQ_FUNC             = "EI"        # acquisition function: EI | PI | UCB

# ─── Random Search ──────────────────────────────────────────────────────────
RS_N_CALLS = 100  # Total number of random evaluations to perform

# ─── Grid Search ────────────────────────────────────────────────────────────
GRID_N_CALLS = 100  # Total number of grid points (combinations) to evaluate

# ─── Genetic Algorithm ──────────────────────────────────────────────────────
GA_POP_SIZE = 21    # Number of individuals in the population
GA_N_GEN    = 5     # Number of generations (iterations) to evolve the population
GA_CXPB     = 0.7   # Crossover probability: chance that two individuals will exchange parts
GA_MUTPB    = 0.2   # Mutation probability: chance that an individual will be randomly altered

# ─── Particle Swarm Optimization ────────────────────────────────────────────
PSO_SWARM_SIZE = 5      # Number of particles (candidate solutions) in the swarm
PSO_N_ITERS    = 20     # Number of iterations to update particle positions
PSO_C1         = 1.5    # Cognitive coefficient (influence of personal best)
PSO_C2         = 1.5    # Social coefficient (influence of global best)
PSO_W          = 0.7    # Inertia weight (controls exploration vs. exploitation)

# ─── Grey Wolf Optimizer ────────────────────────────────────────────────────
POP_EPOCHS = 19   # Number of iterations (generations)
POP_SIZE   = 5    # Number of candidate solutions (wolves) in the population

# ─── Whale Optimization Algorithm ───────────────────────────────────────────
WOA_EPOCHS   = 19   # Number of iterations (generations)
WOA_POP_SIZE = 5    # Number of search agents (whales) in the population

# ─── Ant Colony Optimization (ACOR) ─────────────────────────────────────────
ANTOA_EPOCHS   = 10  # Number of iterations (generations)
ANTOA_POP_SIZE = 5   # Number of agents (ants) in the population

# ─── Simulated Annealing ────────────────────────────────────────────────────
SA_N_ITER  = 24     # Number of maximum iterations (cooling cycles)
SA_T0      = 5230   # Initial temperature (higher = more exploration)
SA_ALPHA   = 2e-5   # Restart temperature ratio for SciPy dual_annealing

# Compatibility name used by the optimizer file
SA_RESTART_TEMP_RATIO = SA_ALPHA

# Compatibility variables, only needed if the optimizer imports them
SA_MAXFUN          = 100
SA_VISIT           = 2.62
SA_ACCEPT          = -5.0
SA_NO_LOCAL_SEARCH = True

# ─── Nelder-Mead Simplex ────────────────────────────────────────────────────
NM_MAX_ITER = 52         # Max number of iterations allowed (Each iteration may involve 1–3 OGS simulations)
NM_XTOL     = 1e-4       # Convergence threshold for parameter changes
NM_FTOL     = 1e-4       # Convergence threshold for objective value changes

# ─── Memetic Algorithm (GA + Local Search) ─────────────────────────────────
MEMETIC_POP_SIZE     = 8     # Number of individuals in the GA population
MEMETIC_N_GEN        = 3     # Number of generations to run
MEMETIC_LS_ENABLED   = True  # Enable/disable local search refinement
MEMETIC_LS_METHOD    = "dual_annealing"  # Local search algorithm to use
MEMETIC_LS_FREQ      = 1     # Run local search every N generations
MEMETIC_LS_MAXITER   = 3     # Maximum iterations for local search

# ─── Hybrid PSO + Gradient Descent ──────────────────────────────────────────
HYBRID_PSO_SWARM_SIZE = 5           # Number of particles in PSO swarm
HYBRID_PSO_N_ITERS    = 19          # Number of PSO iterations (global search)
HYBRID_PSO_C1         = 2.0         # PSO cognitive parameter (individual learning)
HYBRID_PSO_C2         = 2.0         # PSO social parameter (group learning)
HYBRID_PSO_W          = 0.5         # PSO inertia weight (exploration vs exploitation)

GD_METHOD             = "L-BFGS-B"  # Local optimizer method (quasi-Newton)
GD_MAX_ITER           = 9          # Max iterations for gradient descent
GD_TOL                = 1e-8        # Gradient tolerance (early stopping threshold)
