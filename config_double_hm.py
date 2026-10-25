import os
from pathlib import Path
from skopt.space import Real

PROJECT_DIR = Path(__file__).resolve().parent

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# ─── Input/Output File Paths ────────────────────────────────────────────────
MASSFLOW_KEY        = "MassFlowRate"                         # Field name to extract MassFlowRate from VTU
DISPLACEMENT_KEY    = "displacement"                         # Field name to extract displacement from VTU

BOTTOM_NODE_IDS     = [0, 41, 82]                            # Bottom nodes used for total MassFlowRate
TOP_NODE_ID         = 202                                    # Top-center node used for settlement
BOTTOM_CENTER_NODE_ID = 82                                   # Only for checking/diagnostics

VERTICAL_COMPONENT  = 1                                      # displacement[:, 1] = vertical displacement u_y
SETTLEMENT_POSITIVE_DOWNWARD = True                          # settlement_top = -u_y

VTU_PATTERN         = "*_t_*.vtu"                            # OGS main-domain VTU output pattern
NODE_ID             = TOP_NODE_ID                            # Backward compatibility with old one-node code

PRJ_TEMPLATE        = PROJECT_DIR / "double_hm_template.prj"     # Path to the template OGS project file to be modified
OGS_BIN             = Path(os.environ.get("OGS_BIN", "/home/reza/Opengeosys/build/release/bin/ogs"))       # Path to the OGS executable binary

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
OPTIMIZER = os.environ.get("OGS_OPTIMIZER", "pso_with_gradient_descent")

# ─── Loss Function Selection ────────────────────────────────────────────────
# Select loss function to guide optimization:
# rmse → Root Mean Squared Error
# mse  → Mean Squared Error
# mae  → Mean Absolute Error
# r2   → Coefficient of Determination
LOSS_FUNCTION = "rmse"

# ─── Double-Target Weighting ────────────────────────────────────────────────
# Select weights for the hydraulic and mechanical targets:
# total_loss = w_flow * normalized_MassFlowRate_loss
#            + w_settlement * normalized_settlement_loss
TARGET_WEIGHTS = {
    "MassFlowRate": 0.5,                                      # Hydraulic target weight
    "settlement_top": 0.5                                     # Mechanical target weight
}

# ─── Output Management ──────────────────────────────────────────────────────
BASE_OUT_ROOT   = PROJECT_DIR / "optimization_runs_double_hm"      # Root folder for optimization output folders
OUT_ROOT        = BASE_OUT_ROOT / OPTIMIZER                                        # Output folder specific to selected optimizer
LOG_CSV         = PROJECT_DIR / f"double_hm_optim_log_{OPTIMIZER}.csv"
# Path to log CSV file recording each trial (parameters, target losses, total loss, etc.)
REF_CSV         = PROJECT_DIR / "reference_double_hm.csv"        # Reference measurements to be compared with simulations
# Required columns: time, MassFlowRate, settlement_top

# Each evaluation gets a unique run_id/run_dir, recorded in LOG_CSV.
# Plotting resolves that recorded path; rounded parameter names are not identities.

# ─── Parameter Search Space ──────────────────────────────────────────────────
BOUNDS = [
    Real(1e-14, 1e-11, prior="log-uniform", name="permeability"),       # log-uniform sampling for intrinsic permeability [m²]
    Real(5e5,   5e6,   prior="uniform",     name="youngs_modulus"),     # uniform sampling for Young's modulus [Pa]
    Real(0.20,  0.45,  prior="uniform",     name="poisson_ratio"),      # uniform sampling for Poisson's ratio [-]
    Real(0.20,  0.40,  prior="uniform",     name="porosity"),           # uniform sampling for porosity [-]
    Real(0.5,   3.0,   prior="uniform",     name="bishop_exponent")     # uniform sampling for Bishop exponent [-]
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
SA_N_ITER  = 12     # Number of maximum iterations (cooling cycles)
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
NM_MAX_ITER = 54         # Max number of iterations allowed (Each iteration may involve 1–3 OGS simulations)
NM_XTOL     = 1e-4       # Convergence threshold for parameter changes
NM_FTOL     = 1e-4       # Convergence threshold for objective value changes

# ─── Memetic Algorithm (GA + Local Search) ─────────────────────────────────
MEMETIC_POP_SIZE     = 5     # Number of individuals in the GA population
MEMETIC_N_GEN        = 6     # Number of generations to run
MEMETIC_LS_ENABLED   = True  # Enable/disable local search refinement
MEMETIC_LS_METHOD    = "dual_annealing"  # Local search algorithm to use
MEMETIC_LS_FREQ      = 6     # Run local search every N generations
MEMETIC_LS_MAXITER   = 3     # Maximum iterations for local search

# ─── Hybrid PSO + Gradient Descent ──────────────────────────────────────────
HYBRID_PSO_SWARM_SIZE = 6           # Number of particles in PSO swarm
HYBRID_PSO_N_ITERS    = 6          # Number of PSO iterations (global search)
HYBRID_PSO_C1         = 2.0         # PSO cognitive parameter (individual learning)
HYBRID_PSO_C2         = 2.0         # PSO social parameter (group learning)
HYBRID_PSO_W          = 0.5         # PSO inertia weight (exploration vs exploitation)

GD_METHOD             = "L-BFGS-B"  # Local optimizer method (quasi-Newton)
GD_MAX_ITER           = 1          # Max iterations for gradient descent
GD_TOL                = 1e-8        # Gradient tolerance (early stopping threshold)
