# Automatic Calibration of OpenGeoSys Hydro-Mechanical Models

This repository contains Python scripts for automatic parameter calibration of OpenGeoSys (OGS) hydro-mechanical simulations using several optimization algorithms.

The framework repeatedly modifies selected parameters in an OGS project file, runs the numerical model, extracts the simulated responses from the VTU output files, compares them with reference data, and uses the resulting error as the objective function for optimization.

## Code structure

Two versions of the calibration framework are provided:

### Single-target calibration

Files:

* `config_single_hm.py`
* `utils_single_hm.py`
* `optimizers_single_hm.py`
* `run_single_hm.py`

This version calibrates the model against **MassFlowRate** at a selected node.

The calibrated parameters in the provided example are:

* intrinsic permeability
* Young's modulus

The simulated and reference values are compared only at exactly matching time steps.

### Double-target calibration

Files:

* `config_double_hm.py`
* `utils_double_hm.py`
* `optimizers_double_hm.py`
* `run_double_hm.py`

This version simultaneously considers two model responses:

* **MassFlowRate**, calculated over the specified bottom boundary nodes
* **settlement**, extracted at the specified top node

The provided example calibrates five parameters:

* intrinsic permeability
* Young's modulus
* Poisson's ratio
* porosity
* Bishop exponent

Because mass flow and settlement have different units and magnitudes, their errors are normalized separately before being combined. Their relative importance can be changed using `TARGET_WEIGHTS` in `config_double_hm.py`.

## How to use

Before running the calibration, edit the corresponding `config_*.py` file.

The main settings that normally need to be changed are:

1. Path to the OGS executable (`OGS_BIN`)
2. Path to the OGS template project (`PRJ_TEMPLATE`)
3. Path to the reference CSV file (`REF_CSV`)
4. Output directory
5. Parameter bounds (`BOUNDS`)
6. Optimization method (`OPTIMIZER`)
7. Loss function (`LOSS_FUNCTION`)
8. Optimizer-specific settings such as population size or number of evaluations

The reference CSV for the single-target version must contain:

```text
time, MassFlowRate
```

The reference CSV for the double-target version must contain:

```text
time, MassFlowRate, settlement_top
```

After configuration, run:

```bash
python run_single_hm.py
```

or:

```bash
python run_double_hm.py
```

## Available optimization methods

The framework includes:

* Bayesian optimization
* Random search
* Grid search
* Genetic algorithm
* Particle swarm optimization
* Grey Wolf Optimizer
* Whale Optimization Algorithm
* Ant Colony Optimization
* Simulated annealing
* Nelder-Mead
* Memetic algorithm
* Hybrid PSO with gradient-based local optimization

The desired method is selected through the `OPTIMIZER` variable in the corresponding configuration file.

## Output

Each parameter evaluation is performed in a separate output directory. The tested parameter values and corresponding objective values are recorded in a CSV log file. At the end of the optimization, the best parameter set found by the selected method is reported.

## Requirements

The scripts require a working OpenGeoSys installation and Python packages including NumPy, pandas, SciPy, scikit-learn, scikit-optimize, meshio, DEAP, PySwarms, PyGAD, and MEALPY.

The OGS project template and associated mesh files must be available before running the calibration.
