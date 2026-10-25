# A Multi-Algorithm Optimization Framework for Parameter Calibration of Coupled-Process Simulations in OpenGeoSys

A Python framework for optimization-based parameter calibration of [OpenGeoSys (OGS)](https://www.opengeosys.org/) numerical simulations. It connects OGS project files with 12 optimization methods, runs the numerical model for each candidate parameter set, extracts selected outputs, and evaluates their agreement with reference data.

The current implementations are demonstrated using **two hydro-mechanical calibration examples**: a single-target problem with two adjustable parameters and a double-target problem with five adjustable parameters. The overall procedure can be adapted to other OGS processes and calibration targets, but doing so requires model-specific changes to the parameter-editing and output-extraction functions.

## Calibration procedure

For each candidate parameter set, the scripts:

1. Read a configured OGS `.prj` template and replace the selected parameter values.
2. Create a separate directory for the model evaluation and run the OGS executable.
3. Read the requested response variables from the generated `.vtu` files.
4. Compare the simulated values with an existing reference CSV at exactly matching timestamps.
5. Compute an error metric (the objective function supplied to the optimizer) and record the evaluation.
6. Continue until the selected optimization method reaches its configured stopping condition.

The Python code does not generate artificial reference measurements or substitute approximate responses for missing OGS output. Successful evaluations use OGS-generated VTU output; unsuccessful evaluations are assigned a failure penalty and logged. However, the current extraction functions can skip VTU files that cannot be read, and the objective uses only timestamps common to both series. Users should therefore review the recorded outputs and the number of matched timestamps.

## Calibration examples

### 1. Single-target calibration

The single-target example considers the calibration of two hydro-mechanical parameters using one hydraulic response.

**Adjustable parameters:**
- Intrinsic permeability (m²)
- Young's modulus (Pa)

**Calibration target:**
- `MassFlowRate` at a specified mesh-point array index (`NODE_ID`, initially 64).

The simulated mass flow rate is extracted from the OGS VTU output and compared with the reference time series.

The reference data are read from `reference_single_hm.csv`, which requires the columns `time` and `MassFlowRate`.

RMSE, MSE, MAE and R² are supported. The comparison uses exactly matching timestamps without interpolation or time weighting.

The example is referred to as the **borehole benchmark**. The source OGS project, its version and any modifications should be documented with the benchmark files.

### 2. Double-target calibration

The double-target example considers five adjustable hydro-mechanical parameters and two response quantities.

**Adjustable parameters:**
- Intrinsic permeability (m²)
- Young's modulus (Pa)
- Poisson's ratio (-)
- Porosity (-)
- Bishop effective-stress exponent (-)

**Calibration targets:**

1. **Hydraulic response:** Sum of `MassFlowRate` at the configured bottom mesh-point indices (initially `[0, 41, 82]`).
2. **Mechanical response:** Vertical displacement at the configured top mesh-point index (initially `202`), converted to positive-downward settlement when configured.

The reference data are read from `reference_double_hm.csv`, requiring the columns `time`, `MassFlowRate` and `settlement_top`.

The two response errors are calculated separately and normalized to account for their different magnitudes and units. The combined objective function is formed using the configured target weights.

For RMSE and MAE, normalization uses the range of matched reference values, with a fallback scale when necessary. For MSE, the error is divided by the square of this scale. When R² is selected, the weighted negative R² score is minimized.

The default weights are 0.5 for each target.

This example is intended for a **hydro-mechanical drainage-column problem** associated with the Liakopoulos benchmark. The original OGS benchmark documentation is available here:

[OpenGeoSys – Liakopoulos Richards-Mechanics Benchmark](https://www.opengeosys.org/docs/benchmarks/richards-mechanics/liakopoulos/)

The configuration indices are mesh-specific and should be checked when using another OGS model.

## Available optimization algorithms

The framework includes 12 optimization algorithms covering different search strategies.

| Optimization method | Configuration name |
|---|---|
| Bayesian Optimization (BO) | `bayesian_optimization` |
| Random Search (RS) | `random_search` |
| Grid Search (GS) | `grid_search` |
| Genetic Algorithm (GA) | `genetic_algorithm` |
| Particle Swarm Optimization (PSO) | `particle_swarm_optimization` |
| Grey Wolf Optimizer (GWO) | `grey_wolf_optimizer` |
| Whale Optimization Algorithm (WOA) | `whale_optimization` |
| Ant Colony Optimization (ACOR) | `ant_colony_optimization` |
| Simulated Annealing (SA) | `simulated_annealing` |
| Nelder-Mead (NM) | `nelder_mead` |
| Memetic Algorithm (MA) | `memetic_algorithm` |
| Hybrid PSO–Gradient-Based Optimization | `pso_with_gradient_descent` |

Bayesian Optimization uses a Gaussian-process surrogate model to guide parameter selection. Random Search and Grid Search evaluate candidate parameters without relying on an adaptive population.

The population-based methods explore the search space using evolutionary or swarm-inspired search strategies. The Memetic Algorithm combines a genetic algorithm with an additional search stage, while the hybrid PSO method applies L-BFGS-B refinement after particle swarm optimization.

The Simulated Annealing implementation uses SciPy's `dual_annealing` method.

The Grid Search implementation may select a reproducible subset of a larger grid when the total number of combinations exceeds the configured evaluation budget. It is therefore not necessarily exhaustive.

## Repository structure

The main files are organized according to the two calibration examples.

| File | Description |
|---|---|
| `config_single_hm.py` | Configuration of the single-target example |
| `config_double_hm.py` | Configuration of the double-target example |
| `utils_single_hm.py` | OGS execution, output extraction and objective evaluation for the single-target example |
| `utils_double_hm.py` | OGS execution, output extraction and objective evaluation for the double-target example |
| `optimizers_single_hm.py` | Optimization algorithms for the single-target example |
| `optimizers_double_hm.py` | Optimization algorithms for the double-target example |
| `run_single_hm.py` | Main execution script for single-target calibration |
| `run_double_hm.py` | Main execution script for double-target calibration |
| `run_records.py` | Run identification, evaluation logging and best-result selection |
| `requirements.txt` | Required Python dependencies |

The calibration scripts also require OGS project templates, associated mesh files and reference CSV data.

The default filenames are:

- `single_hm_template.prj`
- `double_hm_template.prj`
- `reference_single_hm.csv`
- `reference_double_hm.csv`

These benchmark input files must be supplied separately. The `.prj` files may also reference additional mesh or input files required by OGS.

## Installation

### Requirements

- Python
- OpenGeoSys (OGS)
- NumPy
- Pandas
- SciPy
- scikit-learn
- scikit-optimize
- meshio
- DEAP
- PySwarms
- PyGAD
- Mealpy

Install the Python dependencies using:

```bash
python -m pip install -r requirements.txt
```

OpenGeoSys must be installed separately. See the [official OpenGeoSys website](https://www.opengeosys.org/) for installation instructions.

The package versions are not pinned in the current requirements file. For reproducibility, the Python environment, library versions and OGS version should be recorded.

## Configuration

Before running the calibration, open the corresponding configuration file:

- `config_single_hm.py` for the single-target example.
- `config_double_hm.py` for the double-target example.

The configuration files define the OGS executable path, project template, reference data, optimization algorithm, error metric, parameter bounds and algorithm settings.

The OGS executable path should be adjusted to the local installation.

The selected optimizer is controlled by:

```python
OPTIMIZER = "bayesian_optimization"
```

Other algorithms can be selected using their configuration names from the table above.

The objective metric can be selected using:

```python
LOSS_FUNCTION = "rmse"
```

Supported metrics are `rmse`, `mse`, `mae` and `r2`.

For the double-target example, the relative contributions of the hydraulic and mechanical responses are defined using:

```python
TARGET_WEIGHTS = {
    "MassFlowRate": 0.5,
    "settlement_top": 0.5
}
```

These settings can be adjusted for a specific calibration problem.

The parameter ranges, mesh-point indices, output field names and XML parameter locations are model-specific and must be verified before applying the scripts to another OGS model.

## Running the calibration

For single-target calibration:

```bash
python run_single_hm.py
```

For double-target calibration:

```bash
python run_double_hm.py
```

The optimizer can also be selected using an environment variable.

For example, on Linux:

```bash
OGS_OPTIMIZER=bayesian_optimization python run_single_hm.py
```

or:

```bash
OGS_OPTIMIZER=particle_swarm_optimization python run_double_hm.py
```

The OGS executable can be specified using the `OGS_BIN` environment variable when using the updated configuration files.

## Objective function and reference data

For each candidate parameter set, the optimization framework performs an OGS simulation and extracts the configured response quantities.

The simulated time series is then compared with the reference data using the selected error metric.

The reference values are loaded from external CSV files. They are not generated or modified by the optimization algorithms.

Only timestamps that match exactly between the reference and simulated outputs are used in the comparison. The current scripts do not interpolate, extrapolate or apply time weighting.

For the single-target case, the selected error metric is calculated directly from the matched response values.

For the double-target case, the individual errors are normalized and combined using the specified weights.

A failed simulation or unsuccessful output extraction receives a penalty and is recorded in the evaluation log.

**The framework does not automatically require every reference timestamp to be represented in the simulated output.** Users should check that a sufficient number of matching points is available before interpreting a calibration result.

## Output and logging

Each model evaluation is assigned a unique run identifier and output directory.

The output structure separates evaluations according to the calibration example and selected optimization algorithm.

The evaluation logs contain information such as:

- Optimization algorithm
- Evaluated parameter values
- Calculated objective function value
- Error metric
- Simulation timestamp
- Run identifier and output directory
- Execution status and failure information

For the double-target example, the log also includes the individual hydraulic and mechanical errors, normalization scales and number of matched observations.

The best successful evaluation is selected from the recorded results and written to a separate CSV file.

When rerunning the same optimizer, an existing nonempty evaluation log must be archived, renamed or removed before beginning a new run.

## Limitations and reproducibility

The framework provides a common calibration procedure for applying multiple optimization algorithms to OGS models. However, several considerations should be noted.

**Model-specific implementation:** The current scripts are configured for two hydro-mechanical examples. Applications involving other coupled processes require modifications to the model parameters, output extraction and calibration targets.

**Reference data:** Reference datasets must be supplied independently and their origins documented. The optimization algorithms do not generate reference measurements.

**Timestamp matching:** The objective function uses only exactly matching timestamps. Missing or unreadable simulation outputs may reduce the number of observations included in the comparison.

**Optimization budgets:** The configured iterations, generations and population sizes do not always result in identical numbers of OGS evaluations. Some algorithms include initialization evaluations or additional local-search steps.

**Parameter calibration:** A low objective function value indicates agreement between the simulated and reference responses under the selected metric. It does not necessarily imply a unique or physically correct parameter set.

**Reproducibility:** Results may depend on the OGS version, numerical settings, Python dependencies and random-number initialization. The configured random seed improves repeatability for supported stochastic optimizers but does not guarantee identical results across all environments.

## Citation

If this repository is used in research, please cite the associated publication when its final bibliographic information becomes available.

The underlying OGS benchmarks should also be cited separately where applicable.

For the Liakopoulos benchmark, see:

[OpenGeoSys Liakopoulos benchmark documentation](https://www.opengeosys.org/docs/benchmarks/richards-mechanics/liakopoulos/)

## License

This repository is distributed under the terms provided in the `LICENSE` file.
