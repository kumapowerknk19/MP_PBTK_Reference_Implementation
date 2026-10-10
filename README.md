# MP_PBTK_Reference_Implementation
Title：A macrophage-explicit physiologically based toxicokinetic framework for evaluating long-term human organ burdens of ingested microplastics

This repository provides a cleaned and documented reference implementation
of the macrophage-explicit PBTK framework described in the associated
manuscript.

The repository is intentionally limited to the core computational components
needed to understand, run, and adapt the framework. It is not a verbatim
archive of development-stage scripts and does not include plotting code.

## Contents

```text
pbtk/
  model.py
  calibration.py
  analysis.py
config/
  reference_config.json
input/
  mouse_calibration_targets.csv
README.md
```

### `pbtk/model.py`
Implements the macrophage-explicit mouse and human PBTK models, including:
- vascular free-particle pools,
- vascular monocyte-associated pools,
- tissue macrophage-associated pools,
- monocyte uptake,
- macrophage uptake,
- monocyte-to-macrophage transfer,
- macrophage turnover,
- blood-flow transport,
- repeated oral exposure, and
- mass-conserving propagation between dosing events.

### `pbtk/calibration.py`
Implements the mouse calibration workflow for:
- `k_a`,
- `k_b_prime`, and
- `f_abs`.

The calibration uses differential evolution followed by bounded least-squares
refinement.

### `pbtk/analysis.py`
Provides reusable routines for:
- long-term human simulations,
- one-at-a-time sensitivity analysis,
- local sensitivity/elasticity analysis,
- Latin hypercube parameter sampling and parameter-compensation analysis, and
- macrophage-turnover uncertainty analysis.

### `config/reference_config.json`
Contains the reference physiological parameters, kinetic parameters,
calibration bounds, particle properties, exposure scenarios, and analysis
settings used by the reference implementation.

### `input/mouse_calibration_targets.csv`
Contains the mouse calibration targets and standard errors used in the
reference calibration.

## Requirements

Python 3.10 or later is recommended.

Required packages:

```bash
pip install numpy pandas scipy
```

## Example usage

Run Python from the repository root.

### Reference mouse calibration

```python
from pbtk.model import load_config
from pbtk.calibration import calibrate

config = load_config()
result = calibrate(config=config, seed=42)

print(result["k_a"])
print(result["k_b_prime"])
print(result["f_abs"])
print(result["Q"])
```

Expected reference values are approximately:

```text
k_a       = 3.38e-10 mL cell^-1 h^-1
k_b_prime = 1.04e-3 h^-1
f_abs     = 1.46e-2
Q         = 0.741
```

### Long-term human simulation

```python
from pbtk.model import load_config, reference_kinetics
from pbtk.analysis import human_burden_at_years

config = load_config()
kinetics = reference_kinetics(config)

result = human_burden_at_years(
    config=config,
    kinetics=kinetics,
    particles_per_day=config["exposure_particles_per_day"]["Central"],
    years=[10, 70, 100],
)

print(result)
```

### One-at-a-time sensitivity analysis

```python
from pbtk.model import load_config
from pbtk.analysis import oat_sensitivity

config = load_config()
result = oat_sensitivity(config)
print(result)
```

### Parameter-compensation sampling

For a short demonstration:

```python
from pbtk.model import load_config
from pbtk.analysis import lhs_parameter_sampling

config = load_config()
broad, local, correlations = lhs_parameter_sampling(
    config=config,
    profile="quick",
)

print(correlations)
```

To use the sampling sizes described in the manuscript, replace
`profile="quick"` with:

```python
profile="manuscript"
```

### Macrophage-turnover uncertainty

```python
from pbtk.model import load_config
from pbtk.analysis import turnover_uncertainty

config = load_config()
profile, human = turnover_uncertainty(
    config=config,
    profile="quick",
)

print(profile)
print(human)
```

For the manuscript-scale optimization settings, use:

```python
profile="manuscript"
```

## Adapting the framework

Particle diameter, particle density, oral exposure, kinetic parameters,
macrophage turnover, and other model inputs can be modified through
`reference_config.json` or supplied programmatically.

This functionality is intended to facilitate exploration of alternative
particle and exposure scenarios. However, changing particle properties in the
code does **not** by itself establish biological applicability to another
particle type. Particle-specific gastrointestinal translocation, cellular
uptake, retention, and clearance parameters should be used when appropriate
data are available.

## Scope and reproducibility

The governing equations, model assumptions, parameter definitions, and
analysis procedures are described in the associated manuscript. This repository
provides the corresponding reusable reference implementation of the core
computational workflow.

The public files were reorganized and documented for clarity and reuse and are
not intended to reproduce the authors' complete development history or
development-stage code base. No plotting routines are included.

## Citation

Please cite the associated manuscript when using or adapting this framework.
A versioned repository release and permanent archive identifier can be added
here after publication or archival.
