# MAGPI: Multifidelity-Augmented Gaussian Process Inputs

Code accompanying the paper **["Multifidelity-augmented Gaussian process inputs for surrogate modeling from scarce data"](https://www.sciencedirect.com/science/article/pii/S0045782526005025)**, published in *Computer Methods in Applied Mechanics and Engineering* (also available as a preprint on [arXiv:2603.22050](https://arxiv.org/abs/2603.22050)).

## Overview

Supervised machine learning methods can learn efficient surrogate models that replace expensive high-fidelity simulations or experiments, making many-query analyses such as optimization, uncertainty quantification, and inference tractable. But when training data must come from an expensive model or experiment, the amount of available data is often small, and surrogates trained on it can be unreliable.

**Multifidelity machine learning** addresses this by supplementing scarce high-fidelity data with abundant, cheaper low-fidelity data (e.g., from simplified physics or coarser grids), aiming for a surrogate that is more accurate than any single low-fidelity model but cheaper than the high-fidelity model itself.

This repository implements **MAGPI (Multifidelity-Augmented Gaussian Process Inputs)**, a new multifidelity training approach for Gaussian process (GP) regression. Rather than modifying the GP kernel to encode fidelity relationships, MAGPI uses low-fidelity model predictions as *additional input features* that augment the input space of each higher-fidelity GP. Each fidelity level's model is trained recursively on the raw inputs plus the predictive means of every lower-fidelity model. This is similar in spirit to cokriging estimators — since it leverages all available low-fidelity models simultaneously — while retaining the computational efficiency of autoregressive approaches like NARGP. Numerical experiments across several test problems show that MAGPI achieves both higher predictive accuracy and lower computational cost than existing state-of-the-art multifidelity GP methods.

## Repository Structure

```
MAGPI/
├── magpi/
│   └── magpi.py          # Core library: MAGPI, Kennedy-O'Hagan, and NARGP multifidelity GP models
├── analytical-example/
│   └── analytical.py     # 3-fidelity analytical toy problem comparing MAGPI vs. baselines
├── flame-speed/
│   ├── lfs.py             # Laminar flame speed surrogate across multiple temperature fidelities
│   ├── plotter.py
│   ├── data/               # Cantera-derived flame speed training data
│   ├── models/              # Saved (pickled) trained models
│   └── results/              # Generated figures
└── velocimetry/
    ├── TrainingNotebook.ipynb  # Multifidelity velocity-field surrogate from RANS/LES CFD data
    ├── util.py
    ├── plotter.py
    ├── data/                    # RANS and multi-resolution LES velocity fields
    ├── models/
    └── results/
```

* **`magpi/magpi.py`** contains the reusable model classes:
  - `MAGPI` — the proposed multifidelity-augmented-input GP regressor
  - `KennedyOHagan` — linear autoregressive (Kennedy & O'Hagan) cokriging baseline
  - `NARGP` — nonlinear autoregressive GP baseline
* **`analytical-example/`**, **`flame-speed/`**, and **`velocimetry/`** are the three numerical case studies from the paper: a synthetic 3-fidelity test function, a laminar flame speed model built from Cantera simulations at multiple temperatures, and a turbulent flow-field surrogate built from RANS and multi-resolution LES simulations, respectively.

## Installation

The core library depends on [`jaxgp`](https://github.com/atticusdrex/jaxgp), a lightweight JAX-based GP regression package by the same author, along with standard scientific Python packages.

```bash
git clone https://github.com/atticusdrex/MAGPI.git
cd MAGPI

git clone https://github.com/atticusdrex/jaxgp.git

pip install jax jaxlib numpy scipy scikit-learn matplotlib
```

`magpi.py` imports everything it needs from `jaxgp` via `from jaxgp.gp import *`, so make sure the `jaxgp` package is importable (e.g., on your `PYTHONPATH`, or installed with `pip install -e ./jaxgp` if it provides a `setup.py`/`pyproject.toml`).

## Quick Start

```python
import sys
sys.path.append("..")
from magpi.magpi import *

# data_dict maps fidelity level (0 = lowest) -> {'X', 'Y', 'noise_var'}
data_dict = {
    0: {'X': X_low,  'Y': Y_low,  'noise_var': 1e-6},
    1: {'X': X_med,  'Y': Y_med,  'noise_var': 1e-6},
    2: {'X': X_high, 'Y': Y_high, 'noise_var': 1e-6},
}

# Build and train a MAGPI model, one fidelity level at a time
model = MAGPI(data_dict, RBF, Linear, max_cond=1e5, epsilon=1e-12)
for level in range(3):
    model.optimize(level, params=['k_param', 'm_param', 'noise_var'],
                    lr=2e-1, epochs=2000)

# Predict at the highest fidelity level
mean, cov = model.predict(Xtest, level=2, full_cov=False)
```

See [`analytical-example/analytical.py`](analytical-example/analytical.py) for a complete worked example that also trains `KennedyOHagan` and `NARGP` baselines for comparison.

## Citation

If you use this code or the MAGPI method in your research, please cite the paper:

```bibtex
@article{rex2026magpi,
  title   = {Multifidelity-augmented {G}aussian process inputs for surrogate modeling from scarce data},
  author  = {Rex, Atticus and Qian, Elizabeth and Peterson, David},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {461},
  pages   = {119229},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.119229},
  url     = {https://www.sciencedirect.com/science/article/pii/S0045782526005025}
}
```

A preprint is also available on arXiv:

```bibtex
@misc{rex2026magpiarxiv,
  title         = {{MAGPI}: Multifidelity-Augmented {G}aussian Process Inputs for Surrogate Modeling from Scarce Data},
  author        = {Rex, Atticus and Qian, Elizabeth and Peterson, David},
  year          = {2026},
  eprint        = {2603.22050},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url           = {https://arxiv.org/abs/2603.22050}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
