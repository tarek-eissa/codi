# Contextual Out-of-Distribution Integration (CODI)

## Description
This repository contains a Python implementation of the CODI method described in the paper entitled "**CODI: Enhancing machine learning-based molecular profiling through contextual out-of-distribution integration**" (https://doi.org/10.1101/2024.06.15.598503).

## Installation
A package is available via PyPI:
```bash
pip install pycodi
```

## Usage
For detailed information on how to initialize the CODI method and configure its parameters, please refer to the `Example usage.ipynb` Jupyter Notebook and the code documentation in the `pycodi.py` file. The basic usage format is as follows:
```python
from pycodi import CODI

# Initialize an instance of CODI with possible sources of characterized variability
codi = CODI(variability_sources=<your_variability_sources>, random_state=<your_random_state>)

# Create synthetic samples based on an input set of training samples X and associated sample labels y
X_gen, y_gen = codi.generate_samples(X=<your_X>, y=<your_y>, seed_strategy=<your_seed_strategy>, n_per_seed=<your_n_per_seed>)

```

## Citation
> T. Eissa, M. Huber, B. Obermayer-Pietsch, B. Linkohr, A. Peters, F. Fleischmann, M. Žigman, CODI: Enhancing machine learning-based molecular profiling through contextual out-of-distribution integration, PNAS Nexus (2024). https://doi.org/10.1093/pnasnexus/pgae449.
