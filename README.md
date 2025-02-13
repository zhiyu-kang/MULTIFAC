# MULTIFAC: Multiple Linked Tensor Factorization

MULTIFAC is a Python-based implementation for multiple linked tensor factorization. It provides functionalities for simulating data, tuning hyperparameters, and solving the factorization problem for multi-way linked data. The code is run via command-line (Bash scripts).

---

## Usage
The main script for running MULTIFAC is **`Script_complete.py`**, which supports three main operations:

1. **Simulation**: Generate synthetic tensor data with specified dimensions, rank, and noise level.
2. **Tuning**: Perform hyperparameter tuning to select the best regularization parameters.
3. **Solving**: Run the MULTIFAC tensor decomposition model.
   
MULTIFAC is executed from the command line using Python. The general command format is:

```bash
python Script_complete.py --action <action> [options]
```

Where:
- `<action>`: The operation to perform (`simulation`, `tuning`, or `solving`).
- `[options]`: Additional parameters for customizing the run.

---

### 1. Generating Simulated Data
To generate simulated tensor data, run:
```bash
python Script_complete.py --action simulation --format python --shape1 10 10 8 --shape2 10 14 5 3 --rank 2 3 3 --snr 3 --task 1
```

Explanation of parameters:
- `--format python`: Specifies that the output format is Python pkl file.
- `--shape1 10 10 8`: Defines the dimensions of the first tensor (10 × 10 × 8).
- `--shape2 10 14 5 3`: Defines the dimensions of the second tensor (10 × 14 × 5 × 3).
- `--rank 2 3 3`: Sets the ranks of underlying structures (2 for shared strucutre, 3 for individual structures of the first and second tensors).
- `--snr 3`: Specifies the signal-to-noise ratio.
- `--task 1`: Task ID for organizing multiple runs.

---

### 2. Tuning Hyperparameters

To tune the regularization hyperparameters, run:
```bash
python Script_complete.py --action tuning --sigma_lower 0.1 --sigma_upper 500 --sigma_num 20 --nfolds 10 --task 1
```

Explanation of parameters:
- `--sigma_lower 0.1`: Lower bound for the tuning parameter grid search.
- `--sigma_upper 500`: Upper bound for the tuning parameter grid search.
- `--sigma_num 20`: Number of values tested in the range.
- `--nfolds 10`: Number of cross-validation folds.
- `--task 1`: Task ID for organizing multiple runs.

---

### 3. Running the Model

To perform the MULTIFAC tensor factorization, run:
```bash
python Script_complete.py --action solving --task 1
```

Explanation of parameters:
- `--task 1`: Specifies the task ID for solving.

---

## Citation

If you use MULTIFAC in your research, please cite the relevant paper:

