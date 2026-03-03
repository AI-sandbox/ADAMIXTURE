![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adamixture.svg)
![PyPI - Version](https://img.shields.io/pypi/v/adamixture)
![PyPI - License](https://img.shields.io/pypi/l/adamixture)
![PyPI - Status](https://img.shields.io/pypi/status/adamixture)
![PyPI - Downloads](https://img.shields.io/pypi/dm/adamixture)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18289231-blue)](https://doi.org/10.5281/zenodo.18289231)
# ADAMIXTURE: Adaptive First-Order Optimization for Biobank-Scale Genetic Clustering

ADAMIXTURE is an unsupervised global ancestry inference method that scales the ADMIXTURE model to biobank-sized datasets. It combines the Expectation–Maximization (EM) framework with the ADAM first-order optimizer, enabling parameter updates after a single EM step. This approach accelerates convergence while maintaining comparable or improved accuracy, substantially reducing runtime on large genotype datasets. For more information, we recommend reading [our pre-print](https://www.biorxiv.org/content/10.64898/2026.02.13.700171v1.abstract).

The software can be invoked via CLI and has a similar interface to ADMIXTURE (_e.g._ the output format is completely interchangeable).

![nadm_mna](assets/logo.png)

## System requirements

### Hardware requirements
The successful usage of this package requires a computer with enough RAM to be able to handle the large datasets the network has been designed to work with. Due to this, we recommend using compute clusters whenever available to avoid memory issues.

### Software requirements

We recommend creating a fresh Python 3.10 virtual environment using `virtualenv` (or `conda`), and then install the package `adamixture` there. As an example, for `virtualenv`, one should launch the following commands:

```console
$ virtualenv --python=python3.9 ~/venv/nadmenv
$ source ~/venv/nadmenv/bin/activate
(nadmenv) $ pip install adamixture
```

## Installation Guide

The package can be easily installed in at most a few minutes using `pip` (make sure to add the `--upgrade` flag if updating the version):

```console
(nadmenv) $ pip install adamixture
```

## Running ADAMIXTURE

To train a model, simply invoke the following commands from the root directory of the project. For more info about all the arguments, please run `adamixture --help`. Note that VCF and BED are supported as of now:

As an example, the following ADMIXTURE call

```console
$ ./admixture snps_data.bed 8 -s 42
```

would be mimicked in ADAMIXTURE by running

```console
$ adamixture --k 8 --data_path snps_data.bed --save_dir SAVE_PATH --init_file INIT_FILE --name snps_data --seed 42
```

Two files will be output to the `SAVE_PATH` directory (the `name` parameter will be used to create the whole filenames):

- A `.P` file, similar to ADMIXTURE.
- A `.Q` file, similar to ADMIXTURE.

Logs are printed to the `stdout` channel by default. If you want to save them to a file, you can use the command `tee` along with a pipe:

```console
$ adamixture --k 8 ... | tee run.log
```

## Multi-K Sweep

Instead of running ADAMIXTURE for a single K, you can automatically sweep over a range of K values using `--min_k` and `--max_k`. The data is loaded once, and each K is trained sequentially:

```console
$ adamixture --min_k 2 --max_k 10 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_data --threads 8
```

This will produce `snps_data.2.P`, `snps_data.2.Q`, `snps_data.3.P`, `snps_data.3.Q`, ..., `snps_data.10.P`, `snps_data.10.Q`.

You can combine `--min_k`/`--max_k` with `--cv` (see below) to run cross-validation for each K automatically.

## Cross-Validation

ADAMIXTURE supports K-fold cross-validation to help select the optimal number of ancestral populations. Enable it with `--cv`:

```console
$ adamixture --k 7 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_data --cv 5 --threads 8
```

This performs 5-fold cross-validation **after** the standard training completes. For each fold:

1. Samples are randomly split into training and test sets.
2. P and Q are refined on the training set using Adam-EM.
3. Q is projected onto the test set (P fixed) using Adam-EM.
4. A deviance residual is computed on the test set.

The output is the mean ± standard deviation of the deviance across folds, saved to a single `.cv` file (e.g. `snps_data.cv`) with one row per K:

```
K	avg	std
7	0.171560	0.000950
```

**Multi-K + Cross-Validation** — to sweep K=2..10 with 5-fold CV at each K:

```console
$ adamixture --min_k 2 --max_k 10 --cv 5 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_data --threads 8
```

All K results are collected into a single `snps_data.cv` file:

```
K	avg	std
2	0.185432	0.001234
3	0.172345	0.000987
...
10	0.169876	0.001456
```

The optimal K is typically the one with the **lowest** cross-validation error.

When running with `--cv`, a plot is automatically saved as `name.cv.png` showing the cross-validation error as a function of K, with error bars (± SD) and the optimal K highlighted in red.
## Other options

- `--lr` (float, default: `0.005`):  
  Learning rate used by the Adam optimizer in the EM updates.

- `--min_lr` (float, default: `1e-6`):  
  Minimum learning rate used by the Adam optimizer in the EM updates.

- `--lr_decay` (float, default: `0.5`):  
  Learning rate decay factor.

- `--beta1` (float, default: `0.80`):  
  Exponential decay rate for the first moment estimates in Adam.

- `--beta2` (float, default: `0.88`):  
  Exponential decay rate for the second moment estimates in Adam.

- `--reg_adam` (float, default: `1e-8`):  
  Numerical stability constant (epsilon) for the Adam optimizer.

- `--seed` (int, default: `42`):  
  Random number generator seed for reproducibility.

- `--k` (int):  
  Number of ancestral populations (clusters) to infer. Required if `--min_k`/`--max_k` are not specified.

- `--min_k` (int):  
  Minimum K for a multi-K sweep (inclusive). Must be used together with `--max_k`.

- `--max_k` (int):  
  Maximum K for a multi-K sweep (inclusive). Must be used together with `--min_k`.

- `--cv` (int, default: `0`):  
  Number of folds for cross-validation. Set to 0 to disable (default). Common values: 5 or 10.

- `--cv_tole` (float, default: `0.1`):  
  Convergence tolerance for cross-validation log-likelihood. Convergence is reached when the absolute change in log-likelihood is below this threshold.

- `--no_freqs` (flag):  
  If set, the P (allele frequencies) matrix is not saved to disk. Only the Q (admixture proportions) file will be written.

- `--max_iter` (int, default: `1500`):  
  Maximum number of Adam-EM iterations.

- `--check` (int, default: `5`):  
  Frequency (in iterations) at which the log-likelihood is evaluated.

- `--max_als` (int, default: `1000`):  
  Maximum number of iterations for the ALS solver.

- `--tole_als` (float, default: `1e-4`):  
  Convergence tolerance for the ALS optimization.

- `--reg_als` (float, default: `1e-5`):  
  Regularization parameter for ALS.

- `--power` (int, default: `5`):  
  Number of power iterations used in randomized SVD (RSVD).

- `--tole_svd` (float, default: `1e-1`):  
  Convergence tolerance for the SVD approximation.

- `--threads` (int, default: `1`):  
  Number of CPU threads used during execution.

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to that effect. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" should contact the authors for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice.

## Cite

When using this software, please cite the following pre-print:

```bibtex
@article{saurina2026adamixture,
  title={ADAMIXTURE: Adaptive First-Order Optimization for Biobank-Scale Genetic Clustering},
  author={Saurina-i-Ricos, Joan and Mas Monserrat, Daniel and Ioannidis, Alexander G.},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.02.13.700171},
  url={https://doi.org/10.64898/2026.02.13.700171}
}
