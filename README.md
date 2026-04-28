<p align="center">
  <img src="assets/logo.png" alt="ADAMIXTURE logo" width="800">
</p>

<h3 align="center">
  Adaptive First-Order Optimization for Biobank-Scale Genetic Clustering
</h3>

<p align="center">
  <img src="https://img.shields.io/pypi/pyversions/adamixture.svg" alt="Python Version">
  <img src="https://img.shields.io/pypi/v/adamixture" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/l/adamixture" alt="License">
  <img src="https://img.shields.io/pypi/status/adamixture" alt="Status">
  <img src="https://img.shields.io/pypi/dm/adamixture" alt="Downloads">
  <a href="https://doi.org/10.5281/zenodo.18289231"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18289231-blue" alt="DOI"></a>
</p>

---

**ADAMIXTURE** is an unsupervised global ancestry inference method that scales the ADMIXTURE model to biobank-sized datasets. It combines the Expectation–Maximization (EM) framework with the Adam first-order optimizer, enabling parameter updates after a single EM step. This approach accelerates convergence while maintaining comparable or improved accuracy, substantially reducing runtime on large genotype datasets. For more information, we recommend reading [our preprint](https://www.biorxiv.org/content/10.64898/2026.02.13.700171).

The software can be invoked via CLI and has a similar interface to ADMIXTURE (_e.g._ the output format is completely interchangeable).

## System requirements

### Hardware requirements
The successful usage of this package requires a computer with enough RAM to be able to handle the large datasets the network has been designed to work with. Due to this, we recommend using compute clusters whenever available to avoid memory issues.

### Software requirements

We recommend creating a fresh Python 3.10+ virtual environment. For a faster installation experience, we highly recommend using [uv](https://github.com/astral-sh/uv).

> [!IMPORTANT]  
> If you plan to use GPU acceleration, ensure that the CUDA toolkit is correctly loaded (e.g., `module load cuda`) **before** starting the installation. This ensures that the dependencies and internal components are correctly configured for your hardware.

As an example, using `uv` (recommended):
```console
$ uv venv --python 3.10
$ source .venv/bin/activate
$ uv pip install adamixture
```


> [!IMPORTANT]
> **macOS Users**: ADAMIXTURE requires OpenMP for parallel processing. You **must** install `libomp` (e.g., via Homebrew) before installing the package, otherwise the compilation will fail:
> ```console
> $ brew install libomp
> ```

## Installation Guide

The package can be easily installed in at most a few minutes using `pip` (make sure to add the `--upgrade` flag if updating the version):

```console
$ pip install adamixture
```

## Running ADAMIXTURE

To train a model, simply invoke the following commands from the root directory of the project. For more info about all the arguments, please run `adamixture --help`. Note that **BED**, **VCF** and **PGEN** are supported:

As an example, the following ADMIXTURE call

```console
$ ./admixture snps_data.bed 8 -s 42
```

would be equivalent in ADAMIXTURE by running

```console
$ adamixture -k 8 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_data -s 42
```

Two files will be output to the `SAVE_PATH` directory (the `name` parameter will be used to create the full filenames):

- A `.P` file, similar to ADMIXTURE.
- A `.Q` file, similar to ADMIXTURE.

Logs are printed to the `stdout` channel by default. If you want to save them to a file, you can use the command `tee` along with a pipe:

```console
$ adamixture -k 8 ... | tee run.log
```

### Running with multi-threading

To run ADAMIXTURE using multiple CPU threads, use the `-t` flag:

```console
$ adamixture -k 8 --data_path data.bed --save_dir out/ --name test -t 8
```

### Running with GPU acceleration

To leverage GPU acceleration (highly recommended for large datasets), use the `--device` flag:

- **NVIDIA GPU (CUDA)**:
  ```console
  $ adamixture -k 8 --data_path data.bed --save_dir out/ --name test --device gpu
  ```
- **macOS Apple Silicon (MPS)**:
  ```console
  $ adamixture -k 8 --data_path data.bed --save_dir out/ --name test --device mps
  ```

> [!TIP]
> **GPU Acceleration**: Using GPUs greatly speeds up processing and is highly recommended for large datasets. You can specify the hardware to use with the `--device` parameter:
> - For NVIDIA GPUs, use `--device gpu` (requires CUDA).
> - For macOS users with Apple Silicon (M1/M2/M3/M4/M5), use `--device mps` to enable Metal Performance Shaders (MPS) acceleration. 
> - Note that biobank-scale datasets are best handled on dedicated CUDA-capable GPUs due to high RAM requirements. 

> [!TIP]
> **Biobank-Scale Execution & High K Values**: For large-scale datasets (e.g., UK Biobank, All of Us) with high K values, we recommend the following parameter settings for optimal convergence and performance:
> ```console
> --patience_adam 5 \
> --lr_decay 0.85 \
> --lr 0.0075
> ```

## Multi-K Sweep

Instead of running ADAMIXTURE for a single K, you can automatically sweep over a range of K values using `--min_k` and `--max_k`. The data is loaded once, and each K is trained sequentially:

```console
$ adamixture --min_k 2 --max_k 10 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_sweep
```

## Cross-validation

**ADAMIXTURE** includes an internal cross-validation (CV) procedure to help estimate the most appropriate number of ancestral populations ($K$). To enable it, use the `--cv` flag:

```console
$ adamixture -k 8 --cv 5 --data_path data.bed --save_dir out/ --name test
```

When enabled, a fraction of the genotype entries is masked during training, and the prediction error (cross-validation error) for these masked entries is calculated. The $K$ with the lowest CV error, or the one where the error curve starts to flatten (the "elbow" point), is typically considered the most optimal.

## Plotting

ADAMIXTURE includes native support for generating high-quality visualizations of ancestry proportions. Plots are optimized for biobank-scale data using rasterization for the data area, ensuring consistent performance and sharp text.

### Single-run Plotting
To generate a plot automatically after training, use the `--plot` flag:

```console
$ adamixture -k 8 --data_path data.bed --save_dir out/ --name test --plot pdf 300
```
Arguments for `--plot` are optional:
- **Format** (e.g., `pdf`, `png`, `jpg`). Default: `png`.
- **Resolution** (DPI, e.g., `300`). Default: `300`.

#### Advanced Plotting Arguments
The following flags are available for both `adamixture --plot` and `adamixture-plot`:
- **Population Labels**: Use `--labels` to provide a file with one population name per sample. Samples will be grouped by population and then sorted by ancestry.
- **Custom Colors**: Use `--colors` to provide a file with hex or named colors (one per line). The file must contain at least as many colors as the highest K value in your result.

### Multi-run Plotting
For comparing multiple runs or different K values (similar to the `pong` tool), use the `adamixture-plot` command. 

> [!NOTE]
> This command is a standalone post-processing tool. It **does not retrain** the models; it only visualizes and aligns existing `.Q` matrices provided in a **filemap**.

```console
$ adamixture-plot --filemap project.filemap --labels populations.txt --colors my_palette.txt --output comparison.pdf
```

#### Filemap Format
A filemap is a three-column, tab-delimited file. Each line describes a single Q matrix:
1. **Unique ID**: Must contain at least one letter and cannot contain `#` or `.`.
2. **K Value**: The number of clusters.
3. **Path**: Path to the `.Q` file (relative to the filemap's directory).

Example `project.filemap`:
```text
RunA_K3    3    results/run1.Q
RunB_K5    5    results/run2.Q
RunC_K5    5    results/run3.Q
```

> [!TIP]
> **Cluster Alignment**: `adamixture-plot` automatically aligns clusters across runs of the same K using a greedy maximum-overlap algorithm, ensuring that same-colored bars represent the same ancestral components across different subplots.
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

- `--patience_adam` (int, default: `2`):  
  Patience for reducing the learning rate in Adam-EM.

- `--tol_adam` (float, default: `0.1`):  
  Tolerance for stopping the Adam-EM algorithm.

- `--data_path` (str, required):  
  Path to the genotype data (BED, VCF or PGEN).

- `--save_dir` (str, required):  
  Directory where the output files will be saved.

- `--name` (str, required):  
  Experiment/model name used as prefix for output files.

- `--device` (str, default: `cpu`):  
  Target hardware for computation. Choices: `cpu`, `gpu` (NVIDIA/CUDA), or `mps` (Apple Metal).

- `-s` (int, default: `42`):  
  Random number generator seed for reproducibility.

- `-k` (int):  
  Number of ancestral populations (clusters) to infer. Required if `--min_k`/`--max_k` are not specified.

- `--min_k` (int):  
  Minimum K for a multi-K sweep (inclusive). Must be used together with `--max_k`.

- `--max_k` (int):  
  Maximum K for a multi-K sweep (inclusive). Must be used together with `--min_k`.

- `--cv` (int, default: `0`):  
  Enable v-fold cross-validation on genotype entries. If specified without a value (e.g., `--cv`), it defaults to 5-fold CV.

- `--plot` (args, optional):
  Generate plot after training. Usage: `--plot [format] [resolution]`. e.g., `--plot pdf 300`.

- `--labels` (str):
  Path to population labels file (one label per line). Used for sorting and grouping in plots.

- `--colors` (str):
  Path to custom colors file (one color per line). Must match K (in `adamixture`) or max K (in `adamixture-plot`).

- `--no_freqs` (flag):  
  If set, the P (allele frequencies) matrix is not saved to disk. Only the Q (admixture proportions) file will be written.

- `--max_iter` (int, default: `1500`):  
  Maximum number of Adam-EM iterations.

- `--check` (int, default: `5`):  
  Frequency (in iterations) at which the log-likelihood is evaluated.

- `--max_als` (int, default: `1000`):  
  Maximum number of iterations for the ALS solver.

- `--tol_als` (float, default: `1e-4`):  
  Convergence tolerance for the ALS optimization.

- `--power` (int, default: `5`):  
  Number of power iterations used in randomized SVD.

- `--tol_svd` (float, default: `1e-1`):  
  Convergence tolerance for the SVD approximation.

- `--chunk_size` (int, default: `4096`):  
  Number of SNPs in chunk operations for SVD.

- `-t` (int, default: `1`):  
  Number of CPU threads used during execution.


## Troubleshooting

### CUDA issues
If you get an error similar to the following when using the GPU:

`OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`

Simply installing `nvcc` using conda or mamba should fix it:

```console
$ conda install -c nvidia nvcc
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Cite

When using this software, please cite the following preprint:

```bibtex
@article{saurina2026adamixture,
  title={ADAMIXTURE: Adaptive First-Order Optimization for Biobank-Scale Genetic Clustering},
  author={Saurina-i-Ricos, Joan and Mas Monserrat, Daniel and Ioannidis, Alexander G.},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.02.13.700171},
  url={https://doi.org/10.64898/2026.02.13.700171}
}
