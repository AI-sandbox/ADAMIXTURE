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

## Multi-K Sweep

Instead of running ADAMIXTURE for a single K, you can automatically sweep over a range of K values using `--min_k` and `--max_k`. The data is loaded once, and each K is trained sequentially:

```console
$ adamixture --min_k 2 --max_k 10 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_sweep
```

## Cross-validation

Use `--cv` to estimate the optimal K by masking a fraction of genotype entries and measuring prediction error. → [Full documentation](docs/cross_validation.md)

```console
$ adamixture -k 8 --cv 5 --data_path data.bed --save_dir out/ --name test
```

## Plotting

Native high-quality visualizations with hierarchical population labels (`--labels`, `--labels2`, `--labels3`) and multi-run alignment. → [Full documentation](docs/plotting.md)

```console
$ adamixture -k 8 --data_path data.bed --save_dir out/ --name test --plot pdf 300
```

## Projection Mode

Estimate ancestry proportions for new samples using a pre-trained, fixed P matrix (Q-only optimisation). K is detected automatically from P. → [Full documentation](docs/projection.md)

```console
$ adamixture-project \
    --data_path new_samples.bed \
    --p_path trained_model/results.8.P \
    --save_dir projection_out/ \
    --name projected
```

## Supervised Mode

Anchor the model with known population labels for a subset of samples while estimating Q freely for unlabeled ones. Labels use the same format as `--labels` (population name or `-`). → [Full documentation](docs/supervised.md)

```console
$ adamixture-supervised \
    --data_path all_samples.bed \
    --labels labels.txt \
    --save_dir supervised_out/ \
    --name supervised_run \
    -k 8
```

## Other options

All hyperparameters and flags can be explored with:

```console
$ adamixture --help
```

Key optimizer arguments:

| Argument | Default | Description |
|---|---|---|
| `--lr` | `0.005` | Adam learning rate |
| `--beta1` | `0.80` | Adam β₁ |
| `--beta2` | `0.88` | Adam β₂ |
| `--reg_adam` | `1e-8` | Adam ε (numerical stability) |
| `--lr_decay` | `0.5` | Learning rate decay factor |
| `--min_lr` | `1e-4` | Minimum learning rate |
| `--patience_adam` | `3` | Checks without improvement before decaying lr |
| `--tol_adam` | `0.1` | Convergence tolerance |
| `--max_iter` | `10000` | Maximum Adam-EM iterations |
| `--check` | `5` | Log-likelihood evaluation frequency |
| `-t` | `1` | Number of CPU threads |
| `-s` | `42` | Random seed |
| `--chunk_size` | `4096` | Number of SNPs in chunk operations |


## Troubleshooting and Tips

→ [Full documentation](docs/troubleshooting.md)

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
