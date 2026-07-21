<p align="center">
  <img src="assets/logo.png" alt="ADAMIXTURE logo" width="800">
</p>

<h3 align="center">
  Fast Biobank-Scale Population Genetics Clustering
</h3>

<p align="center">
  <img src="https://img.shields.io/pypi/pyversions/adamixture.svg" alt="Python Version">
  <img src="https://img.shields.io/pypi/v/adamixture" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/l/adamixture" alt="License">
  <img src="https://img.shields.io/pypi/status/adamixture" alt="Status">
  <img src="https://img.shields.io/pypi/dm/adamixture" alt="Downloads">
</p>

---

**ADAMIXTURE** is a fast CPU/GPU implementation of ADMIXTURE for biobank-scale genetic clustering. `.P` and `.Q` outputs remain compatible with ADMIXTURE.

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

To train a model, simply invoke the following commands from the root directory of the project. For more info about all the arguments, please run `adamixture --help`. Note that **BED**, **VCF** and **PGEN** are supported.

Supported input files include:

- PLINK BED: `.bed`, `.bed.gz`, `.bed.zst`, with `.bim`/`.fam` sidecars that may also be plain, `.gz` or `.zst`.
- PLINK PGEN: `.pgen` or `.pgen.zst`, with `.pvar`/`.psam` sidecars that may be plain, `.gz` or `.zst`.
- VCF: `.vcf`, `.vcf.gz` and `.vcf.zst`.

As an example, the following ADMIXTURE call

```console
$ ./admixture snps_data.bed 8 -s 42
```

would be equivalent in ADAMIXTURE by running

```console
$ adamixture -k 8 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_data -s 42
```

By default, the following files will be output to the `SAVE_PATH` directory (the `name` parameter will be used to create the full filenames):

- A `.P` file, similar to ADMIXTURE.
- A `.Q` file, similar to ADMIXTURE.
- A `.png` plot file containing the visualization of the inferred ancestry proportions (Q matrix).

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

For a step-by-step guide on how to run a complete analysis workflow, see the [1000 Genomes Tutorial](tutorials/1000_genomes.ipynb).

## Multi-K Sweep

Instead of running ADAMIXTURE for a single K, you can automatically sweep over a range of K values using `--min_k` and `--max_k`. The data is loaded once, and each K is trained sequentially:

```console
$ adamixture --min_k 2 --max_k 10 --data_path snps_data.bed --save_dir SAVE_PATH --name snps_sweep
```

## Cross-validation

Use `--cv` to estimate the optimal K by masking a fraction of genotype entries and measuring prediction error. → [Full documentation](docs/cross_validation.md)

```console
$ adamixture -k 8 --cv --data_path data.bed --save_dir out/ --name test
```

## Plotting

By default, ADAMIXTURE automatically generates a `png` plot at `300` DPI without needing any additional flags. Powered by [**Clumppling**](https://github.com/PopGenClustering/Clumppling), it automatically aligns clusters across runs and K values. → [Full documentation](docs/plotting.md)

Plots can include hierarchical population labels if you provide the arguments (`--labels`, `--labels2`, `--labels3`).

If you want to customize the format and resolution (e.g., to generate a PDF), use `--plot` (or `--plot_single` for individual per-K plots in multi-K sweeps):

- **Single K runs** (`-k`): Use `--plot`.
  ```console
  $ adamixture -k 8 --data_path data.bed --save_dir out/ --name test --plot pdf 300
  ```

- **Multi-K sweeps** (`--min_k` and `--max_k`): Use `--plot` to configure the combined sweep plot (or `--plot_single` for individual plots per K).
  ```console
  $ adamixture --min_k 2 --max_k 10 --data_path data.bed --save_dir out/ --name test --plot pdf 300
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

Key arguments:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|---|---|---|
| `--init` | `als` | Initialization method: SVD+ALS (`als`) or random EM priming (`em`). |
| `--tol` | `0.1` | Convergence tolerance for log-likelihood changes. |
| `--max_iter` | `10000` | Maximum optimization iterations. |
| `-t` | `1` | Number of CPU threads. |
| `-s` | `42` | Random seed. |
| `--device` | `cpu` | Device to use: `cpu`, `gpu`, or `mps`. |
| `--chunk_size` | `8192` | Number of SNPs in chunk operations. |
| `--n_inits` | `5` | Number of independent initializations (keeps run with best log-likelihood). |
| `--chrom_mode` | `autosomes` | Chromosome filter: `autosomes` keeps autosomes `1 to --autosomes`; `all` keeps every chromosome. |
| `--autosomes` | `22` | Number of autosomes kept when `--chrom_mode autosomes` (equivalent to `--specific_chrom 1 2 ... 22`). |
| `--specific_chrom` | `None` | List of specific chromosomes to analyze when `--chrom_mode autosomes` (overrides `--autosomes`). |
| `--no_freqs` | `False` | Do not save the `.P` allele-frequency matrix. |

## Algorithm note

The ADAMIXTURE [paper](https://doi.org/10.1093/bioinformatics/btag236) introduced Adam-EM as an adaptive first-order optimizer for admixture inference. The package still includes this solver via `--algorithm adamem`.

In the current implementation, the default is `--algorithm brqn`. Empirical benchmarking showed that block relaxation with ZAL quasi-Newton acceleration, when paired with our improved SVD+ALS initialization, reaches high-quality solutions in fewer iterations and better wall-clock time. For that reason, BR-QN is the default solver, while Adam-EM remains available for experimentation and reproducibility. Adam-EM tuning parameters are documented in [Troubleshooting and Tips](docs/troubleshooting.md).


## Troubleshooting and Tips

Common issues and platform notes:
- **macOS**: Install `libomp` via Homebrew (`brew install libomp`) before compiling.
- **Windows**: Pre-compiled wheels are provided on PyPI (`pip install adamixture`). If compiling from source (`pip install -e .`), install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with **Desktop development with C++**.
- **CUDA**: Set `CUDA_HOME` or install `nvcc` via Conda (`conda install -c nvidia nvcc`).

→ [Full documentation](docs/troubleshooting.md)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Cite

When using this software, please cite the following paper:

```bibtex
@article{saurina2026adamixture,
  title={ADAMIXTURE: adaptive first-order optimization for biobank-scale genetic clustering},
  author={Saurina-i-Ricos, Joan and Mas Montserrat, Daniel and Ioannidis, Alexander G.},
  journal={Bioinformatics},
  volume={42},
  number={Supplement\_1},
  pages={btag236},
  year={2026},
  doi={10.1093/bioinformatics/btag236},
  url={https://doi.org/10.1093/bioinformatics/btag236}
}
```
