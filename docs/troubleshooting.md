# Troubleshooting and Tips

## macOS: missing `libomp`

ADAMIXTURE requires OpenMP for parallel processing. On macOS you **must** install `libomp` (e.g., via Homebrew) **before** installing the package, otherwise compilation will fail:

```console
$ brew install libomp
```

## CUDA issues

If you get an error similar to the following when using the GPU:

`OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`

Simply installing `nvcc` using conda or mamba should fix it:

```console
$ conda install -c nvidia nvcc
```

## Trying Adam-EM

The default algorithm is `brqn`, an ADMIXTURE SQP + ZAL quasi-Newton solver with improved SVD+ALS initialization. Adam-EM is still available as an experimental/alternative solver:

```console
$ adamixture \
    --algorithm adamem \
    -k 8 \
    --data_path data.bed \
    --save_dir out/ \
    --name test
```

Adam-EM-specific parameters:

| Argument | Default | Description |
|---|---|---|
| `--lr` | `0.005` | Adam learning rate |
| `--beta1` | `0.80` | Adam beta1 (first moment decay) |
| `--beta2` | `0.88` | Adam beta2 (second moment decay) |
| `--reg_adam` | `1e-8` | Adam epsilon for numerical stability |
| `--lr_decay` | `0.5` | Learning-rate decay factor |
| `--min_lr` | `1e-4` | Minimum learning rate |
| `--patience_adam` | `3` | Checks without improvement before decaying the learning rate |
| `--tol` | `0.1` | Convergence tolerance used by the Adam-EM stopping/check logic |
| `--max_iter` | `10000` | Maximum Adam-EM iterations |
| `--check` | `5` | Log-likelihood evaluation frequency |

## Biobank-Scale Adam-EM & High K Values

If you explicitly run `--algorithm adamem` on large-scale datasets (>100,000 samples, e.g., UK Biobank, All of Us) or high K values, these settings can be useful:

```console
--patience_adam 5 \
--lr_decay 0.85 \
--lr 0.0075
```
