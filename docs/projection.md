# Projection Mode

The `adamixture-project` command estimates ancestry proportions for a **new set of samples** using a **pre-trained, fixed allele-frequency matrix P**. P is never updated — only Q is optimized. This is useful for placing new individuals onto an existing ancestry model without re-training.

> [!NOTE]
> The target genotype data must have **exactly the same SNPs** (same order, same number of rows) as were used when training the original P matrix.

```console
$ adamixture-project \
    --data_path new_samples.bed \
    --p_path trained_model/results.8.P \
    --save_dir projection_out/ \
    --name projected
```

> [!NOTE]
> K is inferred automatically from the number of columns in the P matrix — no need to pass `-k`.

## Key arguments

| Argument | Description |
|---|---|
| `--data_path` | Path to target genotype data (BED, VCF or PGEN) |
| `--p_path` | Path to pre-trained P matrix (M × K, whitespace-delimited) |
| `--save_dir` | Output directory |
| `--name` | Run name prefix for output files |
| `--device` | Computation device: `cpu`, `cuda`, or `mps` (default: `cpu`) |

The output is a single `.Q` file (one row per sample, K columns). All `--plot`, `--labels`, `--labels2`, `--labels3`, and `--colors` flags work identically to the main `adamixture` command.
