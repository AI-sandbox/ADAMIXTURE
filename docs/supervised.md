# Supervised Mode

The `adamixture-supervised` command uses **known population labels** for a subset of samples to anchor the model while estimating ancestry for all individuals. After each Adam-EM update, the Q rows of labeled samples are "snapped back" to a near-one-hot encoding corresponding to their assigned population. This forces the allele-frequency model (P) to be anchored by real reference genotype data.

```console
$ adamixture-supervised \
    --data_path all_samples.bed \
    --labels labels.txt \
    --save_dir supervised_out/ \
    --name supervised_run \
    -k 8
```

## Labels file format

The `--labels` file uses the **same format as population labels for plotting**: one entry per line, one per sample, in the same order as the genotype data.

```text
European
African
Asian
-
European
-
Asian
```

- A population name → labeled sample (Q snapped to that ancestry after each update)
- `-` → unlabeled sample (Q estimated freely)

Population names are mapped to integers automatically in order of first appearance.

## Supervision level

By default, supervision uses `--labels` (level 1). You can change this with `--level`:

| `--level` | File used for supervision |
|---|---|
| `1` *(default)* | `--labels` |
| `2` | `--labels2` |
| `3` | `--labels3` |

This lets you supervise at a coarser or finer grouping without duplicating files — the same label files serve both supervision and plotting.

```console
# Supervise using the coarser level-2 grouping
$ adamixture-supervised --data_path data.bed \
    --labels fine.txt --labels2 coarse.txt \
    --level 2 -k 5 --save_dir out/ --name run
```

## Key arguments

| Argument | Description |
|---|---|
| `--data_path` | Path to genotype data (BED, VCF or PGEN) |
| `--labels` | Labels file (required). Population name or `-` per sample |
| `--level` | Labels level to use for supervision: `1`, `2`, or `3` (default: `1`) |
| `--save_dir` | Output directory |
| `--name` | Run name prefix |
| `-k` / `--k` | Number of ancestral populations K |
| `--device` | Computation device: `cpu`, `cuda`, or `mps` (default: `cpu`) |
| `--no_freqs` | Skip saving the P matrix |

Outputs: a `.Q` file (all samples) and optionally a `.P` file. All `--plot`, `--labels`, `--labels2`, `--labels3`, and `--colors` flags are supported.
