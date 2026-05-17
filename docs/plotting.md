# Plotting

ADAMIXTURE includes native support for generating high-quality visualizations of ancestry proportions. Inspired by [**pong**](https://github.com/ramachandran-lab/pong), it automatically aligns clusters across runs of the same K using a greedy maximum-overlap algorithm, ensuring that same-colored bars represent the same ancestral components across different subplots.

## Single-run Plotting

To generate a plot automatically after training, use the `--plot` flag:

```console
$ adamixture -k 8 --data_path data.bed --save_dir out/ --name test --plot pdf 300
```

Arguments for `--plot` are optional:
- **Format** (e.g., `pdf`, `png`, `jpg`). Default: `png`.
- **Resolution** (DPI, e.g., `300`). Default: `300`.

### Population label flags

The following flags are available for both `adamixture --plot` and `adamixture-plot`:

- **Population Labels (Level 1)**: Use `--labels` to provide a file with one population name per sample. Samples will be grouped by population and sorted by ancestry within each group.
- **Hierarchical Grouping (Level 2)**: Use `--labels2` to provide a file with one coarser group label per sample (e.g., super-population or region). A bracket annotation tier is drawn under the level-1 tick marks.
- **Hierarchical Grouping (Level 3)**: Use `--labels3` to provide a file with one even coarser label per sample (e.g., continent). A second bracket annotation tier is drawn below the level-2 tier.
- **Custom Colors**: Use `--colors` to provide a file with hex or named colors (one per line). See [Custom Colors File Format](#custom-colors-file-format) below for details.

All three label files must have the same number of lines as there are samples. Levels 2 and 3 are only shown if the corresponding file is provided.

## Sweeps and Multi-K Plotting (`--min_k` and `--max_k`)

When training across a sweep of $K$ values using `--min_k` and `--max_k`, you can choose between two plotting modes:

- **Individual plots per $K$ (`--plot`)**:
  Generates a separate plot file for each $K$ in the sweep (e.g., `name.K.png`), sequentially aligned so that ancestral components keep consistent colors from $K=i$ to $K=i+1$.
  ```console
  $ adamixture --min_k 5 --max_k 8 --data_path data.bed --save_dir out/ --name test --plot png 300
  ```

- **Combined single plot (`--plot_single`)**:
  Generates a **single combined multi-panel plot** containing all $K$ subplots stacked vertically (similar to `adamixture-plot`), fully aligned for easy visualization of ancestry shifts. The resulting file will be named `name.minK_maxK.png`.
  ```console
  $ adamixture --min_k 5 --max_k 8 --data_path data.bed --save_dir out/ --name test --plot_single png 300
  ```

## Multi-run Plotting

For comparing multiple runs or different K values (similar to the `pong` tool) using existing results, use the `adamixture-plot` command.

> [!NOTE]
> This command is a standalone post-processing tool. It **does not retrain** the models; it only visualizes and aligns existing `.Q` matrices provided in a **filemap**.

```console
$ adamixture-plot \
    --filemap project.filemap \
    --labels populations.txt \
    --labels2 regions.txt \
    --labels3 continents.txt \
    --colors my_palette.txt \
    --output comparison.pdf
```

The `--labels`, `--labels2`, and `--labels3` flags accept files with **one label per line, one line per sample**, matching the order of samples in the Q matrices. When multiple levels are provided:
- **`--labels`** (level 1): Finest grouping. Used to draw vertical boundaries and x-axis tick marks between populations.
- **`--labels2`** (level 2): Intermediate grouping (e.g., super-population). Drawn as a bracket annotation tier below the tick marks.
- **`--labels3`** (level 3): Coarsest grouping (e.g., continent). Drawn as a second bracket tier below level 2.

### Filemap Format

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

### Custom Colors File Format

The custom colors file supplied to `--colors` is a simple text file containing one color code per line. Color codes can be specified as:
- **HEX codes**: e.g., `#FF5733`
- **Standard CSS color names**: e.g., `crimson`, `royalblue`, `forestgreen`

The file must contain **at least as many colors as the highest $K$ value** in your run or sweep.

**Example `colors.txt`:**
```text
#1f77b4
#ff7f0e
#2ca02c
#d62728
#9467bd
#8c564b
#e377c2
#7f7f7f
```
