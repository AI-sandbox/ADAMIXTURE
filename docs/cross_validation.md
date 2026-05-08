# Cross-Validation

ADAMIXTURE includes an internal cross-validation (CV) procedure to help estimate the most appropriate number of ancestral populations ($K$). To enable it, use the `--cv` flag:

```console
$ adamixture -k 8 --cv 5 --data_path data.bed --save_dir out/ --name test
```

When enabled, a fraction of the genotype entries is masked during training, and the prediction error (cross-validation error) for these masked entries is calculated. The $K$ with the lowest CV error, or the one where the error curve starts to flatten (the "elbow" point), is typically considered the most optimal.

## Usage with a K sweep

Cross-validation is most informative when combined with a multi-K sweep:

```console
$ adamixture --min_k 2 --max_k 12 --cv 5 \
    --data_path data.bed --save_dir out/ --name sweep
```

This trains models for K = 2 to 12, reporting the CV error at each K so you can identify the optimal number of populations.

## Argument

| Argument | Description |
|---|---|
| `--cv` | Number of CV folds. Use `--cv` alone to default to 5-fold, or specify e.g. `--cv 10` |
