# Troubleshooting and Tips

## CUDA issues

If you get an error similar to the following when using the GPU:

`OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`

Simply installing `nvcc` using conda or mamba should fix it:

```console
$ conda install -c nvidia nvcc
```

## Biobank-Scale Execution & High K Values

For large-scale datasets (e.g., UK Biobank, All of Us) or high K values, these parameter settings tend to give better convergence:

```console
--patience_adam 5 \
--lr_decay 0.85 \
--lr 0.0075
```
