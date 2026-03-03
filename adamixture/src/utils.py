import logging
import random
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def plot_cv_results(cv_results: dict, plot_path: Path) -> None:
    """Generate a Nature-style cross-validation error plot.
    
    Args:
        cv_results (dict): {K: {'avg': float, 'std': float}} CV results per K.
        plot_path (Path): Output path for the PNG file.
    """
    k_vals = sorted(cv_results.keys())
    avgs = [cv_results[k]['avg'] for k in k_vals]
    stds = [cv_results[k]['std'] for k in k_vals]
    
    # Find optimal K (lowest error)
    best_idx = int(np.argmin(avgs))
    best_k = k_vals[best_idx]
    best_avg = avgs[best_idx]
    
    # Nature-style plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # Error bars + line (black)
    ax.errorbar(k_vals, avgs, yerr=stds, 
                fmt='o-', color='black', markersize=5, 
                linewidth=1.2, capsize=3, capthick=0.8,
                elinewidth=0.8, markerfacecolor='black',
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=2)
    
    # Highlight best K (red)
    ax.plot(best_k, best_avg, 'o', color='#E03030', markersize=9,
            markeredgecolor='#B02020', markeredgewidth=1.2, zorder=3)
    
    # Annotation for best K
    y_range = max(avgs) - min(avgs) if max(avgs) != min(avgs) else 0.01
    ax.annotate(f'K = {best_k}',
                xy=(best_k, best_avg),
                xytext=(best_k + 0.3, best_avg + y_range * 0.15),
                fontsize=10, fontweight='bold', color='#B02020',
                arrowprops=dict(arrowstyle='-', color='#B02020', lw=0.8),
                ha='left', va='bottom')
    
    ax.set_xlabel('K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-validation error', fontsize=12, fontweight='bold')
    ax.set_xticks(k_vals)
    
    # Light horizontal grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    fig.savefig(str(plot_path), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def read_data(tr_file: str) -> np.ndarray:
    """
    Reads SNP data from a file and applies imputation if specified..

    Args:
        tr_file (str): Path to the SNP data file.
        imputation (str): Type of imputation to apply ('mean' or 'zero').
        tr_pops_f (str, optional): denotes the path containing the main populations file. Defaults to None.

    Returns:
        da.core.Array: A Dask array containing the SNP data.
    """
    snp_reader = SNPReader()
    G = snp_reader.read_data(tr_file)
    log.info(f"    Data contains {G.shape[1]} samples and {G.shape[0]} SNPs.")
   
    return G

def write_outputs(Q: np.ndarray, run_name: str, K: int, out_path: str, P: np.ndarray=None) -> None:
    """
    Save the Q and optional P matrices to specified output files.

    Args:
        Q (numpy.ndarray): Q matrix to be saved.
        run_name (str): Identifier for the run, used in file naming.
        K (int): Number of clusters, included in the file name.
        out_path (str or Path): Directory where the output files should be saved.
        P (numpy.ndarray, optional): P matrix to be saved, if provided. Defaults to None.

    Returns:
        None
    """
    out_path = Path(out_path)
    np.savetxt(out_path/f"{run_name}.{K}.Q", Q, delimiter=' ')
    if P is not None:
        np.savetxt(out_path/f"{run_name}.{K}.P", P, delimiter=' ')
        log.info("    Q and P matrices saved.")
    else:
        log.info("    Q matrix saved.")
    return

def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
