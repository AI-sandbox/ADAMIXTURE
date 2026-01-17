import logging
import random
import sys
import numpy as np

from pathlib import Path

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

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
