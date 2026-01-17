import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List
from argparse import ArgumentError, ArgumentTypeError
from pathlib import Path
import numpy as np

from . import utils
from ..model.adamixture import train

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, G: np.ndarray) -> None:
    """Wrapper function to start training
    """
    (save_dir, name, seed, lr, beta1, 
    beta2, reg_adam, max_iter, 
    check, max_als, tole_als, power, tole_svd,
    reg_als, lr_decay, min_lr) = (args.save_dir, args.name, int(args.seed), float(args.lr),
                float(args.beta1), float(args.beta2), float(args.reg_adam), int(args.max_iter),
                int(args.check), int(args.max_als), float(args.tole_als), int(args.power), 
                float(args.tole_svd), float(args.reg_als), float(args.lr_decay), float(args.min_lr))
            
    K = int(args.k)
    P, Q = train(G, K, seed, lr, beta1, beta2, reg_adam, max_iter, 
                check, max_als, tole_als, power, tole_svd, reg_als, lr_decay, min_lr)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    utils.write_outputs(Q, name, K, save_dir, P)

    return

def main(args: List[str], t0: float):
    """Training entry point
    """
    try:        
        log.info("")
        log.info(f"    Running on K = {args.k}.")
        log.info("")
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
        # READ DATA:
        G = utils.read_data(args.data_path)

        # TRAIN MODEL:
        fit_model(args, G)
        
        t1 = time.time()
        log.info("")
        log.info(f"    Total elapsed time: {t1-t0:.2f} seconds.")
        log.info("")
        
        logging.shutdown()

    except (ArgumentError, ArgumentTypeError) as e:
        log.error(f"    Error parsing arguments")
        logging.shutdown()
        raise e
        
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        logging.shutdown()

if __name__ == '__main__':
    main(sys.argv[1:])