"""
Everything related to IO.

Reading / writing configuration, matrices and permutations.
"""

import os
import hashlib
# import csv
import json
import yaml
from typing import Any, Dict, List, Optional, cast
import numpy as np
from .conf import *


class ClanaCfg:
    """Methods related to clanas configuration and permutations."""

    @classmethod
    def read_clana_perm(cls, perm_file: str) -> Dict[str, Any]:
        """
        Read a .clana config file which contains permutations.

        Parameters
        ----------
        perm_file : str

        Returns
        -------
        perm : Dict[str, Any]
        
        """
        
        if os.path.isfile(perm_file):
            with open(perm_file) as stream:
                perm = yaml.safe_load(stream)
        
        else:
            perm = {"data": {}}
        
        return perm

    @classmethod
    def get_cfg_path_from_cm_path(cls) -> str:
        """
        Get the configuration path from the path of the confusion matrix.

        Returns
        -------
        cfg_path : str
        
        """
        
        return config['visualize']['save_perm_path']

    @classmethod
    def get_perm(cls, cm, cm_name: str, cm_file: str) -> List[int]:
        """
        Get the best permutation found so far for a given cm_file.

        Fallback: list(range(n))

        Parameters
        ----------
        cm : np.ndarray
        cm_name : str
        cm_file : str

        Returns
        -------
        perm : List[int]
        
        """

        perm_file = cls.get_cfg_path_from_cm_path()
        cfg = cls.read_clana_perm(perm_file)
        n = len(cm)
        perm = list(range(n))
        
        if cm_name in cfg["data"]:
            cm_file_md5 = md5(cm, cm_file=cm_file)
            if cm_file_md5 in cfg["data"][cm_name]:
                logger.info(
                    "## Loaded permutation found in %d iterations" % (
                        cfg["data"][cm_name][cm_file_md5]["iterations"])
                )
                perm = cfg["data"][cm_name][cm_file_md5]["permutation"]
        
        return perm

    @classmethod
    def store_permutation(
        cls, cm: np.ndarray, cm_name: str, cm_file: str, permutation: np.ndarray, iterations: int
    ) -> None:
        """
        Store a permutation.

        Parameters
        ----------
        cm : np.ndarray 
        cm_name : str
        cm_file : str
        permutation : np.ndarray
        iterations : int
        
        """

        perm_file = cls.get_cfg_path_from_cm_path()
        
        if os.path.isfile(perm_file):
            cfg = ClanaCfg.read_clana_perm(perm_file)
            logger.info("## Read perm at '%s'" % perm_file)
        
        else:
            cfg = {"data": {}}
        
        if cm_name not in cfg["data"]:
            cfg["data"][cm_name] = {}
        
        cm_file_md5 = md5(cm, cm_file=cm_file)
        
        if cm_file_md5 not in cfg["data"][cm_name]:
            cfg["data"][cm_name][cm_file_md5] = {
                "permutation": permutation.tolist(),
                "iterations": 0,
            }
        
        cfg["data"][cm_name][cm_file_md5]["permutation"] = permutation.tolist()
        cfg["data"][cm_name][cm_file_md5]["iterations"] += iterations
        
        # Write file
        with open(perm_file, "w") as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)
            logger.info("## Save perm at '%s'" % perm_file)


def read_confusion_matrix(cm_: list, cm_file: str, make_max: float = float("inf")) -> np.ndarray:
    """
    Load confusion matrix.

    Parameters
    ----------
    cm_ : list 
    cm_file : str
        Path to a JSON file which contains a confusion matrix (List[List[int]])
    make_max : float, optional (default: +Infinity)
        Crop values at this value.

    Returns
    -------
    cm : np.ndarray
    
    """
    
    if cm_file is not None:
        with open(cm_file) as f:
            cm_ = json.load(f)
    
    cm = np.array(cm_)

    # Crop values
    n = len(cm)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cm[i][j] = cast(int, min(cm[i][j], make_max))

    return cm



def read_permutation(cm: np.ndarray, cm_name: str, cm_file: str) -> List[int]:
    """
    Load permutation.

    Parameters
    ----------
    cm: np.ndarray
    cm_name: str
    cm_file : str

    Returns
    -------
    perm : List[int]
        Permutation of the numbers 0, ..., n-1
    
    """

    # if not os.path.isfile(cm_file):
    #     raise ValueError(
    #         logger.error("%s is not a file" % (cm_file)
    #         )
    #     )

    perm = ClanaCfg.get_perm(cm, cm_name, cm_file)
    return perm


def read_labels(n:int, labels_, labels_file: str) -> List[str]:
    """
    Load labels.

    Parameters
    ----------
    n : int
    labels_: DataFrame
    labels_file : str

    Returns
    -------
    labels : List[str]
    
    """
    
    if labels_file:
        labels = load_labels(labels_file, n)
    
    elif labels_ is not None:
        labels = labels_['labels'].values.tolist()
    
    else:
        labels = [str(el) for el in range(n)]
    
    return labels


def load_labels(labels_file: str, n: int) -> List[str]:
    """
    Load labels from a CSV file.

    Parameters
    ----------
    labels_file : str
    n : int

    Returns
    -------
    labels : List[str]
    
    """
    
    if n < 0:
        raise ValueError(
            logger.error("n= %d needs to be non-negative" % (n)
            )
        )
    
    if os.path.isfile(labels_file):
        # Read CSV file
        with open(labels_file) as fp:
            reader = csv.reader(fp, delimiter=";", quotechar='"')
            next(reader, None)  # skip the headers
            parsed_csv = list(reader)
            labels = [el[0] for el in parsed_csv]  # short by default
    
    else:
        labels = [str(el) for el in range(n)]
    
    return labels


def md5(cm, cm_file) -> str:
    """Compute MD5 hash of a file."""
    
    hash_md5 = hashlib.md5()
    
    if cm_file:
        with open(cm_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(cm)
    else:
        hash_md5.update(cm)
    
    return hash_md5.hexdigest()
