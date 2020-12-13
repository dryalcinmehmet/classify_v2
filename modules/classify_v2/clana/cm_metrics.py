"""Metrics and problems for confusion matrix."""

from typing import List
import numpy as np
from tabulate import tabulate

from .conf import *


def get_accuracy(cm: np.ndarray) -> float:
    """
    Get the accuaracy by the confusion matrix cm.

    Parameters
    ----------
    cm : ndarray

    Returns
    -------
    accuracy : float

    Examples
    --------
    >>> import numpy as np
    >>> cm = np.array([[10, 20], [30, 40]])
    >>> get_accuracy(cm)
    0.5
    >>> cm = np.array([[20, 10], [30, 40]])
    >>> get_accuracy(cm)
    0.6
    """
    return float(sum(cm[i][i] for i in range(len(cm)))) / float(cm.sum())


def get_cm_problems(cm:np.ndarray, labels: List[str]) -> None:
    """
    Find problems of a classifier by analzing its confusion matrix.

    Parameters
    ----------
    cm : ndarray
    labels : List[str]
    
    """

    n = len(cm)

    # Find classes which are not present in the dataset
    for i in range(n):
        if sum(cm[i]) == 0:
            logger.info("The class '%s' was not in the dataset." % labels[i])

    # Find classes which are never predicted
    cm = cm.transpose()
    never_predicted = []
    for i in range(n):
        if sum(cm[i]) == 0:
            never_predicted.append(labels[i])
    
    if len(never_predicted) > 0:
        logger.info(tabulate(list(map(lambda x:[x], never_predicted)),
                                tablefmt='psql', 
                                headers=["WARNING - The following classes were never predicted:"], 
                                showindex=False)
        )
