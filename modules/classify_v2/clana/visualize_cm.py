""" Plot optimization score and confusion matrix."""

from typing import List
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from .conf import *


def plot_cm(
    cm: np.ndarray,
    zero_diagonal: bool,
    labels: [List[str]],
    save_cm_plot: bool
) -> None:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
    zero_diagonal : bool
    labels : Optional[List[str]]
        If this is not given, then numbers are assigned to the classes
    save: bool 
    output: str
    
    """
    
    # for dark places in visualize plot 
    cm = [[0.001 if y==0 else y for y in x] for x in cm]
    
    n = len(cm)
    if zero_diagonal:
        for i in range(n):
            cm[i][i] = 0
    
    if n > 20:
        size = int(n / 4.0)
    
    else:
        size = 10
    
    fig = plt.figure(figsize=(size, size), dpi=80)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    
    x = list(range(len(cm)))
    plt.xticks(x, labels, rotation=config["visualize"]["xlabels_rotation"])
    y = list(range(len(cm)))
    plt.yticks(y, labels, rotation=config["visualize"]["ylabels_rotation"])
    
    if config["visualize"]["norm"] == "LogNorm":
        norm = LogNorm(vmin=max(1, np.min(cm)), vmax=np.max(cm))
    
    elif config["visualize"]["norm"] is None:
        norm = None
    
    else:
        raise NotImplementedError(
            logger.error("%s is not implemented. " 
                         "Try None or LogNorm" % (config["visualize"]["norm"])
            )
        )
    
    res = ax.imshow(
        np.array(cm),
        cmap=config["visualize"]["colormap"],
        interpolation=config["visualize"]["interpolation"],
        norm=norm,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(res, cax=cax)
    plt.tight_layout()
    
    if save_cm_plot:
        output = config['visualize']['output_cm_path']
        plt.savefig(output)
        logger.info("## Save figure at '%s'" % output)


def plot_opt(
    s: int, 
    cs: int,
) -> None:
    """
    Plot optimization scores.
    
    Parameters
    ----------
    s: int
    cs: int
    output: str
    
    """
    
    plt.clf()
    plt.plot(s, cs)
    plt.xticks(rotation=90)
    plt.title('Clana Optimization Score', fontweight='bold', y=1.01, fontsize=12)
    plt.grid(lw = 0.35)
    plt.ylabel('Score', fontsize='large')
    plt.xlabel('Step',fontsize='large')
    plt.legend(['Current Score'], loc='upper right')
    plt.tight_layout()
    # plt.show()
    output=config['visualize']['output_score_path']
    plt.savefig(output, dpi=200)
    logger.info("## Save opt figure at '%s'" % output)

