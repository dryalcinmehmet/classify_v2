"""
Optimize confusion matrix.

For more information, see

* http://cs.stackexchange.com/q/70627/2914
* http://datascience.stackexchange.com/q/17079/8820
"""

import json

from tabulate import tabulate
from sklearn.metrics import silhouette_score

from . import clustering, cm_metrics, io
from .optimize import (
    calculate_score,
    simulated_annealing
)
from .conf import *


def clanaMain(
    cm_: list,
    cm_name: str,
    steps: int,
    labels_ : list,
    save_cm_plot: bool,
    save_score_plot: bool,
    save_hierarchy_labels: bool,
    cm_file: str = None,
    labels_file: str = None

) -> None:
    """
    Run optimization and generate output.
    Optimize and visualize a confusion matrix.

    Parameters
    ---------
    cm: list
    cm_name: str
    steps : int
    labels_: list
    save_cm_plot: bool
    save_score_plot: bool
    save_hierarchy_labels: bool
    cm_file : str
    labels_file : str

    Returns
    -------
    extract_labels : list
    
    """
    
    cm = io.read_confusion_matrix(cm_, cm_file=cm_file)
    perm = io.read_permutation(cm, cm_name, cm_file=cm_file)
    labels = io.read_labels(len(cm), labels_, labels_file=labels_file)
    
    n, m = cm.shape
    if n != m:
        raise ValueError(
            logger.error("Confusion matrix is expected to be square," 
                         " but was %i x %i" % (n, m)
            )
        )
    
    if len(labels) != n:
        logger.error(
            "Confusion matrix is %i x %i, but len(labels) : %i" % (n, n, len(labels))
        )

    # label warnings
    cm_metrics.get_cm_problems(cm, labels)

    # optimize current cm and plot score
    result = simulated_annealing(
                current_cm=cm, 
                current_perm=perm, 
                score=calculate_score, 
                deterministic=True, 
                steps=steps,
                save_score_plot=save_score_plot
    )
    
    # store permutation
    io.ClanaCfg.store_permutation(
        cm=cm,
        cm_name=cm_name,
        cm_file=cm_file, 
        permutation=result.perm, 
        iterations=steps
    )
    
    # get labels
    labels = [labels[i] for i in result.perm]
    
    logger.info(tabulate(list(map(lambda x:[x], labels)),
                        tablefmt='psql', 
                        headers=["#", "Classes:"],
                        showindex=True)
    )
    
    # get accuracy
    acc = cm_metrics.get_accuracy(cm)
    logger.info("## Accuracy: %.2f" % (acc * 100))
    
    # plot clana cm
    # visualize_cm.plot_cm(
    #     result.cm,
    #     zero_diagonal=False,
    #     labels=labels,
    #     save_cm_plot=save_cm_plot
    # )

    if len(cm) < 5:
        logger.warning(
            "You only have %i classes. Clustering for less than 5 classes "
            "should be done manually." % (len(cm))
        )

    # find cluster in cm
    grouping = clustering.extract_clusters(result.cm, labels)
    
    y_pred = [0]
    cluster_i = 0
    for el in grouping:
        if el:
            cluster_i += 1
        y_pred.append(cluster_i)
    logger.info("## Silhouette score: %d" % silhouette_score(cm, y_pred))
    
    # Store grouping as hierarchy
    if save_hierarchy_labels:
        with open(config['visualize']['output_hierarchy_path'], "w") as outfile:
            hierarchy = clustering.apply_grouping(labels, grouping)
            hierarchy_mixed = clustering.remove_single_element_groups(hierarchy)
            str_ = json.dumps(
                hierarchy_mixed,
                indent=4,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            outfile.write(str_)

    # Extract the labels
    extract_labels = []
    for group in clustering.apply_grouping(labels, grouping):
        if len(group) > 1:
            extract_labels.extend(group)
            logger.info(
                tabulate(list(map(lambda x:[x], group)),
                tablefmt='psql', 
                headers=["#", "Extract labels: %i" % len(group)], 
                showindex=True)
            )

    logger.info(
        tabulate(list(map(lambda x:[x], extract_labels)),
        tablefmt='psql', 
        headers=["#", "All extract labels: %i" % len(extract_labels)], 
        showindex=True)
    )
    return extract_labels    

