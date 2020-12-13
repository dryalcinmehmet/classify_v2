# clana

`clana` is a toolkit for classifier analysis. One key contribution of clana is
Confusion Matrix Ordering (CMO) as explained in chapter 5 of [Analysis and Optimization of Convolutional Neural Network Architectures](https://arxiv.org/abs/1707.09725). It is a technique
that can be applied to any multi-class classifier and helps to understand which
groups of classes are most similar.

## Usage as a library

```
>>> import numpy as np
>>> arr = np.array([[9, 4, 7, 3, 8, 5, 2, 8, 7, 6],
                    [4, 9, 2, 8, 5, 8, 7, 3, 6, 7],
                    [7, 2, 9, 1, 6, 3, 0, 8, 5, 4],
                    [3, 8, 1, 9, 4, 7, 8, 2, 5, 6],
                    [8, 5, 6, 4, 9, 6, 3, 7, 8, 7],
                    [5, 8, 3, 7, 6, 9, 6, 4, 7, 8],
                    [2, 7, 0, 8, 3, 6, 9, 1, 4, 5],
                    [8, 3, 8, 2, 7, 4, 1, 9, 6, 5],
                    [7, 6, 5, 5, 8, 7, 4, 6, 9, 8],
                    [6, 7, 4, 6, 7, 8, 5, 5, 8, 9]])
>>> from clana.optimize import simulated_annealing
>>> result = simulated_annealing(arr)
>>> result.cm
array([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
       [8, 9, 8, 7, 6, 5, 4, 3, 2, 1],
       [7, 8, 9, 8, 7, 6, 5, 4, 3, 2],
       [6, 7, 8, 9, 8, 7, 6, 5, 4, 3],
       [5, 6, 7, 8, 9, 8, 7, 6, 5, 4],
       [4, 5, 6, 7, 8, 9, 8, 7, 6, 5],
       [3, 4, 5, 6, 7, 8, 9, 8, 7, 6],
       [2, 3, 4, 5, 6, 7, 8, 9, 8, 7],
       [1, 2, 3, 4, 5, 6, 7, 8, 9, 8],
       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
>>> result.perm
array([2, 7, 0, 4, 8, 9, 5, 1, 3, 6])
```

You can visualize the `result.cm` and use the `result.perm` to get your labels
in the same order:

```
# Just some example labels
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
>>> labels = [str(el) for el in range(11)]
>>> np.array(labels)[result.perm]
array(['2', '7', '0', '4', '8', '9', '5', '1', '3', '6'], dtype='<U2')
```
