# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    indices = np.random.permutation(x.shape[0])
    training_ratio = int(np.floor(ratio * x.shape[0]))

    x_training = x[indices[0:training_ratio]]
    y_training = y[indices[0:training_ratio]]
    x_testing = x[indices[training_ratio:]]
    y_testing = y[indices[training_ratio:]]

    return [[x_training, y_training], [x_testing, y_testing]]
