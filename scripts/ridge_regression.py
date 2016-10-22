# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    w = np.linalg.solve((tx.T).dot(tx) + lamb*np.identity(tx.shape[1]), (tx.T).dot(y))
    mse = compute_mse(y, tx, w)
    return mse, w
