# -*- coding: utf-8 -*-
"""Exercise 3.
Least Square
"""

import numpy as np

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve((tx.T).dot(tx), (tx.T).dot(y))
    mse = compute_mse(y, tx, w)
    return mse, w
