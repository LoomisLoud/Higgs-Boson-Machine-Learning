# -*- coding: utf-8 -*-
"""
This file serves as a helper file to choose a polynomial basis from.
It has not been optimized and contains repetitions. We decided that
focusing on getting a better model was more important, and we invested
as much time as possible on that. It would have been easy to modulate
this code, but was a choice to spend ressources elsewhere for this
project

We have basically 3 types of basis.
The degree one to pretty much the degree that you need, self explanatory
Then we have the second degree with combinations which is to combine
every feature with one another which produces a pretty big basis.
The last one is the third degree basis with combinations of elements
that do not all have the same degree, and only taking the 15 first
features, see the report to know more about that.
"""

import numpy as np

"""
Returns a polynomial basis formed of all degrees until a chosen one.
"""
def build_poly_until_degree(tx, degree):
    d = len(tx[0])
    n = len(tx)

    degrees = range(1,degree + 1)
    degrees_number = len(degrees)
    stdX_Ncols = tx.shape[1]
    number_of_rows = degrees_number * stdX_Ncols

    mat = np.zeros((n, number_of_rows))

    print("Computing from degree 1 to", degree, "without combinations...")
    for i in degrees:
        start_index = (i - 1) * stdX_Ncols
        end_index = start_index + stdX_Ncols
        mat[:,start_index:end_index] = tx**i

    return mat

"""
Returns a polynomial basis formed of only the first degree and
the second and third degree with the aformentioned combinations.
"""
def build_poly_combinations_st(tx):
    d = len(tx[0])
    n = len(tx)

    indices_s_deg = []
    indices_t_deg = []

    print("Creating indices...")
    # Creating indices for subsets of degree 2
    for i in range (d):
        for t in range (i,d):
            indices_s_deg.append([t, i])
    indices_s_deg = np.array(indices_s_deg).T

    # Creating indices for subsets of degree 3
    max_t_degree = 15
    for i in range (max_t_degree):
        for t in range (i,max_t_degree):
            for j in range(t,max_t_degree):
                if not (i == t and i == j):
                    indices_t_deg.append([j, t, i])
    indices_t_deg = np.array(indices_t_deg).T

    stdX_Ncols = tx.shape[1]
    indices_s_Ncols = indices_s_deg.shape[1]
    indices_t_Ncols = indices_t_deg.shape[1]

    number_of_rows = indices_s_Ncols + stdX_Ncols + indices_t_Ncols

    mat = np.zeros((n, number_of_rows))

    print("Computing first degree...")
    # First degree
    mat[:, :stdX_Ncols] = tx

    print("Computing second degree with combinations...")
    # Second degree gotten from indices
    mat[:,stdX_Ncols:stdX_Ncols + indices_s_Ncols] = tx[:, indices_s_deg[0]] * tx[:, indices_s_deg[1]]

    print("Computing third degree with some combinations...")
    # Third degree gotten from indices
    mat[:, number_of_rows - indices_t_Ncols: number_of_rows] = tx[:, indices_t_deg[0]] * tx[:, indices_t_deg[1]] * tx[:, indices_t_deg[2]]

    return mat

"""
Returns a polynomial basis formed of only the first degree and
the second degree with the aformentioned combinations, with also
the third to tenth degrees without combinations.
"""
def build_poly_combinations_s(tx):
    d = len(tx[0])
    n = len(tx)

    indices_s_deg = []

    print("Creating indices...")
    # Creating indices for subsets of degree 2
    for i in range (d):
        for t in range (i,d):
            indices_s_deg.append([t, i])
    indices_s_deg = np.array(indices_s_deg).T

    degrees = range(3,11)
    degrees_number = len(degrees)
    stdX_Ncols = tx.shape[1]
    indices_s_Ncols = indices_s_deg.shape[1]

    number_of_rows = indices_s_Ncols + stdX_Ncols + degrees_number * stdX_Ncols

    mat = np.zeros((n, number_of_rows))

    print("Computing first degree...")
    # First degree
    mat[:, :stdX_Ncols] = tx

    print("Computing second degree with combinations...")
    # Second degree gotten from indices
    mat[:,stdX_Ncols:stdX_Ncols + indices_s_Ncols] = tx[:, indices_s_deg[0]] * tx[:, indices_s_deg[1]]

    print("Computing from degree 3 to 10 without combinations...")
    # Improve 3 to 10 degree
    for i in degrees:
        start_index = indices_s_Ncols + (i - 2) * stdX_Ncols
        end_index = start_index + stdX_Ncols
        mat[:,start_index:end_index] = tx**i

    return mat

"""
Returns a polynomial basis formed of all the degrees and combinations
mentioned in the first comment.
"""
def build_poly_combinations_st_all(tx):
    d = len(tx[0])
    n = len(tx)

    indices_s_deg = []
    indices_t_deg = []

    print("Creating indices...")
    # Creating indices for subsets of degree 2
    for i in range (d):
        for t in range (i,d):
            indices_s_deg.append([t, i])
    indices_s_deg = np.array(indices_s_deg).T

    # Creating indices for subsets of degree 3
    max_t_degree = 15
    for i in range (max_t_degree):
        for t in range (i,max_t_degree):
            for j in range(t,max_t_degree):
                if not (i == t and i == j):
                    indices_t_deg.append([j, t, i])
    indices_t_deg = np.array(indices_t_deg).T

    degrees = range(3,11)
    degrees_number = len(degrees) + 1
    stdX_Ncols = tx.shape[1]
    indices_s_Ncols = indices_s_deg.shape[1]
    indices_t_Ncols = indices_t_deg.shape[1]

    number_of_rows = indices_s_Ncols + degrees_number * stdX_Ncols + indices_t_Ncols

    mat = np.zeros((n, number_of_rows))

    print("Computing first degree...")
    # First degree
    mat[:, :stdX_Ncols] = tx

    print("Computing second degree with combinations...")
    # Second degree gotten from indices
    mat[:,stdX_Ncols:stdX_Ncols + indices_s_Ncols] = tx[:, indices_s_deg[0]] * tx[:, indices_s_deg[1]]

    print("Computing from degree 3 to 10 without combinations...")
    # Improve 3 to 10 degree
    for i in degrees:
        start_index = indices_s_Ncols + (i - 2) * stdX_Ncols
        end_index = start_index + stdX_Ncols
        mat[:,start_index:end_index] = tx**i

    print("Computing third degree with some combinations...")
    # Third degree gotten from indices
    mat[:, number_of_rows - indices_t_Ncols: number_of_rows] = tx[:, indices_t_deg[0]] * tx[:, indices_t_deg[1]] * tx[:, indices_t_deg[2]]

    return mat
