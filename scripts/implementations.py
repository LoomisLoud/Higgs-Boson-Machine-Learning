# -*- coding: utf-8 -*-
"""
Implementations of all the needed methods to run
the project, including the 6 asked methods (The first 6).
"""
import numpy as np
from helpers import batch_iter

"""
Least squares in gradient descent
"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):

        #computing the gradient and the loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        #update w by gradient
        w = w - gamma * grad

    return w, loss

"""
Least squares in stochastic gradient descent
"""
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1

    w = initial_w
    for n_iter in range(max_iters):

        #compute gradient and loss
        loss = compute_loss(y, tx, w)
        tao = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        y_b, tx_b = next(tao)
        grad = compute_gradient(y_b, tx_b, w)

        #update w by gradient
        w = w - gamma * grad

    return w, loss

"""
Simple Least squares
"""
def least_squares(y, tx):
    w = np.linalg.solve((tx.transpose()).dot(tx), (tx.transpose()).dot(y))
    loss = compute_loss(y, tx, w)

    return w, loss

"""
Ridge regression using matrix computation
"""
def ridge_regression(y, tx, lambda_):
    w = np.linalg.solve((tx.transpose()).dot(tx) + 2 * tx.shape[0] * lambda_ * np.eye((tx.shape[1])), (tx.transpose()).dot(y))
    loss = compute_loss(y, tx, w)

    return w, loss

"""
Logistic regression with gradient descent
"""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(max_iters):
        grad = np.zeros(tx.shape[1])

        sigma = sigmoid(tx.dot(w))
        SX = tx * (sigma - sigma*sigma).reshape(-1,1)
        XSX = tx.transpose().dot(SX)

        for aw in range(tx.shape[0]):
            grad = grad + (-1 / tx.shape[0]) * (y[aw] * tx[aw,:] * sigmoid(-y[aw] * np.dot(tx[aw,:], w)))

        w = w - gamma * np.linalg.solve(XSX, grad)

        if i % 5 == 0 and i != 0:
            gamma = gamma * 0.55

    loss = compute_logistic_loss(y, tx, w)

    return w, loss

"""
The regularized logistic regression using gradient descent
"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(max_iters):
        grad = np.zeros(tx.shape[1])

        sigma = sigmoid(tx.dot(w))
        SX = tx * (sigma - sigma*sigma).reshape(-1,1)
        XSX = tx.transpose().dot(SX) + lambda_*np.eye((tx.shape[1]))

        for aw in range(tx.shape[0]):
            grad = grad + (-1 / tx.shape[0]) * (y[aw] * tx[aw,:] * sigmoid(-y[aw] * np.dot(tx[aw,:],w)))

        w = w - gamma * np.linalg.solve(XSX, grad) - gamma * lambda_*w

        if i % 5 == 0 and i != 0:
            gamma = gamma * 0.55

    loss = compute_logistic_loss(y, tx, w)

    return w, loss

"""
Computes the MSE of the given weights applied to tx
"""
def compute_loss(y, tx, w):
    e = y - np.dot(tx, w)
    mse = np.dot(e.transpose(), e) / (2 * len(tx))

    return mse

"""
Computes the MSE using the sigmoid
"""
def compute_logistic_loss(y, tx, w):
    a = 1 / (2 * y.shape[0])
    b = np.sum(np.square(y - (2 * sigmoid(np.dot(tx, w)) - 1)))

    return a * b

"""
The simple sigmoid function applicable to a scalar
"""
def sig(a):
    if a > 0:
        return 1.0 / (1 + np.exp(-a))
    else:
        return np.exp(a) / (1.0 + np.exp(a))

"""
The vectorized version of sigmoid
"""
sigmoid = np.vectorize(sig)

"""
Computes the gradient with respect to y, tx and w
"""
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)

    return (-1 / y.shape[0]) * tx.transpose().dot(e)
