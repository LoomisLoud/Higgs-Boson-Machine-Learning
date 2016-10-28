# -*- coding: utf-8 -*-
"""
Implementations of all the needed methods to run
the project, including the 6 asked methods.
"""
import numpy as np

"""
The implementation of least_squares_GD
"""
# TODO Make it match the needed function in submission
def least_squares_GD(y, tx, gamma, max_iters):
    initial_w = np.zeros(tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)

        # TODO Explain this line
        gamma = gamma/1.005

        w = w - gamma * grad

        # storing w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses

"""
The least squares method used for least squares gradient descent.
"""
def least_squares(y, tx):
    w = np.linalg.solve((tx.T).dot(tx), (tx.T).dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

"""
The ridge regression method implementation.
"""
def ridge_regression(y, tx, lambda_):
    w = np.linalg.solve((tx.T).dot(tx) + lamb_*np.identity(tx.shape[1]), (tx.T).dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

"""
The logistic regression using gradient descent.
"""
# TODO Improve, and fix the formatting to pass normal tests.
# TODO Talk about if this is in fact irls or logistic. Rename ?
def irls(X, y):
    theta = np.zeros(X.shape[1])
    theta_ = np.inf
    eps = 50000
    for aqua in range (20):
        grad = np.zeros(X.shape[1])
        a = np.dot(X, theta)
        pi = sigmoid(a)
        SX = X * (pi - pi*pi).reshape(-1, 1)
        XSX = np.dot(X.T, SX)

        for aw in range (len(X)):
            grad = grad + (-1 / len(X)) * (y[aw] * X[aw,:] * sigmoid(-y[aw] * np.dot(X[aw,:], theta)))

        theta = theta - eps * np.linalg.solve(XSX, grad)
        print(sum(y==np.sign(np.dot(X, theta))) / len(y))

        if aqua % 5==0 and aqua !=0:
            eps = eps * 0.5
    return theta

"""
The regularized logistic regression using gradient descent.
"""
# TODO Improve, and fix the formatting to pass normal tests.
def reg_irls(X, y):
    theta = np.zeros(X.shape[1])
    theta_ = np.inf
    eps = 100000
    lamda = 10**-8
    for aqua in range (15):
        grad = np.zeros(X.shape[1])
        a = np.dot(X, theta)
        pi = sigmoid(a)
        SX = X * (pi - pi*pi).reshape(-1,1)
        XSX = np.dot(X.T, SX) + lamda * np.eye((len(X[0])))
        for aw in range (len(X)):
            grad = grad + (-1 / len(X)) * (y[aw] * X[aw,:] * logistic(-y[aw] * np.dot(X[aw,:], theta)))

        theta = theta - eps * np.linalg.solve(XSX, grad) - eps * lamda * theta
        print(sum(y==np.sign(np.dot(X, theta))) / len(y))

        if aqua % 5==0 and aqua != 0:
            eps = eps * 0.5
    return theta

"""
Splits the dataset based on the split ratio.
"""
def split_data(x, y, ratio, seed=1):
    # setting the seed
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    training_ratio = int(np.floor(ratio * x.shape[0]))

    x_training = x[indices[0:training_ratio]]
    y_training = y[indices[0:training_ratio]]
    x_testing = x[indices[training_ratio:]]
    y_testing = y[indices[training_ratio:]]
    return x_training, x_testing, y_training, y_testing

"""
Computes the MSE of the given weights applied to tx.
"""
def compute_loss(y, tx, w):
    e = y - np.dot(tx, w)
    mse = np.dot(e.transpose(), e) / (2 * len(tx))
    return mse

"""
The simple sigmois function
"""
def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))

"""
Computes the gradient with respect to y, tx and w.
"""
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    return (-1/y.shape[0])*tx.T.dot(e)
