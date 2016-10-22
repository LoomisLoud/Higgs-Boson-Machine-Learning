# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    e = y - tx.dot(w)
    return (-1/y.shape[0])*tx.T.dot(e)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_epochs):
        losses_in_batch = []
        for my, mtx in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(my, mtx, w)
            loss = compute_loss(my, mtx, w)
            w = w - (gamma * gradient)
            losses_in_batch.append(loss)
            ws.append(w)

        losses.append(np.mean(losses_in_batch))
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=np.mean(losses_in_batch), w0=w[0], w1=w[1]))

    return losses, ws
