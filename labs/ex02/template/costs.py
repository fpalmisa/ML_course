# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE

    loss = 0
    for n in range(len(y)):
        MSE = y[n] - w[0] - w[1]*tx[n,1]
        loss = loss + np.power(MSE,2)

    loss = loss/(len(y))
    return loss
    # ***************************************************
