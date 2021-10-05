# -*- coding: utf-8 -*-
"""
Implementations
***************

Collection of machine learning algorithms for the project 1.
"""

import numpy as np


def standardize(tX : np.ndarray) -> np.ndarray:
    """
    Standardizes the columns in x to zero mean and unit variance.

    Parameters
    ----------
    tX : np.ndarray
        Array with the samples as rows and the features as columns.

    Returns
    -------
    tX_std : np.ndarray
        Standardized x with zero feature-means and unit feature-variance.
    """

    tX_std = (tX - np.mean(tX, axis=0)) / np.std(tX, axis=0)

    return tX_std


def least_squares_GD(y, tX, initial_w, max_iters=100, gamma=0.1):
    """
    Gradient descent algorithm for mean squared error (MSE) loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [1] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> initial_w = np.zeros(tX.shape[1], dtype=float)
    >>> max_iters = 100
    >>> gamma = 0.1
    >>> w, loss = least_squares_GD(y, tX, initial_w, max_iters, gamma)

    """

    w = initial_w
    tol = 10-6

    for n_iter in range(max_iters):
        e = y - tX @ w                              #calculate only once for update and mse loss calculation
        
        # Updating w with gradient
        w = w + tX.T @ e * (gamma / len(y))         #only one multiplication scalar-matrix
                                                    #TODO: check if with y too big this creates problems since (gamma/len(y)) could be really small (even under epsilon machine)
        
        # Calculating the new loss and checking if under tolerance
        loss = np.mean(e**2) / 2
        if (loss < tol):
            break
    return loss, w


 def least_squares_SGD(y, tX, initial_w, max_iters=100, gamma=0.1):
    """
    Stochastic gradient descent algorithm for mean squared error (MSE) loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [2] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 8-10, September 23, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> initial_w = np.zeros(tX.shape[1], dtype=float)
    >>> max_iters = 100
    >>> gamma = 0.1
    >>> w, loss = least_squares_GD(y, tX, initial_w, max_iters, gamma)

    """

    w = initial_w
    N = len(y)                                                          #calculated only once

    # Sampling random sequence of indices
    rand_ind = np.random.choice(np.arange(N), max_iters, replace=False)

    for i in range(max_iters):
        y_rand = y[rand_ind[i]]
        tX_rand = tX[rand_ind[i]]
        # Updating weights with scaled negative gradient
        w = w + (gamma * (y_rand - np.inner(tX_rand, w)) / N) * tX_rand         #use np.inner instead of np.dot better only one matrix-vector multiplication
                                                                                #TODO: check if with y too big this creates problems since (gamma/len(y)) could be really small (even under epsilon machine)
                                                                                # maybe also wrong gradient??
        # TODO: find a way to stop the algorithm efficiently
    # Computing loss for the final weights (MSE)
    e = y_rand - tX_rand @ w                                                    #Compute the error e only once
    loss = np.mean(e**2) / 2

    return w, loss