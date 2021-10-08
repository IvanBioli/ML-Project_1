# -*- coding: utf-8 -*-
"""
Implementations
***************

Collection of functions developed for the machine learning project 1.
"""

import numpy as np


def standardize(tX):
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


def polynomial_basis(tX, degrees):
    """
    Creates a polynomial basis from tX.

    Parameters
    ----------
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    degrees : list
        List with the polynomial degrees that should be used as basis elements.

    Returns
    -------
    tX_polynomial : np.ndarray
        tX in a polynomial basis.
    """

    tX_polynomial = tX

    for deg in degrees:

        # Treating degree zero separately to avoid duplicated columns
        if deg == 0:

            tX_polynomial = np.column_stack((np.ones(tX.shape[0]), tX_polynomial))

        elif deg > 1:

            tX_polynomial = np.column_stack((tX_polynomial, standardize(tX**deg)))

    return tX_polynomial


def least_squares_GD(y, tX, w_init, max_iters=100, gamma=0.1, tol=None):
    """
    Gradient descent algorithm for mean squared error (MSE) loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    w_init : np.ndarray
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.
    tol : float or None, default=None
        Tolerance for the norm of the gradient to stop iteration prematurely.
        No stopping criterion is used if set to 'None'.

    Returns
    -------
    w : np.ndarray
        Vector containing the optimized weights.
    loss : float
        Mean squared error loss function evaluated with the optimized weights.

    References
    ----------
    [1] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w_init = np.array([1])
    >>> w, loss = least_squares_GD(y, tX, w_init)
    >>> print(w, loss)
    [[3.]] 0.0

    """

    # Number of samples
    N = len(y)

    # Converting 1D arrays to 2D arrays
    w = w_init.reshape((len(w_init), 1))
    y = y.reshape((N, 1))

    # Checking if 'tX' is a 1D array, and consequently converting to a 2D array
    if len(tX.shape) == 1:

        tX = tX.reshape((N, 1))

    for iter in range(max_iters):

        # Error vector
        e = y - np.dot(tX, w)

        # Gradient for MSE loss
        grad = - np.dot(tX.T, e) / N

        # Updating weights with negative gradient scaled by 'gamma'
        w = w - gamma * grad

        # Stopping criterion
        if (tol != None):

            if (np.linalg.norm(grad) < tol):

                print("NOTE: Stopping criterion met after iteration ", iter)
                break

    # Computing loss (MSE) for the weights in the final iteration
    loss = np.mean(e**2) / 2

    # Converting weights back to 1D arrays
    w = w.reshape(len(w))

    return w, loss


def least_squares_SGD(y, tX, w_init, max_iters=100, gamma=0.1, seed=None):
    """
    Stochastic gradient descent algorithm for mean squared error (MSE) loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    w_init : np.ndarray
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.
    seed : int or None, default=None
        Seed to be used for the random number generator

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
    TODO

    """

    # TODO: Robustness measures for 1d tX matrices

    # Number of samples
    N = len(y)

    # Converting potentially 1D arrays to 2D arrays
    w = w_init

    # Using the desired seed (if specified)
    if seed != None:

        np.random.seed(seed)

    # Sampling random sequence of indices (with replacement)
    rand_ind = np.random.choice(np.arange(N), max_iters)

    for iter in range(max_iters):

        # Picking a random sample
        y_rand = y[rand_ind[iter]]
        tX_rand = tX[rand_ind[iter]]

        # Random error
        e_rand = y_rand - np.inner(tX_rand, w)

        # Random gradient for MSE loss
        grad_rand = - e_rand * tX_rand / N

        # Updating weights with negative gradient scaled by 'gamma'
        w = w - gamma * grad_rand
        
        # TODO: find a way to stop the algorithm efficiently

    # Computing loss (MSE) for the weights in the final iteration
    e = y - tX @ w
    loss = np.mean(e**2) / 2

    return w, loss


def least_squares(y, tX):
    """
    Exact analytical solution for the weights using the normal equation.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [3] M. Jaggi, and M. E. Khan, "Least Squares", Machine Learning (CS-433),
        p. 7, October 5, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = least_squares(y, tX)
    >>> print(w, loss)
    [[3.]] 0.0

    """

    # Number of samples
    N = len(y)

    # Converting a potential 1D array to a 2D array
    y = y.reshape((N, 1))

    # Checking if 'tX' is a 1D array, and consequently converting to a 2D array
    if len(tX.shape) == 1:

        tX = tX.reshape((N, 1))

    # Solving for the exact weights according to the normal equation in [3]
    w = np.linalg.solve(np.dot(tX.T, tX), np.dot(tX.T, y))

    # Computing the error
    e = y - np.dot(tX, w)

    # Computing loss (MSE)
    loss = np.mean(e**2) / 2

    # Converting weights back to 1D arrays
    w = w.reshape(len(w))

    return w, loss


def ridge_regression(y, tX, lambda_=0.1):
    """
    Exact analytical solution for the weights using the ridge-regularized
    normal equations.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    lambda_ : float, default=0.1
        Regularization parameter.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [4] M. Jaggi, and M. E. Khan, "Regularization: Ridge Regression and Lasso",
        Machine Learning (CS-433), p. 3, October 1, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = ridge_regression(y, tX)
    >>> print(w, loss)
    [[2.92207792]] 0.02276943835385406

    """

    # Number of samples
    N = len(y)

    # Converting a potential 1D array to a 2D array
    y = y.reshape((N, 1))

    # Checking if 'tX' is a 1D array, and consequently converting to a 2D array
    if len(tX.shape) == 1:

        tX = tX.reshape((N, 1))

    # Creating "penalty"-term for the normal equations
    penalty = lambda_ * 2 * N * np.identity(tX.shape[1])

    # Solving for the exact weights according to the normal equation in [4]
    w = np.linalg.solve(np.dot(tX.T, tX) + penalty, np.dot(tX.T, y))

    # Computing the error
    e = y - np.dot(tX, w)

    # Computing loss (MSE)
    loss = np.mean(e**2) / 2

    # Converting weights back to 1D arrays
    w = w.reshape(len(w))

    return w, loss


def logistic_regression(y, tX, w_init, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with logistic loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    w_init : np.ndarray
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
    [5] M. Jaggi, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 2-12, October XX, 2021.

    Notes
    -----
    Overflow guards 'exp_guard' and 'log_guard' are there to ensure that
    np.exp(x) won't overflow and that np.log(x) won't be passed any non-positive
    values respectively. That's because for

        x > 710 => exp(x) > 1.8e+308
        x < -710 => exp(-x) > 1.8e+308

    we would observe overflows, as Python's maximum float value is

        np.finfo('d').max = 1.7976931348623157e+308

    Usage
    -----
    TODO

    """

    # Number of samples
    N = len(y)

    # Converting potentially 1D arrays to 2D arrays
    w = w_init.reshape((len(w_init), 1))
    y = y.reshape((N, 1))

    # Checking if 'tX' is a 1D array, and consequently converting to a 2D array
    if len(tX.shape) == 1:

        tX = tX.reshape((N, 1))

    # Defining overflow-guards for np.exp() and np.log() (see Notes above)
    exp_guard = lambda x : np.clip(x, -709, 709)
    log_guard = lambda x : np.maximum(x, 1e-20)
 
    for iter in range(max_iters):

        # Evaluating the sigmoid function
        sigma = 1 / (1 + np.exp( - exp_guard(np.dot(tX, w))))

        # Gradient for MSE loss
        grad = - np.dot(tX.T, sigma - y) / N

        # Updating weights with scaled negative gradient
        w = w - gamma * grad

    # Computing loss for the weights of the final iteration
    sigma = 1 / (1 + np.exp( - exp_guard(np.dot(tX, w))))

    # Computing log-loss for the weights in the final iteration
    loss = (np.dot(y.T, np.log(log_guard(sigma))) + 
        np.dot((1 - y).T, np.log(log_guard(1 - np.exp(exp_guard(sigma))))))

    # Converting weights back to 1D arrays
    w = w.reshape(len(w))

    return w, loss


def reg_logistic_regression(y, tX, lambda_, w_init, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with (ridge) regularized logistic loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    lambda_ : float
        Regularization parameter.
    w_init : np.ndarray
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
    [6] M. Jaggi, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 16-17, October XX, 2021.

    Usage
    -----
    TODO

    """

    # Number of samples
    N = len(y)

    # Converting potentially 1D arrays to 2D arrays
    w = w_init.reshape((len(w_init), 1))
    y = y.reshape((N, 1))

    # Checking if 'tX' is a 1D array, and consequently converting to a 2D array
    if len(tX.shape) == 1:

        tX = tX.reshape((N, 1))

    # Defining overflow-guards for np.exp() and np.log() (see Notes above)
    exp_guard = lambda x : np.clip(x, -709, 709)
    log_guard = lambda x : np.maximum(x, 1e-20)
 
    for iter in range(max_iters):

        # Evaluating the sigmoid function
        sigma = 1 / (1 + np.exp( - exp_guard(np.dot(tX, w))))

        # Gradient for MSE loss
        grad = - np.dot(tX.T, sigma - y) / N

        # Calculating "penalty"-term from regularization
        penalty = lambda_ * w

        # Updating weights with scaled negative gradient and penalty term [5]
        w = w - gamma * (grad + penalty)

    # Computing loss for the weights of the final iteration
    sigma = 1 / (1 + np.exp( - exp_guard(np.dot(tX, w))))

    # Computing log-loss for the weights in the final iteration
    loss = (np.dot(y.T, np.log(log_guard(sigma))) + 
        np.dot((1 - y).T, np.log(log_guard(1 - np.exp(exp_guard(sigma))))))

    # Converting weights back to 1D arrays
    w = w.reshape(len(w))

    return w, loss
