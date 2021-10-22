# -*- coding: utf-8 -*-
"""
Implementations
***************
Collection of functions developed for the machine learning project 1.
The function descriptions were heavily inspired by the numpy package.
"""

import numpy as np

def standardize(tX):
    """
    Standardizes the columns in tX to zero mean and unit variance.

    Parameters
    ----------
    tX : np.ndarray
        Array with the samples as rows and the features as columns.

    Returns
    -------
    tX_std : np.ndarray
        Standardized tX with zero feature-means and unit feature-variance.

    Usage
    -----
    >>> tX = np.array([[1, 2],
                       [0, 4],
                       [1, 3]])
    >>> tX_std = standardize(tX)
    >>> print(tX_std)
    [[ 0.70710678 -1.22474487]
     [-1.41421356  1.22474487]
     [ 0.70710678  0.        ]]

    """

    # Calculating the standardization values (feature-wise)
    feature_mean = np.mean(tX, axis=0)
    feature_std = np.std(tX, axis=0)

    zero_ind = np.argwhere(feature_std == 0)
    if len(zero_ind) == 0: # Checking for features with zero standard deviation
        tX_std = (tX - feature_mean) / feature_std
    else:
        tX_std = tX - feature_mean
        feature_std[zero_ind] = 1 # Leaving zero-variance features unchanged
        tX_std /= feature_std
        print("WARNING: Zero variance feature encountered.")
        print("         Only standardized this one with the feature-mean.")
    return tX_std


def polynomial_basis(tX, degrees, indices=False, std=False):
    """
    Creates a polynomial basis from tX.

    Parameters
    ----------
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    degrees : list or int
        List (or int) with the polynomial degrees (or maximum polynomial degree)
        that should be used as basis elements.
    indices : np.ndarray or False, default=False
        Chose a subset of indices which should be boosted.

    std : bool, default=False
        Standardize features of each polynomial basis element.

    Returns
    -------
    tX_poly : np.ndarray
        tX in a polynomial basis.

    Notes
    -----
    The degree 0 is the offset-vector containing just ones. When passing the
    maximum polynomial degree as an integer as 'degrees=max_degree', this
    offset-vector is automatically added to tX. If you don't want it included,
    pass 'degrees=range(1, max_degree + 1)' instead.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> tX_poly = polynomial_basis(tX, [0, 1, 2])
    >>> print(tX_poly)
    [[ 1.  1.  1.]
     [ 1.  2.  4.]
     [ 1.  3.  9.]
     [ 1.  4. 16.]]

    >>> tX = np.arange(10).reshape((2, 5))
    >>> tX_poly = polynomial_basis(tX, 2, [1, 2])
    >>> print(tX_poly)
    [[ 1.  0.  1.  2.  3.  4.  1.  4.]
    [ 1.  5.  6.  7.  8.  9. 36. 49.]]

    """

    if not indices: # If no indices are selected, use all indices
        indices = np.arange(tX.shape[1])
    if isinstance(degrees, int): # Checking if a maximum degree (int) was passed
        degrees = range(degrees + 1)
    # Always including the original features (degree 1) as a basis-element
    if std:
        tX_poly = standardize(tX)
    else:
        tX_poly = tX
    for deg in degrees:
        if deg == 0: # Degree zero treated separately to avoid column-duplicates
            tX_poly = np.column_stack((np.ones(tX.shape[0]), tX_poly))
        elif deg > 1:
            if std:
                tX_poly = np.column_stack((tX_poly, standardize(tX[:, indices]**deg)))
            else:
                tX_poly = np.column_stack((tX_poly, tX[:, indices]**deg))
    return tX_poly


def compute_loss_mse(y, tX, w):
    """
    Computes the mean square error (MSE) loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    w: np.ndarray
        Vector containing the weights.

    Returns
    -------
    loss : float
        Mean square error loss function evaluated with the weights w.

    References
    ----------
    [1] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> tx = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    >>> y = np.array([5, 6])
    >>> w = np.array([1, 1, 1, 1])
    >>> loss = compute_loss_mse(y, tx, w)
    >>> print(loss)
    10.25

    """
    e = y - np.dot(tX, w.reshape((-1, 1)))
    loss = 1/2 * np.mean(np.power(e, 2))
    return loss


def compute_loss_mae(y, tX, w):
    """
    Computes the mean absolute error (MAE) loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    w: np.ndarray
        Vector containing the weights.

    Returns
    -------
    loss : float
        Mean absolute error loss function evaluated with the weights w.

    References
    ----------
    [1] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> tX = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    >>> y = np.array([5, 6])
    >>> w = np.array([1, 1, 1, 1])
    >>> loss = compute_loss_mae(y, tx, w)
    >>> print(loss)
    4.5

    """
    e = y - np.dot(tX, w.reshape((-1, 1)))
    loss = np.mean(np.abs(e))
    return loss


def least_squares_GD(y, tX, initial_w=None, max_iters=100, gamma=0.1, tol=None):
    """
    Gradient descent algorithm for linear regression with MSE loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray or None, default=None
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
        Mean square error loss function evaluated with the optimized weights.

    References
    ----------
    [1] M. Jaggi, R. Urbanke, and M. E. Khan, "Optimization",
        Machine Learning (CS-433), pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = least_squares_GD(y, tX)
    >>> print(w, loss)
    [3.] 0.0
    """

    if len(tX.shape) == 1: # Checking if 'tX' is a 1D array
        tX = tX.reshape((-1, 1)) # consequently converting to a 2D array

    # Zero vector for 'initial_w' if no initial value was specified
    if initial_w is None:
        initial_w = np.zeros(tX.shape[1])
    
    # Converting 1D arrays to 2D arrays
    w = initial_w.reshape((-1, 1))
    y = y.reshape((-1, 1))

    for _ in range(max_iters):
        e = y - np.dot(tX, w) # Error vector
        grad = - np.dot(tX.T, e) / len(y) # Gradient for MSE loss
        w = w - gamma * grad # Updating with scaled negative gradient
        # Stopping criterion
        if (tol != None):
            if (np.linalg.norm(grad) < tol):
                print("NOTE: Stopping criterion met in iteration ", iter, ".")
                break
    loss = np.mean(np.power(e, 2)) / 2 # Computing loss (MSE)
    w = w.reshape(-1) # Converting weights back to 1D arrays

    return w, loss


def least_squares_SGD(y, tX, initial_w=None, max_iters=100000, gamma=0.1, seed=None):
    """
    Stochastic gradient descent algorithm for mean square error (MSE) loss.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray or None, default=None
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100000
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
        Mean square error loss function evaluated with the final weights.

    References
    ----------
    [3] M. Jaggi, R. Urbanke, and M. E. Khan, "Optimization",
        Machine Learning (CS-433), pp. 8-10, September 23, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = least_squares_SGD(y, tX)
    >>> print(w, loss)
    [3.] 9.121204216617949e-31

    """


    if initial_w is None: # Zero vector for 'initial_w' if none was specified
        w = np.zeros(tX.shape[1])

    if seed is not None: # Using the desired seed (if one is specified)
        np.random.seed(seed)
    rand_ind = np.random.choice(np.arange(len(y)), max_iters) # Random indices

    for iter in range(max_iters):
        # Picking a random sample
        y_rand = y[rand_ind[iter]]
        tX_rand = tX[rand_ind[iter]]
        e_rand = y_rand - np.inner(tX_rand, w) # Random error
        grad_rand = - e_rand * tX_rand / len(y) # Random gradient for MSE loss
        w = w - gamma * grad_rand # Updating with scaled negative gradient
        
        # TODO: find a way to stop the algorithm efficiently

    # Computing loss (MSE) for the weights in the final iteration
    e = y - np.dot(tX, w)
    loss = np.mean(np.power(e, 2)) / 2
    return w, loss


def least_squares(y, tX):
    """
    Exact analytical solution for the weights using the normal equations.

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
        Mean square error loss function evaluated with the final weights.

    References
    ----------
    [4] M. Jaggi, R.Urbanke, and M. E. Khan, "Least Squares",
        Machine Learning (CS-433), p. 7, October 5, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = least_squares(y, tX)
    >>> print(w, loss)
    [3.] 0.0

    """

    y = y.reshape((-1, 1)) # Converting a potential 1D array to a 2D array
    if len(tX.shape) == 1: # Checking if 'tX' is a 1D array
        tX = tX.reshape((-1, 1)) # consequently converting to a 2D array
    w = np.linalg.solve(np.dot(tX.T, tX), np.dot(tX.T, y)) # Solving for w [4]
    loss = compute_loss_mse(y, tX, w) # Computing loss (MSE)
    w = w.reshape(-1) # Converting weights back to 1D arrays
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
        Mean square error loss function evaluated with the final weights.

    References
    ----------
    [5] M. Jaggi, R. Urbanke, and M. E. Khan, "Regularization: Ridge Regression
        and Lasso", Machine Learning (CS-433), p. 3, October 7, 2021.

    Usage
    -----
    >>> tX = np.array([[9, 2, 7, 3, 1, 8],
                       [2, 6, 1, 8, 1, 7]]).T
    >>> y = 3*tX
    >>> w, loss = ridge_regression(y, tX)
    >>> print(w, loss)
    [2.92207792] 0.02276943835385406
    """

    y = y.reshape((-1, 1)) # Converting a potential 1D array to a 2D array
    if len(tX.shape) == 1: # Checking if 'tX' is a 1D array
        tX = tX.reshape((-1, 1)) # consequently converting to a 2D array
    penalty = lambda_ * 2 * len(y) * np.identity(tX.shape[1]) # "Penalty"-term

    # Solving for the exact weights according to the normal equations in [5]
    w = np.linalg.solve(np.dot(tX.T, tX) + penalty, np.dot(tX.T, y))
    loss = compute_loss_mse(y, tX, w) # Computing loss (MSE)
    w = w.reshape(-1) # Converting weights back to 1D arrays
    return w, loss


def logistic_regression(y, tX, initial_w=None, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with logistic loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray or None, default=None
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
        Mean square error loss function evaluated with the final weights.

    References
    ----------
    [6] N. Flammarion, R. Urbanke, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 2-12, October 21, 2021.

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
    >>> tX = np.array([[9, 2, 7, 3, 1, 8],
                       [2, 6, 1, 8, 1, 7]]).T
    >>> y = np.array([-1, 1, -1, 1, 1, -1])
    >>> w, loss = logistic_regression(y, tX)
    >>> print(w, loss)
    [-181.50236696   48.74761968] -0.0
    """
    if len(tX.shape) == 1: # Checking if 'tX' is a 1D array
        tX = tX.reshape((-1, 1)) # consequently converting to a 2D array

    # Zero vector for 'initial_w' if no initial value was specified
    if initial_w is None:
        initial_w = np.zeros(tX.shape[1])
    
    # Converting 1D arrays to 2D arrays
    w = initial_w.reshape((-1, 1))
    y = y.reshape((-1, 1))

    # Defining overflow-guards for np.exp() and np.log() (see Notes above)
    exp_guard = lambda x : np.clip(x, -709, 709)
    log_guard = lambda x : np.maximum(x, 1e-20)
 
    for _ in range(max_iters):
        sigma = 1 / (1 + np.exp(exp_guard(np.dot(tX, w) * y)))
        grad = - np.dot(tX.T, sigma * y)
        w = w - gamma * grad # Updating weights with scaled negative gradient
    # Computing loss for the weights of the final iteration
    sigma = 1 / (1 + np.exp( - exp_guard(np.dot(tX, w))))
    # Computing log-loss for the weights in the final iteration
    loss = np.asscalar(- np.dot(y.T, np.log(log_guard(sigma))) -
                         np.dot((1-y).T, np.log(log_guard(1 - sigma))))
    w = w.reshape(-1) # Converting weights back to 1D arrays
    return w, loss


def reg_logistic_regression(y, tX, lambda_=0.1, initial_w=None, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with (ridge) regularized logistic loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    lambda_ : float, default=0.1
        Regularization parameter.
    initial_w : np.ndarray or None, default=None
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
        Mean square error loss function evaluated with the final weights.

    References
    ----------
    [7] N. Flammarion, R. Urbanke, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 16-17, October 21, 2021.

    Usage
    -----
    >>> tX = np.array([[9, 2, 7, 3, 1, 8],
                       [2, 6, 1, 8, 1, 7]]).T
    >>> y = np.array([-1, 1, -1, 1, 1, -1])
    >>> w, loss = reg_logistic_regression(y, tX)
    >>> print(w, loss)
    [303.7492403  126.33072599] 5687.462876798688
    """

    if len(tX.shape) == 1: # Checking if 'tX' is a 1D array
        tX = tX.reshape((-1, 1)) # consequently converting to a 2D array

    # Zero vector for 'initial_w' if no initial value was specified
    if initial_w is None:
        initial_w = np.zeros(tX.shape[1])
    
    # Converting 1D arrays to 2D arrays
    w = initial_w.reshape((-1, 1))
    y = y.reshape((-1, 1))
    # Defining overflow-guards for np.exp() and np.log() (see Notes above)
    exp_guard = lambda x : np.clip(x, -709, 709)
    log_guard = lambda x : np.maximum(x, 1e-20)
 
    for _ in range(max_iters):
        sigma = 1 / (1 + np.exp(exp_guard(np.dot(tX, w) * y)))
        grad = - np.dot(tX.T, sigma * y)
        penalty = lambda_ * w # Calculating "penalty"-term from regularization
        w = w - gamma * (grad + penalty) # Updating weights [7]
    # Computing loss for the weights of the final iteration
    sigma = 1 / (1 + np.exp( - exp_guard(np.dot(tX, w))))
    # Computing log-loss for the weights in the final iteration
    loss = np.asscalar(- np.dot(y.T, np.log(log_guard(sigma))) -
                         np.dot((1-y).T, np.log(log_guard(1 - sigma))) +
                         lambda_ / 2 * np.dot(w.T, w))
    w = w.reshape(-1) # Converting weights back to 1D arrays
    return w, loss

def lasso_SD(y, tX, initial_w, max_iters=1000, gamma=0.1, lambda_ = 0.1, threshold=None):
    """
    Lasso Subgradient Descent regressor with MSE loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray or None, default=None
        Vector with initial weights to start the iteration from.
    max_iters : int, default=1000
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the subgradient subtraction.
    lambda_ : float, defalut=0.1
        Regularization parameter.
    threshold: float, default=None
        Threshold under which the weight entries are set to zero in order to have sparsity.
        
    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean square error loss function evaluated with the final weights.

    References
    ----------
    [1] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> tX = np.random.rand(1000, 10)
    >>> y = np.random.rand(1000, 1)
    >>> initial_w = np.ones(tX.shape[1] , dtype=float)
    >>> w, loss = lasso_SD(y, tX, initial_w)
    >>> print(w, loss)
    [0.09219308 0.08941125 0.08259039 0.10009369 0.11471479 0.10400056
    0.11830658 0.05418399 0.06966045 0.10933284] 0.043595101687066144
    """
    # Number of samples
    N = len(y)

    # Converting 1D arrays to 2D arrays
    w = initial_w.reshape((len(initial_w), 1))
    y = y.reshape((N, 1))

    # Checking if 'tX' is a 1D array, and consequently converting to a 2D array
    if len(tX.shape) == 1:

        tX = tX.reshape((N, 1))

    for iter in range(max_iters):

        # Error vector
        e = y - np.dot(tX, w)

        # Subgradient for the Lasso loss function
        subgrad = - np.dot(tX.T, e) / N + lambda_ / np.sqrt(1 + iter) * np.sign(w)

        # Updating weights with negative gradient scaled by 'gamma'
        w = w - gamma * subgrad


    # Computing loss (MSE) for the weights in the final iteration
    loss = np.mean(e**2) / 2

    # Converting weights back to 1D arrays
    w = w.reshape(len(w))

    # Setting to zero all the entries of w under the threshold in absolute value
    if (threshold != None):
        w[np.absolute(w) < threshold] = 0

    return w, loss