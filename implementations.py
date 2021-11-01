# -*- coding: utf-8 -*-
"""
Implementations
***************
Collection of functions developed for the machine learning project 1.
"""

import numpy as np

def standardize(tX_base, tX_modify=None):
    """
    Standardize the columns in tX to zero mean and unit variance.

    Parameters
    ----------
    tX_base : np.ndarray
        Base array to be used to calculate column-mean and standard deviation.
    tX_modify : np.ndarray or None
        Array to be standardized. Standardize tX_base if None was passed.

    Returns
    -------
    tX_std : np.ndarray
        Standardized tX with zero feature-means and unit feature-variance.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> tX_std = standardize(tX)
    array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079])
    """

    if tX_modify is None:  # Use base array if no modify array specified
        tX_modify = tX_base

    feature_mean = np.mean(tX_base, axis=0)  # Calcule feature means
    feature_std = np.std(tX_base, axis=0)  # Calcule feature standard deviations

    zero_ind = np.argwhere(feature_std == 0)  # Find zero variance features
    if len(zero_ind) == 0:  # If no such features exist, standardize normally
        tX_std = (tX_modify - feature_mean) / feature_std
    else:  # If zero variance features exist, standardize them only with mean
        tX_std = tX_modify - feature_mean
        feature_std[zero_ind] = 1
        tX_std /= feature_std
        print("WARNING: Zero variance feature(s) in columns", zero_ind)
        print("         Only standardize these with the feature-mean.")

    return tX_std

def polynomial_basis(tX_base, degrees, std=False, tX_modify=None):
    """
    Creates a polynomial basis from tX.

    Parameters
    ----------
    tX_base : np.ndarray
        Base array to be used to calculate column-mean and standard deviation.
    degrees : list or int
        List (or int) with the polynomial degrees (or maximum polynomial degree)
        that should be used as basis elements.
    std : bool, default=False
        Standardize features of each polynomial basis element.
    tX_modify : np.ndarray or None, default=None
        Array to be standardized. Standardize tX_base if None was passed.

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
    >>> tX_poly
    array([[ 1.,  1.,  1.],
           [ 1.,  2.,  4.],
           [ 1.,  3.,  9.],
           [ 1.,  4., 16.]])
    """
    if isinstance(degrees, int):  # Treat integer-valued degree as max degree
        degrees = range(degrees + 1)

    if tX_modify is None:  # Modify base array if no modify array specified
        tX_modify = tX_base

    if std:  # Always include the original features as a basis-element
        tX_poly = standardize(tX_base, tX_modify)
    else:
        tX_poly = tX_modify

    for deg in degrees:
        if deg == 0:  # Treat degree zero separately to avoid column-duplicates
            tX_poly = np.column_stack((np.ones(tX_modify.shape[0]), tX_poly))
        elif deg > 1:  # Append higher degree basis elements to 'tX'
            if std:  # If standardization is desired, standardize them before
                tX_poly = np.column_stack(
                    (tX_poly, standardize(tX_base**deg, tX_modify**deg)))
            else:
                tX_poly = np.column_stack((tX_poly, tX_modify**deg))

    return tX_poly

def substitute_999(tX_base, tX_modify, substitute='median'):
    """
    Replace '-999' values with a substitute.

    Parameters
    ----------
    tX_base : np.ndarray
        Base array to be used to calculate column-mean or -median with.
    tX_modify : np.ndarray
        Array to have -999 replaced by zero or the means/medians of tX_base.
    substitute : str {'zero', 'mean', 'median'}, default='median'
        Value to be substituted for -999. Column-wise mean or median are used.

    Returns
    -------
    tX_subs : np.ndarray
        2D array with the -999 values replaced by the substitute.

    Usage
    -----
    >>> tX_base = np.array([1, 2, 3, 4])
    >>> tX_modify = np.array([1, 2, -999, 4])
    >>> tX_subs = substitute_999(tX_base, tX_modify)
    >>> tX_subs
    array([1. , 2. , 2.5 ,  4])
    """

    # Replace '-999' values with 'nan'
    tX_nan_base = np.where(tX_base != -999, tX_base, np.nan)
    tX_nan_modify = np.where(tX_modify != -999, tX_modify, np.nan)

    # Substitute these values with a 'substitute'
    if substitute == 'zero':
        tX_subs = np.where(~np.isnan(tX_nan_modify), tX_nan_modify, 0)
    elif substitute == 'mean':
        mean = np.nanmean(tX_nan_base, axis=0)
        tX_subs = np.where(~np.isnan(tX_nan_modify), tX_nan_modify, mean)
    elif substitute == 'median':
        median = np.nanmean(tX_nan_base, axis=0)
        tX_subs = np.where(~np.isnan(tX_nan_modify), tX_nan_modify, median)

    return tX_subs

def substitute_outliers(tX_base, tX_modify, substitute='mean', level=3):
    """
    Replace outlier values with a substitute.

    Parameters
    ----------
    tX_base : np.ndarray
        Base array to be used to calculate column-mean or -median with.
    tX_modify : np.ndarray or float or int
        Array to have -999 replaced by zero or the means/medians of tX_base.
    substitute : str {'mean', 'median'}, default='mean'
        Value to replace outliers with. Column-wise mean or median are used.
    level : float, default=3
        The sensitivity level for declaring outliers. The threshold for outliers
        depends on substitute: |level*stdev  - mean| or |level*IQR  - median|.

    Returns
    -------
    tX_substituted : np.ndarray
        2D array with the outliers replaced by the substitute.

    Usage
    -----
    >>> tX_base = np.array([1, 2, 3, 4])
    >>> tX_modify = np.array([1, 2, 1000, 4])
    >>> tX_clean = substitute_outliers(tX_base, tX_modify)
    >>> tX_clean
    array([1. , 2. , 2.5 ,  4])
    """

    if substitute == 'mean':
        center = np.mean(tX_base, axis=0)  # Calcule feature-wise mean
        threshold = level * np.std(tX_base, axis=0)  # Calcule threshold
    elif substitute == 'median':
        center = np.median(tX_base, axis=0)  # Calcule feature-wise median
        Q3 = np.quantile(tX_base, 0.75, axis=0)  # Third quartile
        Q1 = np.quantile(tX_base, 0.25, axis=0)  # First quartile
        threshold = level * (Q3 - Q1)  # Threshold based on inter-quartile range
    tX_substituted = np.where(
        np.abs(tX_modify - center) < threshold, tX_modify, center)

    return tX_substituted

def _preprocess_arrays(y, tX, initial_w=None):
    """
    Convert potential 1D arrays or scalars to 2D arrays to maximize
    compatibility and set weights 'w' for first iteration.

    Parameters
    ----------
    y : np.ndarray or float or int
        Vector with the labels.
    tX : np.ndarray or float or int
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray or float or int or None, default=None
        Vector with initial weights to start the iteration from.

    Returns
    -------
    y : np.ndarray
        2D array with the labels.
    tX : np.ndarray
        2D array with the samples as rows and the features as columns.
    w : np.ndarray
        2D vector of initial weights.

    Notes
    -----
    This is crucial, because in most numpy functions, 1D and 2D arrays aren't
    compatible (i.e. np.dot, np.linalg.solve). Easiest examples are:

    >>> tX = np.array([1, 2])  # 1D tX
    >>> y = np.array([2, 4])
    >>> w = np.array([0])  # Only 1D weight (technically a scalar)
    >>> np.dot(tX, w)
    ValueError: shapes (2,) and (1,) not aligned: 2 (dim 0) != 1 (dim 0)

    >>> np.linalg.solve(1, 2)
    LinAlgError: 0-dimensional array given. Array must be at least 2-dimensional

    By first converting the arrays, it's possible to avoid these error:

    >>> y, tX, w = _preprocess_arrays(y, tX, w)  # Converting arrays
    >>> np.dot(tX, w)
    array([[0],
           [0]])

    >>> y, tX, _ = _preprocess_arrays(1, 2, None)  # Converting arrays
    >>> np.linalg.solve(y, tX)
    array([[2.0]])

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> y, tX, w = _preprocess_arrays(y, tX, None)
    >>> y, tX, w
    (array([[1], [2], [3], [4]]),
    array([[ 3], [ 6], [ 9], [12]]),
    array([[0.]]))
    """

    tX = np.array(tX)  # Convert tX to a numpy array (if it is not yet)
    if len(tX.shape) <= 1:  # Check if 'tX' is 1D, if yes, convert to 2D
        tX = tX.reshape((-1, 1))
    if initial_w is None:  # Use zero-vector for 'initial_w' if none specified
        initial_w = np.zeros(tX.shape[1])  # Works because we convert 'tX' to 2D
    w = np.array(initial_w).reshape((-1, 1))  # Convert 1D array to 2D array
    y = np.array(y).reshape((-1, 1))  # Convert 1D array to 2D array

    return y, tX, w

# Returns mean squared error for labels 'y', design matrix 'tX', and weights 'w'
def _compute_loss_mse(y, tX, w):
    return np.mean((y - np.dot(tX, w))**2) / 2

# Returns gradient of MSE for labels 'y', design matrix 'tX', and weights 'w'
def _compute_grad_mse(y, tX, w):
    return -np.dot(tX.T, y - np.dot(tX, w)) / len(y)

def least_squares_GD(y, tX, initial_w=None, max_iters=100, gamma=0.1):
    """
    Gradient descent algorithm for linear regression with MSE loss.

    Parameters
    ----------
    y : np.ndarray (N,)
        Vector with the labels.
    tX : np.ndarray (N,) or (N, D)
        Array with D samples as rows and N features as columns.
    initial_w : np.ndarray (D,) or None, default=None
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Learning rate, i.e. scaling factor for the gradient subtraction.

    Returns
    -------
    w : np.ndarray (D,)
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
    >>> w, loss
    (array([3.]), 0.0)
    """

    y, tX, w = _preprocess_arrays(y, tX, initial_w)

    for _ in range(max_iters):
        grad = _compute_grad_mse(y, tX, w)  # Compute the MSE gradient
        w = w - gamma * grad  # Update weights with scaled negative gradient

    loss = _compute_loss_mse(y, tX, w)  # Compute the MSE loss
    w = w.reshape(-1)  # Convert weights back to 1D array
    return w, loss

def least_squares_SGD(y, tX, initial_w=None, max_iters=1000, gamma=0.1, seed=None):
    """
    Stochastic gradient descent algorithm for mean square error (MSE) loss.

    Parameters
    ----------
    y : np.ndarray (N,)
        Vector with the labels.
    tX : np.ndarray (N,) or (N, D)
        Array with N samples as rows and D features as columns.
    initial_w : np.ndarray (D,) or None, default=None
        Vector with initial weights to start the iteration from.
    max_iters : int, default=1000
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.
    seed : int or None, default=None
        Seed to be used for the random number generator.

    Returns
    -------
    w : np.ndarray (D,)
        Vector containing the final weights.
    loss : float
        Mean square error loss function evaluated for the final weights.

    References
    ----------
    [3] M. Jaggi, R. Urbanke, and M. E. Khan, "Optimization",
        Machine Learning (CS-433), pp. 8-10, September 23, 2021.

    Notes
    -----
    In order to make this regressor work faster, we do not use external function
    calls for the gradient descent iterations.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = least_squares_SGD(y, tX)
    >>> w, loss
    (array([3.]), 0.0)
    """

    if initial_w is None:  # Use zero vector as 'initial_w' if None is specified
        tX = np.array(tX)  # Convert tX to a numpy array (to use .shape)
        if len(tX.shape) <= 1:  # Check if 'tX' is 1D, if yes, convert to 2D
            tX = tX.reshape((-1, 1))
        w = np.zeros(tX.shape[1])
    else:
        w = initial_w

    if seed is not None:  # Use a seed (if one is specified)
        np.random.seed(seed)
    rand_ind = np.random.choice(np.arange(len(y)), max_iters)  # Random indices

    for i in range(max_iters):
        y_rand = y[rand_ind[i]]  # Picking the random sample
        tX_rand = tX[rand_ind[i]]  # Picking the random sample
        e_rand = y_rand - np.inner(tX_rand, w) # Random error
        grad_rand = - e_rand * tX_rand # Random gradient for MSE loss
        w = w - gamma * grad_rand # Updating with scaled negative gradient

    loss = _compute_loss_mse(y, tX, w)  # Compute the MSE loss
    return w, loss

def least_squares(y, tX):
    """
    Exact analytical solution for the weights using the normal equations.

    Parameters
    ----------
    y : np.ndarray (N,)
        Vector with the labels.
    tX : np.ndarray (N,) or (N, D)
        Array with N samples as rows and D features as columns.

    Returns
    -------
    w : np.ndarray (D,)
        Vector containing the final weights.
    loss : float
        Mean square error loss function evaluated for the final weights.

    References
    ----------
    [4] M. Jaggi, R.Urbanke, and M. E. Khan, "Least Squares",
        Machine Learning (CS-433), p. 7, October 5, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = least_squares(y, tX)
    >>> w, loss
    (array([3.]), 0.0)
    """

    y, tX, _ = _preprocess_arrays(y, tX, None)

    #  Solving for the exact weights according to the normal equations in [4]
    w = np.linalg.solve(np.dot(tX.T, tX), np.dot(tX.T, y))

    loss = _compute_loss_mse(y, tX, w)  # Compute MSE loss
    w = w.reshape(-1)  # Convert weights back to 1D array
    return w, loss

def ridge_regression(y, tX, lambda_=0.1):
    """
    Exact analytical solution for the weights using the ridge-regularized
    normal equations.

    Parameters
    ----------
    y : np.ndarray (N,)
        Vector with the labels.
    tX : np.ndarray (N,) or (N, D)
        Array with N samples as rows and D features as columns.
    lambda_ : float, default=0.1
        Regularization parameter.

    Returns
    -------
    w : np.ndarray (D,)
        Vector containing the final weights.
    loss : float
        Mean square error loss function evaluated for the final weights.

    References
    ----------
    [5] M. Jaggi, R. Urbanke, and M. E. Khan, "Regularization: Ridge Regression
        and Lasso", Machine Learning (CS-433), p. 3, October 7, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = 3*tX
    >>> w, loss = ridge_regression(y, tX)
    >>> w, loss
    (array([2.92207792]), 0.8766233766233765)
    """

    y, tX, _ = _preprocess_arrays(y, tX, None)  #  Convert from 1D to 2D arrays

    # Solving for the exact weights with the regularized normal equations in [5]
    penalty = lambda_ * 2*len(y) * np.identity(tX.shape[1])  # Penalty-term
    w = np.linalg.solve(np.dot(tX.T, tX) + penalty, np.dot(tX.T, y))

    loss = _compute_loss_mse(y, tX, w) + lambda_ * np.linalg.norm(w)**2 # Compute MSE loss and add penalty term
    w = w.reshape(-1)  # Convert weights back to 1D array
    return w, loss

# Defining overflow-guard for np.exp()
def _exp_guard(t):
    return np.clip(t, -709, 709)

# Defining underflow-guard for np.log()
def _log_guard(t):
    return np.maximum(t, 1e-20)

# Sigmoid function
def _compute_sigmoid(t):
    return 1 / (1 + np.exp(_exp_guard(-t)))

# Logistic loss function for labels in {0, 1}
def _compute_loss_log(y, tX, w):
    sigma = _compute_sigmoid(np.dot(tX, w))
    return - np.sum((y * np.log(_log_guard(sigma)) +
                    (1 - y) * np.log(_log_guard(1 - sigma))))

# Gradient of logistic loss function for labels in {0, 1}
def _compute_grad_log(y, tX, w):
    return np.dot(tX.T, (_compute_sigmoid(np.dot(tX, w)) - y))

def logistic_regression(y, tX, initial_w=None, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with logistic loss function.

    Parameters
    ----------
    y : np.ndarray (N,) in {-1, 1} or {0, 1}
        Vector with the labels.
    tX : np.ndarray (N,) or (N, D)
        Array with N samples as rows and D features as columns.
    initial_w : np.ndarray (D,) or None, default=None
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.

    Returns
    -------
    w : np.ndarray (D,)
        Vector containing the final weights.
    loss : float
        Logistic loss function evaluated for the final weights.

    References
    ----------
    [6] N. Flammarion, R. Urbanke, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 2-12, October 21, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = np.array([0, 0, 1, 1])
    >>> w, loss = logistic_regression(y, tX)
    >>> w, loss
    (array([0.28769978]), 2.495831016013658)
    """

    y[y <= 0] = 0  # If labels are in {-1, 1} we convert them to {0, 1}
    
    y, tX, w = _preprocess_arrays(y, tX, initial_w)

    for _ in range(max_iters):
        w = w - gamma * _compute_grad_log(y, tX, w)  # Update [7]

    loss = _compute_loss_log(y, tX, w)  # Compute log-loss for final iteration
    w = w.reshape(-1)  # Convert weights back to 1D array
    return w, loss

def reg_logistic_regression(y, tX, lambda_=0.1, initial_w=None, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with (ridge) regularized logistic loss function.

    Parameters
    ----------
    y : np.ndarray (N,) in {-1, 1} or {0, 1}
        Vector with the labels.
    tX : np.ndarray (N,) or (N, D)
        Array with N samples as rows and D features as columns.
    lambda_ : float, default=0.1
        Regularization parameter.
    initial_w : np.ndarray (D,) or None, default=None
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.

    Returns
    -------
    w : np.ndarray (D,)
        Vector containing the final weights.
    loss : float
        Logistic loss function evaluated for the final weights.

    References
    ----------
    [7] N. Flammarion, R. Urbanke, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 16-17, October 21, 2021.

    Usage
    -----
    >>> tX = np.array([1, 2, 3, 4])
    >>> y = np.array([0, 0, 1, 1])
    >>> w, loss = reg_logistic_regression(y, tX)
    >>> w, loss
    (array([0.25440098]), 2.505664688917194)
    """

    y[y <= 0] = 0  # If labels are in {-1, 1}, convert them to {0, 1}

    y, tX, w = _preprocess_arrays(y, tX, initial_w)
    
    for _ in range(max_iters):
        penalty = 2*len(y) * lambda_ * w  # Penalty-term
        w = w - gamma * (_compute_grad_log(y, tX, w) + penalty)  # Update [7]

    loss = _compute_loss_log(y, tX, w) + lambda_ * np.linalg.norm(w)**2 # Compute log-loss for final iteration and add penalty term
    w = w.reshape(-1)  # Convert weights back to 1D array
    return w, loss

def categorical_to_binary(tX, col=-1):
    """
    Transform a categorical feature to multiple binary features.

    Parameters
    ----------
    tX : np.ndarray (N, D)
        Array with the samples as rows and the features as columns.
    col : int, default=-1
        Categorical column to be binaryfied.

    Returns
    -------
    tX : np.ndarray (N+C, D)
        Array with the 'col' feature with C categories binaryfied.
    """

    ref = tX[:,col].reshape(-1,1)
    tX = np.delete(tX, col, axis = 1)
    values = np.unique(ref)
    for val in values:
        np.hstack((tX,(ref == val)))

    return tX

def higgs_preprocessing(tX_train, tX_valid, degrees, categorical=False):
    """
    Function to simplify data preprocessing for the optimized regressor.

    Parameters
    ----------
    tX_train : np.ndarray (N,) or (N, D)
        Array with the samples as rows and the features as columns.
    y_valid : np.ndarray (N,) in {-1, 1} or {0, 1}
        Vector with the labels.
    degrees : list or int   
        List (or int) with the polynomial degrees (or maximum polynomial degree)
        that should be used as basis elements.

    Returns
    -------
    tXi : np.ndarray (N,) in {-1, 1}
        Subarray with all samples that have value i in feature 'PRI_num_jet'.
    yi : np.ndarray (N,) in {-1, 1}
        Labels of all samples that have value i in feature 'PRI_num_jet'.
    """

    if categorical:
        PRI_jet_num_train = tX_train[:,22].reshape(-1,1)
        PRI_jet_num_valid = tX_valid[:,22].reshape(-1,1)

        tX_train = np.delete(tX_train, 22, axis=1)
        tX_valid = np.delete(tX_valid, 22, axis=1)

    # Substitute all '-999' values with the feature-medians from the train sets
    tX_valid = substitute_999(tX_train, tX_valid, 'median')
    tX_train = substitute_999(tX_train, tX_train, 'median')

    # Substitute all outlying values with the feature-mean from the train sets
    tX_valid = substitute_outliers(tX_train, tX_valid, 'mean', 3)
    tX_train = substitute_outliers(tX_train, tX_train, 'mean', 3)

    # Augment features with a polynomial basis with standardization
    tX_valid = polynomial_basis(tX_train, degrees, True, tX_valid)
    tX_train = polynomial_basis(tX_train, degrees, True, tX_train)
    
    if categorical:
        tX_train = np.hstack((tX_train, PRI_jet_num_train))
        tX_valid = np.hstack((tX_valid, PRI_jet_num_valid))
        tX_train = categorical_to_binary(tX_train)
        tX_valid = categorical_to_binary(tX_valid)
    
    return tX_train, tX_valid

def split_PRI_jet(tX, y):
    """
    Split data according to the value of 'PRI_num_jet' (23rd feature).

    Parameters
    ----------
    tX : np.ndarray (N,) or (N, D)
        Array with the samples as rows and the features as columns.
    y : np.ndarray (N,) in {-1, 1} or {0, 1}
        Vector with the labels.

    Returns
    -------
    tXi : np.ndarray (N,) in {-1, 1}
        Subarray with all samples that have value i in feature 'PRI_num_jet'.
    yi : np.ndarray (N,) in {-1, 1}
        Labels of all samples that have value i in feature 'PRI_num_jet'.
    """

    PRI_jet_num = tX[:,22]

    # Split data according to the 'PRI_jet_num' feature
    tX0 = tX[PRI_jet_num == 0, :]
    y0 = y[PRI_jet_num == 0]
    tX1 = tX[PRI_jet_num == 1, :]
    y1 = y[PRI_jet_num == 1]
    tX2 = tX[PRI_jet_num == 2, :]
    y2 = y[PRI_jet_num == 2]
    tX3 = tX[PRI_jet_num == 3, :]
    y3 = y[PRI_jet_num == 3]

    return tX0, y0, tX1, y1, tX2, y2, tX3, y3

# Remove constant columns from an array 'tX'
def remove_const_cols(tX):
    return np.delete(tX, np.argwhere(np.std(tX, axis=0) == 0), axis=1)

def optimized_regression(base_regressor, tX, y, data, degrees, param, pred_ratio=False):
    """
    Splits the data set into four different sets according to the value of
    'PRI_num_jet' (23rd feature) and solves these subproblems separately.

    Parameters
    ----------
    base_regressor : str {'least_squares_GD', 'least_squares_SGD',
                          'least_squares', 'ridge_regression',
                          'logistic_regression', 'reg_logistic_regression'}
        The base-regressor to be used for solving the subproblems.
    tX : np.ndarray (N,) or (N, D)
        Training array with the samples as rows and the features as columns.
    y : np.ndarray (N,) in {-1, 1} or {0, 1}
        Vector with the labels.
    data : np.ndarray (M,) or (M, D)
        Testing array with the samples as rows and the features as columns.
    degrees : list or int
        List (or int) with the polynomial degrees (or maximum polynomial degree)
        that should be used as basis elements.
    param : dict
        Parameters to be used for the base-regressor.
    pred_ratio : float or False, default=False
        Adjust threshold for predictions to match label ratio of training set.

    Returns
    -------
    y_pred : np.ndarray (N,) in {-1, 1}
        Vector with the predictions.

    References
    ----------
    [8]  C. Adam-Bourdariosa, G. Cowanb , C. Germain , B. Kégl, D. Rousseau,
         Learning to discover: the Higgs boson machine learning challenge. 2014.

    Notes
    -----
    Only applicable to the Higgs boson machine learning challenge data sets.
    """

    # Split the data according to 'PRI_jet_num'
    tX0, y0, tX1, y1, tX2, y2, tX3, y3 = split_PRI_jet(tX, y)

    # Remove constant features
    tX0 = remove_const_cols(tX0)
    tX1 = remove_const_cols(tX1)
    tX2 = remove_const_cols(tX2)
    tX3 = remove_const_cols(tX3)

    # Obtain the indices for all 'PRI_jet_num', and split the data accordingly
    PRI_jet = data[:, 22]
    data0_ind = (PRI_jet == 0)
    data0 = data[data0_ind, :]
    data1_ind = (PRI_jet == 1)
    data1 = data[data1_ind, :]
    data2_ind = (PRI_jet == 2)
    data2 = data[data2_ind, :]
    data3_ind = (PRI_jet == 3)
    data3 = data[data3_ind, :]
    
    # Remove constant features
    data0 = remove_const_cols(data0)
    data1 = remove_const_cols(data1)
    data2 = remove_const_cols(data2)
    data3 = remove_const_cols(data3)

    # Do preprocessing
    tX0, data0 = higgs_preprocessing(tX0, data0, degrees)
    tX1, data1 = higgs_preprocessing(tX1, data1, degrees)
    tX2, data2 = higgs_preprocessing(tX2, data2, degrees)
    tX3, data3 = higgs_preprocessing(tX3, data3, degrees)

    # Train 4 separate models
    w0, _ = eval(base_regressor)(y0, tX0, **param)
    w1, _ = eval(base_regressor)(y1, tX1, **param)
    w2, _ = eval(base_regressor)(y2, tX2, **param)
    w3, _ = eval(base_regressor)(y3, tX3, **param)

    # Building the prediction for all labels
    y_pred = np.zeros(data.shape[0])
    y_pred[data0_ind] = np.dot(data0, w0)
    y_pred[data1_ind] = np.dot(data1, w1)
    y_pred[data2_ind] = np.dot(data2, w2)
    y_pred[data3_ind] = np.dot(data3, w3)

    # Adjust threshold value to obtain a given prediction ratio
    if not pred_ratio:
        threshold = 0
    else:
        threshold = np.quantile(y_pred, pred_ratio)

    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    
    return y_pred
