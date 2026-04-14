"""PCA monitoring statistics: T², Q/SPE, and T² confidence ellipse."""

import numpy as np
from scipy.stats import f


def t2(pca, X):
    """Hotelling's T² for each observation.

    Parameters
    ----------
    pca : fitted sklearn PCA object
    X : array-like, shape (n, p), standardized data

    Returns
    -------
    t2 : array, shape (n,)
    """
    T = pca.transform(X)
    return np.sum(T ** 2 / pca.explained_variance_, axis=1)


def t2_limit(pca, n, alpha=0.95):
    """T² confidence limit based on F-distribution.

    Parameters
    ----------
    pca : fitted sklearn PCA object
    n : int, number of observations
    alpha : float, confidence level (default 0.95)

    Returns
    -------
    t2_crit : float
    """
    k = pca.n_components_
    return (k * (n - 1) / (n - k)) * f.ppf(alpha, k, n - k)


def t2_ellipse(pca, n, alpha=0.95, npts=100):
    """T² confidence ellipse in PC1-PC2 space.

    Parameters
    ----------
    pca : fitted sklearn PCA object
    n : int, number of observations
    alpha : float, confidence level (default 0.95)
    npts : int, number of points on the ellipse

    Returns
    -------
    x, y : arrays, shape (npts,), ellipse coordinates in PC1-PC2 space
    """
    t2_crit = t2_limit(pca, n, alpha)
    theta = np.linspace(0, 2 * np.pi, npts)
    x = np.sqrt(pca.explained_variance_[0] * t2_crit) * np.cos(theta)
    y = np.sqrt(pca.explained_variance_[1] * t2_crit) * np.sin(theta)
    return x, y


def spe(pca, X):
    """Squared Prediction Error (Q statistic) for each observation.

    Parameters
    ----------
    pca : fitted sklearn PCA object
    X : array-like, shape (n, p), standardized data

    Returns
    -------
    q : array, shape (n,)
    """
    T = pca.transform(X)
    X_hat = pca.inverse_transform(T)
    return np.sum((X - X_hat) ** 2, axis=1)
