"""
Kernel implementations for RULSIF algorithm.

This module provides various kernel functions that can be used with the
RULSIF algorithm for change point detection.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union


class BaseKernel(ABC):
    """
    Base class for kernel functions.
    
    All kernel implementations should inherit from this class and
    implement the apply method.
    """
    
    @abstractmethod
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Apply the kernel function to compute kernel matrix.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_features, n_samples_X)
            First set of samples.
            
        Y : np.ndarray of shape (n_features, n_samples_Y)
            Second set of samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples_Y, n_samples_X)
            Kernel matrix K(X, Y).
        """
        pass
    
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Convenience method to call the kernel function."""
        return self.apply(X, Y)


class GaussianKernel(BaseKernel):
    """
    Gaussian (RBF) kernel implementation.
    
    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    
    Parameters
    ----------
    sigma : float, default=1.0
        Width parameter of the Gaussian kernel.
    """
    
    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self.sigma = sigma
    
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel matrix.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_features, n_samples_X)
            First set of samples.
            
        Y : np.ndarray of shape (n_features, n_samples_Y)
            Second set of samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples_Y, n_samples_X)
            Gaussian kernel matrix.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Compute squared distances efficiently
        X_norm = np.sum(X ** 2, axis=0, keepdims=True)
        Y_norm = np.sum(Y ** 2, axis=0, keepdims=True)
        
        # K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        distances = X_norm.T + Y_norm - 2 * np.dot(X.T, Y)
        
        # Apply Gaussian function
        kernel_matrix = np.exp(-distances / (2 * self.sigma ** 2))
        
        return kernel_matrix
    
    def __repr__(self) -> str:
        return f"GaussianKernel(sigma={self.sigma})"


class LinearKernel(BaseKernel):
    """
    Linear kernel implementation.
    
    K(x, y) = x^T * y
    """
    
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute linear kernel matrix.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_features, n_samples_X)
            First set of samples.
            
        Y : np.ndarray of shape (n_features, n_samples_Y)
            Second set of samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples_Y, n_samples_X)
            Linear kernel matrix.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # K(x, y) = x^T * y
        kernel_matrix = np.dot(X.T, Y)
        
        return kernel_matrix
    
    def __repr__(self) -> str:
        return "LinearKernel()"


class PolynomialKernel(BaseKernel):
    """
    Polynomial kernel implementation.
    
    K(x, y) = (gamma * x^T * y + coef0)^degree
    
    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial.
        
    gamma : float, default=1.0
        Kernel coefficient.
        
    coef0 : float, default=0.0
        Constant term.
    """
    
    def __init__(self, degree: int = 2, gamma: float = 1.0, coef0: float = 0.0):
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute polynomial kernel matrix.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_features, n_samples_X)
            First set of samples.
            
        Y : np.ndarray of shape (n_features, n_samples_Y)
            Second set of samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples_Y, n_samples_X)
            Polynomial kernel matrix.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # K(x, y) = (gamma * x^T * y + coef0)^degree
        linear_kernel = np.dot(X.T, Y)
        kernel_matrix = (self.gamma * linear_kernel + self.coef0) ** self.degree
        
        return kernel_matrix
    
    def __repr__(self) -> str:
        return f"PolynomialKernel(degree={self.degree}, gamma={self.gamma}, coef0={self.coef0})"


class LaplacianKernel(BaseKernel):
    """
    Laplacian kernel implementation.
    
    K(x, y) = exp(-||x - y|| / sigma)
    
    Parameters
    ----------
    sigma : float, default=1.0
        Width parameter of the Laplacian kernel.
    """
    
    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self.sigma = sigma
    
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian kernel matrix.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_features, n_samples_X)
            First set of samples.
            
        Y : np.ndarray of shape (n_features, n_samples_Y)
            Second set of samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples_Y, n_samples_X)
            Laplacian kernel matrix.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Compute L1 distances
        n_features, n_samples_X = X.shape
        n_features, n_samples_Y = Y.shape
        
        kernel_matrix = np.zeros((n_samples_Y, n_samples_X))
        
        for i in range(n_samples_Y):
            for j in range(n_samples_X):
                # L1 distance: sum of absolute differences
                distance = np.sum(np.abs(Y[:, i] - X[:, j]))
                kernel_matrix[i, j] = np.exp(-distance / self.sigma)
        
        return kernel_matrix
    
    def __repr__(self) -> str:
        return f"LaplacianKernel(sigma={self.sigma})"


class SigmoidKernel(BaseKernel):
    """
    Sigmoid kernel implementation.
    
    K(x, y) = tanh(gamma * x^T * y + coef0)
    
    Parameters
    ----------
    gamma : float, default=1.0
        Kernel coefficient.
        
    coef0 : float, default=0.0
        Constant term.
    """
    
    def __init__(self, gamma: float = 1.0, coef0: float = 0.0):
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        self.gamma = gamma
        self.coef0 = coef0
    
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid kernel matrix.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_features, n_samples_X)
            First set of samples.
            
        Y : np.ndarray of shape (n_features, n_samples_Y)
            Second set of samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples_Y, n_samples_X)
            Sigmoid kernel matrix.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # K(x, y) = tanh(gamma * x^T * y + coef0)
        linear_kernel = np.dot(X.T, Y)
        kernel_matrix = np.tanh(self.gamma * linear_kernel + self.coef0)
        
        return kernel_matrix
    
    def __repr__(self) -> str:
        return f"SigmoidKernel(gamma={self.gamma}, coef0={self.coef0})"


# Convenience function to create kernel objects
def create_kernel(kernel_type: str, **kwargs) -> BaseKernel:
    """
    Create a kernel object by type name.
    
    Parameters
    ----------
    kernel_type : str
        Type of kernel to create. Options: 'gaussian', 'linear', 'polynomial',
        'laplacian', 'sigmoid'.
        
    **kwargs : dict
        Additional parameters for the kernel.
        
    Returns
    -------
    BaseKernel
        Kernel object of the specified type.
        
    Raises
    ------
    ValueError
        If kernel_type is not recognized.
    """
    kernel_type = kernel_type.lower()
    
    if kernel_type == 'gaussian':
        return GaussianKernel(**kwargs)
    elif kernel_type == 'linear':
        return LinearKernel(**kwargs)
    elif kernel_type == 'polynomial':
        return PolynomialKernel(**kwargs)
    elif kernel_type == 'laplacian':
        return LaplacianKernel(**kwargs)
    elif kernel_type == 'sigmoid':
        return SigmoidKernel(**kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                       f"Supported types: gaussian, linear, polynomial, laplacian, sigmoid")
