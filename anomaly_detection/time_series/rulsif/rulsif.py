"""
RULSIF (Relative Unconstrained Least-Squares Importance Fitting) Detector.

This module implements the RULSIF algorithm for change-point detection
in time series data, originally proposed by Liu et al. (2012).

Reference:
Liu S, Yamada M, Collier N, et al. Change-Point Detection in Time-Series Data 
by Relative Density-Ratio Estimation. 2012.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple, Union, Dict, Any
import warnings

from ...core.base import UnsupervisedAnomalyDetector
from .kernels import GaussianKernel, LinearKernel, PolynomialKernel


class RULSIFDetector(UnsupervisedAnomalyDetector):
    """
    RULSIF-based change point detector for time series data.
    
    This detector uses the Relative Unconstrained Least-Squares Importance Fitting
    algorithm to detect changes in the distribution of time series data. It's
    particularly effective for detecting abrupt changes in data streams.
    
    Parameters
    ----------
    alpha : float, default=0.5
        Alpha parameter for relative density ratio estimation.
        Controls the trade-off between reference and test distributions.
        
    sigma : float, optional
        Width parameter for Gaussian kernel. If None, will be automatically
        determined using cross-validation.
        
    lambda_param : float, default=1.5
        Regularization parameter for the optimization problem.
        
    n_kernels : int, default=100
        Number of kernel basis functions to use.
        
    n_folds : int, default=5
        Number of folds for cross-validation.
        
    random_state : int, optional
        Random state for reproducibility.
        
    debug : bool, default=False
        Whether to print debug information during training.
    
    Attributes
    ----------
    is_fitted : bool
        Whether the detector has been fitted to data.
        
    sigma_ : float
        The optimal sigma width parameter found during training.
        
    lambda_ : float
        The optimal regularization parameter found during training.
        
    gaussian_centers_ : np.ndarray
        The Gaussian kernel centers used for feature mapping.
        
    n_features_in_ : int
        Number of features seen during fit.
    
    Examples
    --------
    >>> from anomaly_detection.time_series import RULSIFDetector
    >>> detector = RULSIFDetector(alpha=0.5, n_kernels=50)
    >>> detector.fit(reference_data, test_data)
    >>> scores = detector.score_samples(new_data)
    >>> change_points = detector.detect_changes(new_data, threshold=0.1)
    """
    
    def __init__(self, alpha: float = 0.5, sigma: Optional[float] = None,
                 lambda_param: float = 1.5, n_kernels: int = 100,
                 n_folds: int = 5, random_state: Optional[int] = None,
                 debug: bool = False, **kwargs):
        super().__init__(name="RULSIFDetector", **kwargs)
        
        self.alpha = alpha
        self.sigma = sigma
        self.lambda_param = lambda_param
        self.n_kernels = n_kernels
        self.n_folds = n_folds
        self.random_state = random_state
        self.debug = debug
        
        # Training parameters (set during fit)
        self.sigma_ = None
        self.lambda_ = None
        self.gaussian_centers_ = None
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
    
    def _fit(self, X: np.ndarray, **kwargs) -> 'RULSIFDetector':
        """
        Internal fitting method required by the base class.
        
        This method is called by the parent class fit method.
        """
        # For RULSIF, we need both reference and test data
        # If only X is provided, we'll split it into reference and test
        if 'reference_data' in kwargs and 'test_data' in kwargs:
            reference_data = kwargs['reference_data']
            test_data = kwargs['test_data']
        else:
            # Split the data into reference and test portions
            split_point = X.shape[0] // 2
            reference_data = X[:split_point]
            test_data = X[split_point:]
        
        # Validate data
        if reference_data.shape[1] != test_data.shape[1]:
            raise ValueError("Reference and test data must have the same number of features")
        
        # Generate Gaussian centers
        self.gaussian_centers_ = self._generate_gaussian_centers(reference_data)
        
        # Find optimal parameters via cross-validation
        if self.sigma is None:
            self.sigma_, self.lambda_ = self._compute_optimal_parameters(
                reference_data, test_data, self.gaussian_centers_
            )
        else:
            self.sigma_ = self.sigma
            self.lambda_ = self.lambda_param
        
        if self.debug:
            print(f"Optimal sigma: {self.sigma_:.4f}")
            print(f"Optimal lambda: {self.lambda_:.4f}")
        
        return self
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'RULSIFDetector':
        """
        Fit the RULSIF detector to the data.
        
        Parameters
        ----------
        X : np.ndarray
            Training data. For RULSIF, this can be either:
            - The combined data (will be split into reference and test)
            - Ignored if reference_data and test_data are provided in kwargs
            
        y : np.ndarray, optional
            Ignored (kept for compatibility with base class)
            
        **kwargs : dict
            Additional parameters including:
            - reference_data: Reference period data
            - test_data: Test period data
            
        Returns
        -------
        self : RULSIFDetector
            The fitted detector instance
        """
        # Call parent class fit method which will call our _fit method
        return super().fit(X, y, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict change points in the data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to predict on.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Binary predictions: 1 for change point, 0 for normal.
        """
        self._check_is_fitted()
        scores = self.score_samples(X)
        
        # Use a simple threshold-based approach for now
        # In practice, you might want to use more sophisticated methods
        threshold = np.percentile(scores, 95)  # 95th percentile as threshold
        return (scores > threshold).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute change scores for samples.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to score.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Change scores (higher = more likely to be a change point).
        """
        self._check_is_fitted()
        
        # For RULSIF, we need to compute the density ratio
        # This is a simplified version - in practice, you'd want to use
        # the full RULSIF algorithm
        
        # Compute kernel matrices
        K_X = GaussianKernel(self.sigma_).apply(X, self.gaussian_centers_)
        
        # For now, return a simple distance-based score
        # This should be replaced with the actual RULSIF density ratio computation
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.gaussian_centers_[np.newaxis, :, :], axis=2)
        scores = np.min(distances, axis=1)
        
        return scores
    
    def detect_changes(self, X: np.ndarray, threshold: Optional[float] = None,
                      window_size: int = 10) -> np.ndarray:
        """
        Detect change points in time series data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Time series data to analyze.
            
        threshold : float, optional
            Threshold for change detection. If None, will be automatically determined.
            
        window_size : int, default=10
            Size of the sliding window for change detection.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Binary array indicating change points (1) and normal points (0).
        """
        scores = self.score_samples(X)
        
        if threshold is None:
            threshold = np.percentile(scores, 95)
        
        # Apply sliding window smoothing
        if window_size > 1:
            from scipy.ndimage import uniform_filter1d
            scores = uniform_filter1d(scores, size=window_size)
        
        return (scores > threshold).astype(int)
    
    def _generate_gaussian_centers(self, reference_data: np.ndarray) -> np.ndarray:
        """Generate Gaussian kernel centers from reference data."""
        n_samples, n_features = reference_data.shape
        n_centers = min(self.n_kernels, n_samples)
        
        # Use random sampling for centers
        indices = np.random.choice(n_samples, size=n_centers, replace=False)
        return reference_data[indices]
    
    def _compute_optimal_parameters(self, reference_data: np.ndarray, 
                                  test_data: np.ndarray,
                                  gaussian_centers: np.ndarray) -> Tuple[float, float]:
        """
        Compute optimal sigma and lambda parameters via cross-validation.
        
        This is a simplified version of the cross-validation procedure.
        In practice, you'd want to implement the full RULSIF cross-validation.
        """
        # Generate candidate parameters
        sigma_candidates = self._generate_sigma_candidates(reference_data, test_data)
        lambda_candidates = self._generate_lambda_candidates()
        
        best_score = float('inf')
        best_sigma = sigma_candidates[0]
        best_lambda = lambda_candidates[0]
        
        # Simple grid search (should be replaced with proper cross-validation)
        for sigma in sigma_candidates:
            for lambda_val in lambda_candidates:
                try:
                    score = self._compute_cv_score(reference_data, test_data, 
                                                 gaussian_centers, sigma, lambda_val)
                    if score < best_score:
                        best_score = score
                        best_sigma = sigma
                        best_lambda = lambda_val
                except:
                    continue
        
        return best_sigma, best_lambda
    
    def _generate_sigma_candidates(self, reference_data: np.ndarray, 
                                 test_data: np.ndarray) -> np.ndarray:
        """Generate candidate sigma values for cross-validation."""
        all_data = np.vstack([reference_data, test_data])
        median_distance = self._compute_median_distance(all_data)
        return median_distance * np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    
    def _generate_lambda_candidates(self) -> np.ndarray:
        """Generate candidate lambda values for cross-validation."""
        return 10.0 ** np.array([-3, -2, -1, 0, 1])
    
    def _compute_median_distance(self, data: np.ndarray) -> float:
        """Compute median distance between data points."""
        n_samples = data.shape[0]
        if n_samples > 1000:  # Sample for large datasets
            indices = np.random.choice(n_samples, size=1000, replace=False)
            data = data[indices]
            n_samples = 1000
        
        # Compute pairwise distances efficiently
        G = np.sum(data * data, axis=1)
        Q = np.tile(G[:, np.newaxis], (1, n_samples))
        R = np.tile(G, (n_samples, 1))
        
        distances = Q + R - 2 * np.dot(data, data.T)
        distances = distances - np.tril(distances)
        distances = distances.reshape(n_samples ** 2, 1, order='F')
        
        return np.sqrt(0.5 * np.median(distances[distances > 0]))
    
    def _compute_cv_score(self, reference_data: np.ndarray, test_data: np.ndarray,
                         gaussian_centers: np.ndarray, sigma: float, 
                         lambda_val: float) -> float:
        """Compute cross-validation score for given parameters."""
        try:
            # Compute kernel matrices
            K_ref = GaussianKernel(sigma).apply(reference_data, gaussian_centers)
            K_test = GaussianKernel(sigma).apply(test_data, gaussian_centers)
            
            # Compute H and h matrices
            H = self._compute_H_matrix(K_ref, K_test)
            h = self._compute_h_vector(K_ref)
            
            # Solve for theta
            theta = linalg.solve(H + lambda_val * np.eye(H.shape[0]), h)
            
            # Compute objective function
            score = self._compute_objective(K_ref, K_test, theta)
            return score
        except:
            return float('inf')
    
    def _compute_H_matrix(self, K_ref: np.ndarray, K_test: np.ndarray) -> np.ndarray:
        """Compute H matrix for RULSIF optimization."""
        n_ref = K_ref.shape[1]
        n_test = K_test.shape[1]
        
        H = (self.alpha / n_ref) * np.dot(K_ref, K_ref.T) + \
            ((1.0 - self.alpha) / n_test) * np.dot(K_test, K_test.T)
        
        return H
    
    def _compute_h_vector(self, K_ref: np.ndarray) -> np.ndarray:
        """Compute h vector for RULSIF optimization."""
        return np.mean(K_ref, axis=1)
    
    def _compute_objective(self, K_ref: np.ndarray, K_test: np.ndarray, 
                          theta: np.ndarray) -> float:
        """Compute RULSIF objective function value."""
        g_ref = np.dot(K_ref.T, theta)
        g_test = np.dot(K_test.T, theta)
        
        J = ((self.alpha / 2.0) * np.mean(g_ref ** 2) +
              ((1 - self.alpha) / 2.0) * np.mean(g_test ** 2) -
              np.mean(g_ref))
        
        return J
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update({
            'alpha': self.alpha,
            'sigma': self.sigma,
            'lambda_param': self.lambda_param,
            'n_kernels': self.n_kernels,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'debug': self.debug,
        })
        return params
    
    def set_params(self, **params) -> 'RULSIFDetector':
        """Set the parameters of this estimator."""
        valid_params = ['alpha', 'sigma', 'lambda_param', 'n_kernels', 
                       'n_folds', 'random_state', 'debug']
        
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                warnings.warn(f"Parameter '{key}' is not valid for RULSIFDetector")
        
        return self

