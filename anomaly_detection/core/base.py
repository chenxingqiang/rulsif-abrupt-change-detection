"""
Base classes for anomaly detection algorithms.

This module provides the base classes that all anomaly detection algorithms
should inherit from, ensuring a consistent interface across the library.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BaseAnomalyDetector(ABC):
    """
    Base class for all anomaly detection algorithms.
    
    This abstract base class defines the interface that all anomaly detection
    algorithms must implement. It provides a consistent API for training,
    prediction, and evaluation.
    """
    
    def __init__(self, name: str = None, **kwargs):
        """
        Initialize the base anomaly detector.
        
        Args:
            name: Name of the detector algorithm
            **kwargs: Additional keyword arguments
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.parameters = kwargs.copy()
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'BaseAnomalyDetector':
        """
        Fit the anomaly detection model to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values (optional, for supervised methods)
            **kwargs: Additional fitting parameters
            
        Returns:
            self: The fitted detector instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for new data.
        
        Args:
            X: Data to predict on, shape (n_samples, n_features)
            
        Returns:
            Array of anomaly labels: 1 for anomaly, 0 for normal
        """
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X: Data to score, shape (n_samples, n_features)
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates for samples.
        
        Args:
            X: Data to predict on, shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, 2) with probabilities [normal, anomaly]
        """
        scores = self.score_samples(X)
        # Convert scores to probabilities (simple sigmoid-like transformation)
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict on the same data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values (optional)
            **kwargs: Additional fitting parameters
            
        Returns:
            Array of anomaly labels
        """
        return self.fit(X, y, **kwargs).predict(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator and
                  contained subobjects that are estimators.
                  
        Returns:
            Parameter names mapped to their values
        """
        return self.parameters.copy()
    
    def set_params(self, **params) -> 'BaseAnomalyDetector':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self: The estimator instance
        """
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key] = value
        return self
    
    def _check_is_fitted(self):
        """Check if the estimator is fitted before calling predict or score_samples."""
        if not self.is_fitted:
            raise RuntimeError(
                f"This {self.name} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
    
    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                       reset: bool = False, validate_separately: bool = False,
                       **check_params) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate input data and set or check the `n_features_in_` attribute.
        
        Args:
            X: Training data
            y: Target values
            reset: If True, the `n_features_in_` attribute is deleted
            validate_separately: If False, only validate X
            **check_params: Additional parameters for validation
            
        Returns:
            Validated X and y
        """
        if not hasattr(self, 'n_features_in_') or reset:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this {self.name} "
                f"is expecting {self.n_features_in_} features as input."
            )
        
        if y is not None and not validate_separately:
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y have inconsistent numbers of samples: "
                    f"X has {X.shape[0]} samples, y has {y.shape[0]} samples."
                )
        
        return X, y
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})"
    
    def __str__(self) -> str:
        """String representation of the detector."""
        return self.__repr__()


class UnsupervisedAnomalyDetector(BaseAnomalyDetector):
    """
    Base class for unsupervised anomaly detection algorithms.
    
    These algorithms do not require labeled training data and can detect
    anomalies based on the inherent structure of the data.
    """
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'UnsupervisedAnomalyDetector':
        """
        Fit the unsupervised anomaly detection model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored (kept for compatibility)
            **kwargs: Additional fitting parameters
            
        Returns:
            self: The fitted detector instance
        """
        X, _ = self._validate_data(X, y)
        self._fit(X, **kwargs)
        self.is_fitted = True
        return self
    
    @abstractmethod
    def _fit(self, X: np.ndarray, **kwargs):
        """
        Internal fitting method for unsupervised algorithms.
        
        Args:
            X: Training data
            **kwargs: Additional parameters
        """
        pass


class SupervisedAnomalyDetector(BaseAnomalyDetector):
    """
    Base class for supervised anomaly detection algorithms.
    
    These algorithms require labeled training data to learn the distinction
    between normal and anomalous samples.
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SupervisedAnomalyDetector':
        """
        Fit the supervised anomaly detection model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels: 0 for normal, 1 for anomaly
            **kwargs: Additional fitting parameters
            
        Returns:
            self: The fitted detector instance
        """
        X, y = self._validate_data(X, y)
        if y is None:
            raise ValueError("Supervised anomaly detection requires target labels y")
        
        # Validate that y contains only 0 and 1
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("Target labels must be binary (0 for normal, 1 for anomaly)")
        
        self._fit(X, y, **kwargs)
        self.is_fitted = True
        return self
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Internal fitting method for supervised algorithms.
        
        Args:
            X: Training data
            y: Target labels
            **kwargs: Additional parameters
        """
        pass

