"""
Tests for RULSIF detector.
"""

import numpy as np
import pytest
from anomaly_detection.time_series import RULSIFDetector


class TestRULSIFDetector:
    """Test cases for RULSIFDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        
    def test_initialization(self):
        """Test RULSIF detector initialization."""
        detector = RULSIFDetector(alpha=0.5, n_kernels=50)
        assert detector.alpha == 0.5
        assert detector.n_kernels == 50
        assert detector.n_folds == 5
        assert not detector.is_fitted
        
    def test_fit_predict(self):
        """Test fitting and prediction."""
        detector = RULSIFDetector(alpha=0.5, n_kernels=20, random_state=42)
        
        # Split data for reference and test
        split_point = len(self.X) // 2
        reference_data = self.X[:split_point]
        test_data = self.X[split_point:]
        
        # Fit the detector
        detector.fit(reference_data=reference_data, test_data=test_data)
        assert detector.is_fitted
        assert detector.sigma_ is not None
        assert detector.lambda_ is not None
        
        # Predict
        predictions = detector.predict(self.X)
        assert len(predictions) == len(self.X)
        assert np.all(np.isin(predictions, [0, 1]))
        
    def test_score_samples(self):
        """Test anomaly score computation."""
        detector = RULSIFDetector(alpha=0.5, n_kernels=20, random_state=42)
        
        # Split data
        split_point = len(self.X) // 2
        reference_data = self.X[:split_point]
        test_data = self.X[split_point:]
        
        # Fit and score
        detector.fit(reference_data=reference_data, test_data=test_data)
        scores = detector.score_samples(self.X)
        
        assert len(scores) == len(self.X)
        assert np.all(scores >= 0)
        
    def test_detect_changes(self):
        """Test change point detection."""
        detector = RULSIFDetector(alpha=0.5, n_kernels=20, random_state=42)
        
        # Split data
        split_point = len(self.X) // 2
        reference_data = self.X[:split_point]
        test_data = self.X[split_point:]
        
        # Fit and detect changes
        detector.fit(reference_data=reference_data, test_data=test_data)
        changes = detector.detect_changes(self.X)
        
        assert len(changes) == len(self.X)
        assert np.all(np.isin(changes, [0, 1]))
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid alpha
        with pytest.raises(ValueError):
            RULSIFDetector(alpha=-0.1)
            
        # Test invalid n_kernels
        with pytest.raises(ValueError):
            RULSIFDetector(n_kernels=0)
            
        # Test invalid n_folds
        with pytest.raises(ValueError):
            RULSIFDetector(n_folds=0)
            
    def test_unfitted_error(self):
        """Test that unfitted detector raises error."""
        detector = RULSIFDetector()
        
        with pytest.raises(RuntimeError):
            detector.predict(self.X)
            
        with pytest.raises(RuntimeError):
            detector.score_samples(self.X)
            
        with pytest.raises(RuntimeError):
            detector.detect_changes(self.X)


if __name__ == "__main__":
    pytest.main([__file__])
