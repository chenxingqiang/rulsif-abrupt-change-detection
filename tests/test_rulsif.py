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

        # Fit the detector - pass X as first argument
        detector.fit(self.X, reference_data=reference_data, test_data=test_data)
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

        # Fit and score - pass X as first argument
        detector.fit(self.X, reference_data=reference_data, test_data=test_data)
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

        # Fit and detect changes - pass X as first argument
        detector.fit(self.X, reference_data=reference_data, test_data=test_data)
        changes = detector.detect_changes(self.X)

        assert len(changes) == len(self.X)
        assert np.all(np.isin(changes, [0, 1]))

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test valid parameters - should not raise
        try:
            detector = RULSIFDetector(alpha=0.1, n_kernels=10, n_folds=3)
            assert detector.alpha == 0.1
            assert detector.n_kernels == 10
            assert detector.n_folds == 3
        except Exception as e:
            pytest.fail(f"Valid parameters should not raise exception: {e}")

        # Test edge cases
        try:
            detector = RULSIFDetector(alpha=0.0, n_kernels=1, n_folds=2)
            assert detector.alpha == 0.0
            assert detector.n_kernels == 1
            assert detector.n_folds == 2
        except Exception as e:
            pytest.fail(f"Edge case parameters should not raise exception: {e}")

    def test_unfitted_error(self):
        """Test that unfitted detector raises error."""
        detector = RULSIFDetector()

        with pytest.raises(RuntimeError):
            detector.predict(self.X)

        with pytest.raises(RuntimeError):
            detector.score_samples(self.X)

        with pytest.raises(RuntimeError):
            detector.detect_changes(self.X)

    def test_data_validation(self):
        """Test data validation."""
        detector = RULSIFDetector()
        
        # Test with 1D data - should raise error due to shape mismatch
        # The base class validation will catch this before our custom validation
        with pytest.raises((ValueError, IndexError)):
            detector.fit(np.array([1, 2, 3]))

    def test_kernel_generation(self):
        """Test kernel center generation."""
        detector = RULSIFDetector(n_kernels=10, random_state=42)
        
        # Generate centers using the correct method name
        centers = detector._generate_gaussian_centers(self.X)
        
        assert centers.shape == (10, self.X.shape[1])
        assert np.all(np.isfinite(centers))

    def test_auto_split_data(self):
        """Test automatic data splitting when reference/test not provided."""
        detector = RULSIFDetector(alpha=0.5, n_kernels=20, random_state=42)
        
        # Fit without providing reference_data and test_data
        detector.fit(self.X)
        
        assert detector.is_fitted
        assert detector.sigma_ is not None
        assert detector.lambda_ is not None
        assert detector.gaussian_centers_ is not None

    def test_custom_threshold(self):
        """Test change detection with custom threshold."""
        detector = RULSIFDetector(alpha=0.5, n_kernels=20, random_state=42)
        
        # Split data
        split_point = len(self.X) // 2
        reference_data = self.X[:split_point]
        test_data = self.X[split_point:]
        
        # Fit
        detector.fit(self.X, reference_data=reference_data, test_data=test_data)
        
        # Test with custom threshold
        threshold = 0.5
        changes = detector.detect_changes(self.X, threshold=threshold)
        
        assert len(changes) == len(self.X)
        assert np.all(np.isin(changes, [0, 1]))


if __name__ == "__main__":
    pytest.main([__file__])
