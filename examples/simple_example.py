#!/usr/bin/env python3
"""
Simple example demonstrating RULSIF anomaly detection.

This script shows how to use the RULSIF detector for basic anomaly detection.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.time_series import RULSIFDetector


def main():
    """Main example function."""
    print("Awesome Anomaly Detection - Simple RULSIF Example")
    print("=" * 50)
    
    # Generate simple synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 2
    
    # Create normal data with some trend
    t = np.linspace(0, 4, n_samples)
    normal_data = np.column_stack([
        np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n_samples),
        np.cos(2 * np.pi * t) + 0.1 * np.random.randn(n_samples)
    ])
    
    # Introduce an anomaly (sudden change)
    anomaly_start = 100
    anomaly_end = 120
    normal_data[anomaly_start:anomaly_end, 0] += 2.0  # Large spike
    
    print(f"Generated data shape: {normal_data.shape}")
    print(f"Anomaly period: samples {anomaly_start} to {anomaly_end}")
    
    # Split data into reference and test periods
    split_point = n_samples // 2
    reference_data = normal_data[:split_point]
    test_data = normal_data[split_point:]
    
    print(f"\nReference data: {reference_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Initialize RULSIF detector
    print("\nInitializing RULSIF detector...")
    detector = RULSIFDetector(
        alpha=0.5,
        n_kernels=30,
        n_folds=3,
        random_state=42
    )
    
    # Fit the detector
    print("Fitting RULSIF detector...")
    detector.fit(reference_data=reference_data, test_data=test_data)
    
    print(f"Optimal sigma: {detector.sigma_:.4f}")
    print(f"Optimal lambda: {detector.lambda_:.4f}")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    anomaly_scores = detector.score_samples(normal_data)
    detected_anomalies = detector.detect_changes(normal_data)
    
    # Analyze results
    print(f"\nResults:")
    print(f"Number of detected anomalies: {np.sum(detected_anomalies)}")
    print(f"Detection rate: {np.sum(detected_anomalies) / len(detected_anomalies):.2%}")
    
    # Check if we detected the known anomaly
    known_anomaly_region = detected_anomalies[anomaly_start:anomaly_end]
    detection_rate_in_region = np.sum(known_anomaly_region) / len(known_anomaly_region)
    print(f"Detection rate in known anomaly region: {detection_rate_in_region:.2%}")
    
    # Show some statistics
    print(f"\nScore statistics:")
    print(f"  Mean score: {np.mean(anomaly_scores):.4f}")
    print(f"  Std score: {np.std(anomaly_scores):.4f}")
    print(f"  Min score: {np.min(anomaly_scores):.4f}")
    print(f"  Max score: {np.max(anomaly_scores):.4f}")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("1. Try different alpha values")
    print("2. Experiment with different kernel parameters")
    print("3. Use your own time series data")
    print("4. Explore the visualization tools")


if __name__ == "__main__":
    main()
