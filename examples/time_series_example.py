#!/usr/bin/env python3
"""
Time Series Anomaly Detection Example

This example demonstrates how to use the RULSIF detector and other
time series anomaly detection methods from the Awesome Anomaly Detection library.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
import os

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.time_series import RULSIFDetector
from anomaly_detection.core.metrics import AnomalyMetrics
from anomaly_detection.core.visualization import AnomalyVisualizer


def generate_synthetic_time_series(n_samples=1000, n_features=3, random_state=42):
    """
    Generate synthetic time series data with known anomalies.
    
    Parameters
    ----------
    n_samples : int
        Number of time points.
        
    n_features : int
        Number of features.
        
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    tuple
        (time_series_data, true_anomaly_labels, anomaly_scores)
    """
    np.random.seed(random_state)
    
    # Generate normal time series with some trend and seasonality
    t = np.linspace(0, 10, n_samples)
    
    # Base signal with trend and seasonality
    base_signal = 2 * np.sin(2 * np.pi * t) + 0.5 * t
    
    # Add noise
    noise = np.random.normal(0, 0.3, (n_samples, n_features))
    
    # Create time series data
    time_series = np.zeros((n_samples, n_features))
    for i in range(n_features):
        time_series[:, i] = base_signal + noise[:, i]
    
    # Introduce anomalies at specific time points
    true_anomalies = np.zeros(n_samples)
    
    # Anomaly 1: Sudden spike around t=2.5
    anomaly1_start = int(0.25 * n_samples)
    anomaly1_end = int(0.3 * n_samples)
    true_anomalies[anomaly1_start:anomaly1_end] = 1
    time_series[anomaly1_start:anomaly1_end, 0] += 3  # Large spike
    
    # Anomaly 2: Gradual drift around t=6
    anomaly2_start = int(0.6 * n_samples)
    anomaly2_end = int(0.7 * n_samples)
    true_anomalies[anomaly2_start:anomaly2_end] = 1
    drift = np.linspace(0, 2, anomaly2_end - anomaly2_start)
    time_series[anomaly2_start:anomaly2_end, 1] += drift
    
    # Anomaly 3: Variance change around t=8
    anomaly3_start = int(0.8 * n_samples)
    anomaly3_end = int(0.9 * n_samples)
    true_anomalies[anomaly3_start:anomaly3_end] = 1
    time_series[anomaly3_start:anomaly3_end, 2] += np.random.normal(0, 1, anomaly3_end - anomaly3_start)
    
    # Generate synthetic anomaly scores (for demonstration)
    anomaly_scores = np.zeros(n_samples)
    anomaly_scores[true_anomalies == 1] = np.random.uniform(0.7, 1.0, np.sum(true_anomalies))
    anomaly_scores[true_anomalies == 0] = np.random.uniform(0.0, 0.3, np.sum(true_anomalies == 0))
    
    return time_series, true_anomalies, anomaly_scores


def demonstrate_rulsif_detector():
    """Demonstrate the RULSIF detector on synthetic data."""
    print("=" * 60)
    print("RULSIF DETECTOR DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data
    print("Generating synthetic time series data...")
    time_series, true_anomalies, true_scores = generate_synthetic_time_series()
    
    print(f"Data shape: {time_series.shape}")
    print(f"Number of true anomalies: {np.sum(true_anomalies)}")
    print(f"Anomaly rate: {np.sum(true_anomalies) / len(true_anomalies):.2%}")
    
    # Split data into reference and test periods
    split_point = len(time_series) // 2
    reference_data = time_series[:split_point]
    test_data = time_series[split_point:]
    
    print(f"\nReference data shape: {reference_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize and fit RULSIF detector
    print("\nInitializing RULSIF detector...")
    rulsif = RULSIFDetector(
        alpha=0.5,
        n_kernels=50,
        n_folds=3,
        random_state=42,
        debug=True
    )
    
    print("Fitting RULSIF detector...")
    rulsif.fit(reference_data=reference_data, test_data=test_data)
    
    print(f"Optimal sigma: {rulsif.sigma_:.4f}")
    print(f"Optimal lambda: {rulsif.lambda_:.4f}")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    detected_anomalies = rulsif.detect_changes(time_series)
    anomaly_scores = rulsif.score_samples(time_series)
    
    print(f"Number of detected anomalies: {np.sum(detected_anomalies)}")
    print(f"Detection rate: {np.sum(detected_anomalies) / len(detected_anomalies):.2%}")
    
    # Evaluate performance
    print("\nEvaluating performance...")
    metrics = AnomalyMetrics.compute_basic_metrics(
        true_anomalies, detected_anomalies, anomaly_scores
    )
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Generate detailed report
    report = AnomalyMetrics.generate_report(metrics, "RULSIF Detector")
    print(report)
    
    return time_series, true_anomalies, detected_anomalies, anomaly_scores, metrics


def demonstrate_visualization(time_series, true_anomalies, detected_anomalies, 
                            anomaly_scores, metrics):
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = AnomalyVisualizer(style='default', figsize=(15, 10))
    
    # Create time series plot with anomalies
    print("Creating time series visualization...")
    fig1 = visualizer.plot_time_series_with_anomalies(
        time_series=time_series,
        anomaly_labels=detected_anomalies,
        anomaly_scores=anomaly_scores,
        title="Synthetic Time Series with RULSIF Detected Anomalies"
    )
    
    # Create anomaly scores distribution plot
    print("Creating score distribution visualization...")
    fig2 = visualizer.plot_anomaly_scores_distribution(
        scores=anomaly_scores,
        labels=true_anomalies,
        title="RULSIF Anomaly Scores Distribution"
    )
    
    # Create confusion matrix plot
    print("Creating confusion matrix visualization...")
    cm = confusion_matrix(true_anomalies, detected_anomalies)
    fig3 = visualizer.plot_confusion_matrix(
        confusion_matrix=cm,
        title="RULSIF Confusion Matrix"
    )
    
    # Create comprehensive dashboard
    print("Creating comprehensive dashboard...")
    fig4 = visualizer.create_dashboard(
        time_series=time_series,
        anomaly_labels=detected_anomalies,
        anomaly_scores=anomaly_scores,
        confusion_matrix=cm,
        title="RULSIF Anomaly Detection Dashboard"
    )
    
    # Save plots
    print("Saving plots...")
    fig1.savefig('rulsif_time_series.png', dpi=300, bbox_inches='tight')
    fig2.savefig('rulsif_score_distribution.png', dpi=300, bbox_inches='tight')
    fig3.savefig('rulsif_confusion_matrix.png', dpi=300, bbox_inches='tight')
    fig4.savefig('rulsif_dashboard.png', dpi=300, bbox_inches='tight')
    
    print("Plots saved as:")
    print("  - rulsif_time_series.png")
    print("  - rulsif_score_distribution.png")
    print("  - rulsif_confusion_matrix.png")
    print("  - rulsif_dashboard.png")
    
    plt.show()


def demonstrate_parameter_sensitivity():
    """Demonstrate the effect of different parameters on RULSIF performance."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Generate data
    time_series, true_anomalies, _ = generate_synthetic_time_series()
    
    # Test different alpha values
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_results = {}
    
    print("Testing different alpha values...")
    for alpha in alpha_values:
        print(f"  Testing alpha = {alpha}")
        
        rulsif = RULSIFDetector(
            alpha=alpha,
            n_kernels=30,
            n_folds=3,
            random_state=42
        )
        
        # Split data
        split_point = len(time_series) // 2
        reference_data = time_series[:split_point]
        test_data = time_series[split_point:]
        
        # Fit and predict
        rulsif.fit(reference_data=reference_data, test_data=test_data)
        detected_anomalies = rulsif.detect_changes(time_series)
        
        # Compute metrics
        metrics = AnomalyMetrics.compute_basic_metrics(true_anomalies, detected_anomalies)
        alpha_results[alpha] = metrics
    
    # Compare results
    print("\nAlpha parameter comparison:")
    print(f"{'Alpha':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    
    for alpha in alpha_values:
        metrics = alpha_results[alpha]
        print(f"{alpha:<8} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    return alpha_results


def main():
    """Main demonstration function."""
    print("Awesome Anomaly Detection - Time Series Example")
    print("=" * 60)
    
    try:
        # Demonstrate RULSIF detector
        results = demonstrate_rulsif_detector()
        time_series, true_anomalies, detected_anomalies, anomaly_scores, metrics = results
        
        # Demonstrate visualization
        demonstrate_visualization(time_series, true_anomalies, detected_anomalies, 
                               anomaly_scores, metrics)
        
        # Demonstrate parameter sensitivity
        alpha_results = demonstrate_parameter_sensitivity()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. RULSIF can effectively detect different types of anomalies in time series")
        print("2. Parameter tuning (especially alpha) affects detection performance")
        print("3. The library provides comprehensive evaluation metrics and visualization tools")
        print("4. The unified API makes it easy to experiment with different algorithms")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

